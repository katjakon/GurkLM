import argparse
import json
import os
from statistics import mean

import torch
from transformers import BertTokenizerFast, BertModel, AutoModelForMaskedLM
from sklearn.metrics import f1_score, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
from tqdm import tqdm

from gurk.modules import FullModel
from gurk.ud_data import get_ud_data, get_code_mapping, get_pos_mapping, POS, CODE
from gurk.classifier import MLPClassifier, UpperBoundClassifier, train_model
from gurk.masked_predict import accuracy_at_n, predict_masked_bert, predict_masked_gurk

MLM = "mlm"
GURK = "gurk"

VALID_TASK = {POS, CODE, MLM}
MODEL_TYPE = {GURK, "google-bert/bert-base-multilingual-cased"}

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def get_predictions(model, dl, pad_token_id):
  """Predict all instances in given dataloader for a model in eval model."""
  model.eval()
  all_preds = []
  all_golds = []
  with torch.no_grad():
    for batch in tqdm(dl, desc="Predicting.."):
      inputs = batch["input_ids"].to(device)
      pad_mask  = inputs == pad_token_id
      # Get prediction
      pred = model(inputs, pad_mask)
      pred = pred.flatten(end_dim=1)
      # Get gold labels
      gold_labels = batch["labels"].to(device)
      gold_labels = gold_labels.flatten(end_dim=1).cpu().detach().numpy()
      pred = torch.argmax(pred, dim=1).cpu().detach().numpy()
      all_preds.extend(pred)
      all_golds.extend(gold_labels)
  return all_golds, all_preds

def compute_metrics(y_true, y_pred, labels):
  "Compute metrics F1 macro, micro, weighted and per class."
  f1_per_class = f1_score(y_true=y_true, y_pred=y_pred, labels=labels, average=None, zero_division=0.0)
  f1_macro = f1_score(y_true=y_true, y_pred=y_pred, labels=labels, average="macro", zero_division=0.0)
  f1_micro = f1_score(y_true=y_true, y_pred=y_pred, labels=labels, average="micro", zero_division=0.0)
  f1_weighted = f1_score(y_true=y_true, y_pred=y_pred, labels=labels, average="weighted", zero_division=0.0)
  return {
      "f1_per_class": f1_per_class,
      "f1_macro": f1_macro,
      "f1_micro": f1_micro,
      "f1_weighted": f1_weighted,
  }

def save_metrics(label_mapping, dl, model, task, output, pad_token_id):
    labels = list(label_mapping.keys())
    str_label = [label_mapping[k] for k in labels]
    golds, preds = get_predictions(model, dl, pad_token_id)
    gold_filtered, pred_filtered = [], []

    for g, p in zip(golds, preds):
        if g == -100:
            continue
        gold_filtered.append(g), pred_filtered.append(p)
    metrics = compute_metrics(y_true=gold_filtered, y_pred=pred_filtered, labels=labels)
    metrics["f1_per_class"] = {
       label_mapping[idx]: score
       for idx, score in enumerate(metrics["f1_per_class"])
    }

    golds_str = [label_mapping.get(g, "None") for g in gold_filtered]
    preds_str = [label_mapping.get(p, "None") for p in pred_filtered]

    ConfusionMatrixDisplay.from_predictions(golds_str, preds_str,
                                    labels=str_label, xticks_rotation="vertical")

    metric_path = os.path.join(output, f"{task}-metrics.json")
    with open(metric_path, "w", encoding="utf-8") as metric_file:
       json.dump(metrics, metric_file, indent=4)

    conf_path = os.path.join(output, f"{task}-confusion.png")
    plt.savefig(conf_path, dpi=200, bbox_inches="tight")
    return preds_str


if __name__ == "__main__":
    # Set up argument parser.
    parser = argparse.ArgumentParser("Evaluating a model on validation tasks.")
    parser.add_argument(
       "--config",
       help="Path to model config file.",
       required=False)
    parser.add_argument(
       "--checkpoint", 
       help="Path to model checkpoint.", 
       required=False
       )
    parser.add_argument(
       "--output-path", 
       help="Path to directory where output files will be stored. Existing files will be overwritten.", 
       required=True
    )
    parser.add_argument(
       "--task", 
       help="Validation task to evaluate a model on.", 
       choices=[POS, CODE, MLM], 
       required=True
    )
    parser.add_argument(
       "--ud-data", 
       help="Name of UD dataset.", 
       required=False, 
       default="qtd_sagt"
    )
    parser.add_argument(
       "--lr",
       help="Learning rate for mlp classifier.",
       default=0.001, 
       type=float, 
       required=False
    )
    parser.add_argument(
       "--epochs", 
       help="Number of epochs to train MLP classifier",
       required=False, 
       default=5, 
       type=int
    )
    parser.add_argument(
       "--batch-size", 
       help="Batch size for training of MLP classifier.", 
       required=False, 
       default=16
    )
    parser.add_argument(
       "--data-tokenizer",
       help="Tokenizer to use on data.",
       required=False, 
       default=None
    )
    parser.add_argument(
       "--model-type",
       help="Either use 'gurk' for custom model or choose a pre-trained BERT model from huggingface.",
       required=False, 
       default=GURK
    )

    args = parser.parse_args()
    config_path = args.config #"configs/run-02-params.json"
    checkpoint_path = args.checkpoint #"gurklm\jan11_fullrun_epoch4_step450954.pt"
    data_tokenizer_name = args.data_tokenizer #"bert-base-cased"
    batch_size = args.batch_size
    ud_ds = args.ud_data
    task =  args.task # "upos" lang-code 
    lr = args.lr
    n_epochs = args.epochs
    output_path = args.output_path # "results"
    model_type = args.model_type

    if task not in VALID_TASK:
        raise ValueError("Invalid validation task. Choose from {}".format(", ".join(VALID_TASK)))
    
    if model_type not in MODEL_TYPE:
       raise ValueError("Invalid model type. Choose from: {}".format(", ".join(MODEL_TYPE)))
    
    no_cp_no_cfg = (config_path is None or checkpoint_path is None) # No config or checkpoint is given!
    if model_type == GURK and no_cp_no_cfg:
       raise ValueError("Need checkpoint and config file for custom GURK model!")
    
    if not os.path.exists(output_path):
        os.mkdir(output_path)

    if model_type == GURK:
        with open(config_path, "r") as config_f: # Load file with hyperparameter configuration.
            params = json.load(config_f)
        tokenizer = BertTokenizerFast.from_pretrained(params["tokenizer-model"])
    else:
       tokenizer = BertTokenizerFast.from_pretrained(model_type)

    if data_tokenizer_name is None:
        if model_type == GURK:
            data_tokenizer_name = params["tokenizer-model"]
        else: 
           data_tokenizer_name = model_type

    tokenizer_data = BertTokenizerFast.from_pretrained(data_tokenizer_name)

    if model_type == GURK:
        model = FullModel(
                vocab_size=tokenizer.vocab_size,
                max_len=params["max_len"],
                **params["model_params"]
            )
        
        print("Loading model...")
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        if task == MLM:
            model = AutoModelForMaskedLM.from_pretrained(model_type).to(device)
        else: 
            model = BertModel.from_pretrained(model_type).to(device)
    print(f"Using device {device}")
    model.to(device)

    print("Loading UD data...")
    train_dl, val_dl, test_dl = get_ud_data(ud_ds, batch_size=batch_size, tokenizer=tokenizer_data, label_type=task)

    if task == MLM:
        accs3 = []
        accs5 = []
        accs1 = []
        model.eval()
        for line in tqdm(test_dl.dataset, desc="Predicting masked tokens.."):
            text = line["text"]
            if model_type == GURK:
               max_len = params["max_len"]
            else:
               max_len = tokenizer.model_max_length
            mask_token_id = tokenizer.mask_token_id
            input_ids =  tokenizer_data(
               text,
               add_special_tokens=False,
               return_tensors="pt",
               max_length=max_len,
               truncation=True)["input_ids"]
            if model_type == GURK:
               pred = predict_masked_gurk(input_ids, model, mask_token_id)
            else:
               pred = predict_masked_bert(input_ids, model, mask_token_id)
            
            acc1 = accuracy_at_n(y_true=input_ids, y_pred=pred, n=1)
            acc3 = accuracy_at_n(y_true=input_ids, y_pred=pred, n=3)
            acc5 = accuracy_at_n(y_true=input_ids, y_pred=pred, n=5)
            accs3.append(acc3.item())
            accs1.append(acc1.item())
            accs5.append(acc5.item())
        
        acc_mlm = {
           "accuracy@1": mean(accs1), 
           "accuracy@3": mean(accs3),
           "accuracy@5": mean(accs5)
        }
        mlm_path = os.path.join(output_path, f"{task}-accuracy.json")
        with open(mlm_path, "w", encoding="utf-8") as mlm_file:
           json.dump(acc_mlm, mlm_file)

    else:
        if task == POS:
            mapping = get_pos_mapping(train_dl.dataset)
        elif task == CODE:
            mapping = get_code_mapping()
        num_classes = len(mapping)

        print("Number of possible classes:", num_classes)
        if model_type == GURK:
            clf_nn = MLPClassifier(num_classes, model, dim=params["model_params"]["model_dim"])
        else:
           dim = model.config.hidden_size
           clf_nn = UpperBoundClassifier(num_classes, model, dim=dim)
           
        # Train model.
        logs = train_model(
            clf_nn,
            train_dl,
            val_dl,
            lr=lr,
            n_epochs=n_epochs,
            pad_token_id=tokenizer.pad_token_id
            )
        
        # Save logs of train model.
        log_path = os.path.join(output_path, f"{task}-logs.json")
        with open(log_path, "w", encoding="utf-8") as log_file:
           json.dump(logs, log_file, indent=4)
        
        
        pred_str = save_metrics(
        label_mapping=mapping,
        dl=test_dl, 
        model=clf_nn, 
        task=task,
        output=output_path, 
        pad_token_id=tokenizer.pad_token_id
        )
        
        

