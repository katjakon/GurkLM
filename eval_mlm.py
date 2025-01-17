import argparse
import json
import random

from transformers import  BertTokenizer
import torch
from tqdm import tqdm

from gurk.modules import FullModel

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

def create_masks(batch, tokenizer, seed=None, mask_p=0.15):
    input_ids = batch["input_ids"]
    padding_mask = input_ids == tokenizer.pad_token_id
    if seed is not None:
        torch.manual_seed(seed)
    rand = torch.rand(input_ids.shape, device=DEVICE) # random probabilities for each token
    p_random = (1 - mask_p) # percentage of masked tokens that should be replaced with random tokens
    p_unchanged = p_random / 2 # Percentage of masked tokens that are left unchanged
    lm_mask = (rand <= mask_p) & (~ padding_mask) # Mask tokens with certain prob but not padding tokens.
    if not lm_mask.any(): # There is no tokens masked!
        row_i, col_i = (~ padding_mask).nonzero(as_tuple=True) # Get indices of tokens which are not PAD
        rand_index = torch.randint(high=row_i.shape[0], size=(1,)) # Choose random one.
        lm_mask[row_i[rand_index], col_i[rand_index]] = True # Make sure at least one token is masked.
        assert lm_mask.any()
    rand[~lm_mask] = torch.inf # Don't consider indices where lm_mask is False
    rand = rand / 0.15 # Rescale according to masking percentage
    mask_random = (rand <= p_random) & (rand > p_unchanged)
    mask_unchanged = rand <= p_unchanged

    return lm_mask, padding_mask, mask_random, mask_unchanged

def process_batch(batch, tokenizer, seed=None):
    batch["input_ids"] = batch["input_ids"].to(DEVICE)
    mask_out = create_masks(batch, tokenizer=tokenizer, seed=seed)
    lm_mask, padding_mask, mask_random, mask_unchanged = mask_out
    masked = batch["input_ids"].detach().clone()
    masked[lm_mask] = tokenizer.mask_token_id
    if seed is not None:
        random.seed(seed)
    masked[mask_random] = random.randrange(1, tokenizer.vocab_size) # Sample random tokens
    masked[mask_unchanged] = batch["input_ids"][mask_unchanged]
    return {
        "original_sequence": batch["input_ids"],
        "masked_sequence": masked,
        "padding_mask": padding_mask,
        "lm_mask": lm_mask
    }

def accuracy_at_n(y_true, y_pred, n=3):
  top_n = torch.topk(y_pred, k=n)[1]
  y_true = y_true.view(-1, 1).to(DEVICE)
  correct = (y_true == top_n).any(1)
  return torch.sum(correct) / correct.size(0)

if __name__ == "__main__":
    parser = argparse.ArgumentParser("Evaluate a model for masked language modelling.")
    parser.add_argument("--checkpoint", help="Path where model checkpoint is stored.")
    parser.add_argument("--config", help="Path to json file where model configurations is stored.")
    parser.add_argument("--dataloader", help="Path to saved dataloader.")
    parser.add_argument("--seed", default=12)
    args = parser.parse_args()
    checkpoint_path = args.checkpoint # "gurklm/jan11-epoch0-step420000.pt"
    config_path = args.config #"configs/run-02-params.json"
    dataloader_path = args.dataloader #"dl_val.pt"
    seed = args.seed

    # Load config with model parameters.
    with open(config_path, "r") as config_f:
        params = json.load(config_f)

    # Load associated tokenizer model from hf.
    print("Loading tokenizer...", end="")
    tokenizer = BertTokenizer.from_pretrained(params["tokenizer-model"])
    print("Done!")

    # Init model and load model weights from checkpoint.
    print("Loading model from checkpoint...", end="")
    model = FullModel(
        vocab_size=tokenizer.vocab_size,
        max_len=params["max_len"],
        **params["model_params"]
    )
    checkpoint = torch.load(checkpoint_path, weights_only=True, map_location=DEVICE)
    model.load_state_dict(checkpoint['model_state_dict'])
    print("Done!")

    # Load dataloader with data for mlm
    dl = torch.load(dataloader_path, weights_only=False)

    model.eval()
    acc_scores = []
    for batch in tqdm(dl):
        batch = process_batch(batch, tokenizer, seed)
        batch_input_ids = batch["masked_sequence"].to(DEVICE) # Here randomly some tokens have been replace by [MASK].
        batch_padding_mask = batch["padding_mask"].to(DEVICE) # Masks which specifies which tokens are [PAD]
        batch_lm_mask = batch["lm_mask"].to(DEVICE) # Masks which specifies where tokens have been replace by [MASK]
        batch_y_true = batch["original_sequence"].to(DEVICE) # True tokens without mask
        batch_y_true = torch.flatten(batch_y_true[batch_lm_mask])

        with torch.no_grad():
            batch_out = model(
                batch_input_ids,
                key_padding_mask=batch_padding_mask,
                pred_mask=torch.flatten(batch_lm_mask)
            )
        batch_pred = batch_out["masked_preds"]
        acc = accuracy_at_n(y_true=batch_y_true, y_pred=batch_pred)
    
    acc_overall = sum(acc_scores) / len(acc_scores)
    print(f"Accuracy@3: {acc_overall}")




