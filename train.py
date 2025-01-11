import argparse
import json
import os

from transformers import BertTokenizer
import torch

from gurk.trainer import Trainer

torch.autograd.anomaly_mode.set_detect_anomaly(True)

if __name__ == "__main__":
    parser = argparse.ArgumentParser("Train a GurkModel")
    parser.add_argument("--config", help="A json file which specifies all hyperparameters for a run.")
    args = parser.parse_args()

    # Load configartions
    config_path = args.config
    with open(config_path, "r", encoding="utf-8") as cfg:
        params_dict = json.load(cfg)
    
    print("Load Tokenizer...", end="")
    tokenizer = BertTokenizer.from_pretrained(params_dict["tokenizer-model"])
    print("done!")

    # Set up optimizer
    optimizer = None
    optim_class = params_dict["optimizer"].lower()

    if optim_class == "sgd":
        optimizer = torch.optim.SGD
    if optim_class == "adam":
        optimizer = torch.optim.Adam
    if optim_class == "adamw":
        optimizer = torch.optim.AdamW
    
    if optimizer is None:
        raise ValueError(f"Optimizer class {optim_class} is not supported!")
    
    # Set up Scheduler
    scheduler = params_dict.get("scheduler", None)

    if scheduler == "LinearLR":
        scheduler = torch.optim.lr_scheduler.LinearLR
    elif scheduler is not None:
        raise ValueError(f"Scheduler {scheduler} is not supported!")

    # Set up trainer:
    trainer = Trainer(
    tokenizer=tokenizer,
    # train_dir=params_dict["train_path"],
    # test_dir=params_dict["val_path"],
    train_dl=params_dict["train_path"],
    test_dl=params_dict["val_path"],
    model_params=params_dict["model_params"],
    optim_params=params_dict["optim_params"], 
    optimizer=optimizer,
    n_epochs=params_dict["n_epochs"],
    batch_size=params_dict["batch_size"],
    mask_p=params_dict["masking_p"],
    max_len=params_dict["max_len"],
    scheduler=scheduler,
    scheduler_params=params_dict.get("scheduler_params", {})
)
    
    # Start training.
    os.makedirs(
        params_dict["checkpoint_path"],
        exist_ok=params_dict["start_from_ckp"]
        )

    trainer.train(
    out_dir=params_dict["checkpoint_path"],
    start_from_chp=params_dict["start_from_ckp"],
    save_steps=params_dict["save_steps"],
    chp_path=params_dict["load_ckp"]
)
    




