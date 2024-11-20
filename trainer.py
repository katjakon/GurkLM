import os
import torch 
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
import random

from dataset import GurkDataset
from modules import FullModel

class Trainer:

    def __init__(self,
                 tokenizer,
                 train_dir,
                 test_dir,
                 model_params,
                 optimizer,
                 optim_params, 
                 n_epochs, 
                 batch_size, 
                 mask_p, 
                 max_len,
                 p_only_masked=0.8
                 ):
        self.train_dir = train_dir
        self.test_dir = test_dir
        self.tokenizer = tokenizer
        self.model_params = model_params
        self.optim_params = optim_params
        self.optimizer = optimizer
        self.n_epochs = n_epochs
        self.batch_size = batch_size
        self.mask_p = mask_p
        self.p_only_masked = p_only_masked
        self.max_len = max_len
        self.model = None
        self.start_epoch = 0
    
    def _get_train_dataloader(self):
        gurk_dataset = GurkDataset(
            data_dir=self.train_dir
        )
        return DataLoader(
            gurk_dataset,
            collate_fn=self._process_batch,
            batch_size=self.batch_size
        )
    
    def _get_test_dataloader(self):
        gurk_dataset = GurkDataset(
            data_dir=self.test_dir
        )
        return DataLoader(
            gurk_dataset,
            collate_fn=self._process_batch,
            batch_size=self.batch_size
        )

    def _create_masks(self, input_ids):
        padding_mask = input_ids == self.tokenizer.pad_token_id
        rand = torch.rand(input_ids.shape) # random probabilities for each token
        p_random = (1 - self.p_only_masked) # percentage of masked tokens that should be replaced with random tokens
        p_unchanged = p_random / 2 # Percentage of masked tokens that are left unchanged
        lm_mask = (rand <= self.mask_p) & (~ padding_mask) # Mask tokens with certain prob but not padding tokens.
        rand[~lm_mask] = torch.inf # Don't consider indices where lm_mask is False
        rand = rand / self.mask_p # Rescale according to masking percentage
        mask_random = (rand <= p_random) & (rand > p_unchanged)
        mask_unchanged = rand <= p_unchanged

        return lm_mask, padding_mask, mask_random, mask_unchanged

    def _process_batch(self, batch):
        tok_output = self.tokenizer(
            batch,
            is_split_into_words=True,
            add_special_tokens=False,
            padding="max_length",
            max_length=self.max_len,
            return_tensors="pt"
            )
        input_ids = tok_output["input_ids"]
        mask_out = self._create_masks(input_ids)
        lm_mask, padding_mask, mask_random, mask_unchanged = mask_out
        masked = input_ids.detach().clone()
        masked[lm_mask] = self.tokenizer.mask_token_id
        masked[mask_random] = random.randrange(1, self.tokenizer.vocab_size) # Sample random tokens
        masked[mask_unchanged] = input_ids[mask_unchanged]
        return {
            "original_sequence": input_ids,
            "masked_sequence": masked,
            "padding_mask": padding_mask,
            "lm_mask": lm_mask
        }
    
    def train(self, out_dir, save_steps=1, start_from_chp=False, chp_path=None):
        dataloader = self._get_train_dataloader()
        if start_from_chp is False:
            self.model = FullModel(
                vocab_size=self.tokenizer.vocab_size,
                max_len=self.max_len, 
                **self.model_params
            )
            self.optimizer = self.optimizer(
                params=self.model.parameters(),
                **self.optim_params
            )
        else:
            if chp_path is None:
                raise ValueError("No path for checkpoint given!")
            last_epoch = self._load_checkpoint(path = chp_path)
            self.start_epoch = last_epoch
        
        loss_function = nn.CrossEntropyLoss(ignore_index=self.tokenizer.pad_token_id)

        for epoch in range(self.start_epoch, self.n_epochs):
            self.model.train()
            print(f"Epoch {epoch+1}")
            step = 0
            for batch in dataloader:
                batch_input_ids = batch["masked_sequence"] # Here randomly some tokens have been replace by [MASK].
                batch_padding_mask = batch["padding_mask"] # Masks which specifies which tokens are [PAD]
                batch_lm_mask = batch["lm_mask"] # Masks which specifies where tokens have been replace by [MASK]
                batch_y_true = batch["original_sequence"] # True tokens without mask
                batch_y_true = torch.flatten(batch_y_true[batch_lm_mask])

                # Make predictions
                batch_out = self.model(
                batch_input_ids,
                key_padding_mask=batch_padding_mask,
                pred_mask=torch.flatten(batch_lm_mask))
                batch_pred = batch_out["masked_preds"]

                # Calculate loss
                loss = loss_function(batch_pred, batch_y_true)
                print(f"Batch loss: {loss}")
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                step += 1

                if step % save_steps == 0:
                    self._save_checkpoint(
                        epoch=epoch,
                        loss=loss, 
                        step=step, 
                        out=out_dir
                    ) 
    
    def _save_checkpoint(self, epoch, loss, step, out):
        print(f"Saving checkpoint to {out} ...", end="")
        out_path = os.path.join(
            out, 
            f"checkpoint-epoch{epoch}-step{step}.pt"
        )
        torch.save({
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'loss': loss,
            'step': step
            }, out_path)
        print("Done")
    

    def _load_checkpoint(self, path):
        print(f"Loading checkpoint from {path}")
        self.model = FullModel(
            vocab_size=self.tokenizer.vocab_size,
            max_len=self.max_len, 
            **self.model_params
        )
        self.optimizer = self.optimizer(
            params=self.model.parameters(),
            **self.optim_params)

        checkpoint = torch.load(path, weights_only=True)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        epoch = checkpoint['epoch']
        return epoch




