import os

from datasets import Dataset
import torch 
import torch.nn as nn
from torch.utils.data import DataLoader
import random
import time

from .dataset import GurkDataset
from .data_management import GurkFolderDataset
from .modules import FullModel

# silence tqdm
from tqdm import tqdm
from functools import partialmethod
tqdm.__init__ = partialmethod(tqdm.__init__, disable=True)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class Trainer:

    def __init__(self,
                 tokenizer,
                 train_path,
                 val_path,
                 model_params,
                 optimizer,
                 optim_params, 
                 n_epochs, 
                 batch_size, 
                 mask_p, 
                 max_len,
                 p_only_masked=0.8,
                scheduler = None,
                scheduler_params = None):
        self.train_dataset = train_path
        self.val_dataset = val_path
        self.tokenizer = tokenizer
        self.model_params = model_params
        self.optim_params = optim_params
        self.optimizer = optimizer
        self.n_epochs = n_epochs
        self.batch_size = batch_size
        self.mask_p = mask_p
        self.p_only_masked = p_only_masked
        self.max_len = max_len
        self.scheduler = scheduler
        self.scheduler_params = scheduler_params if scheduler_params is not None else {}
        self.model = None
        self.start_epoch = 0
        self.loss_fn = nn.CrossEntropyLoss(ignore_index=self.tokenizer.pad_token_id)

    def _get_dataloader(self, dataset_path):
        """Load a dataset and create a DataLoader object."""
        # Load dataset
        gurk_dataset = GurkFolderDataset(dataset_path).get_dataset()
        # Prepare DataLoader
        gurk_dataset.set_format(type="torch")
        gurk_dataset = gurk_dataset.map(self._process_batch, batched=True, batch_size=self.batch_size)
        return DataLoader(
            gurk_dataset,
            batch_size=self.batch_size
        )

    def _get_train_dataloader(self):
        """Create the data loader for the training data."""
        print("Loading training data...")
        train_loader = self._get_dataloader(self.train_dataset)
        print("-> Done!")
        return train_loader

    def _get_val_dataloader(self):
        """Create the data loader for the validation data."""
        print("Loading validation data...")
        val_loader = self._get_dataloader(self.val_dataset)
        print("-> Done!")
        return val_loader

    def _create_masks(self, input_ids):
        """Create the masks for masked token prediction."""
        padding_mask = input_ids == self.tokenizer.pad_token_id
        rand = torch.rand(input_ids.shape, device=DEVICE)  # Random probabilities for each token
        p_random = (1 - self.p_only_masked)  # Percentage of masked tokens that should be replaced with random tokens
        p_unchanged = p_random / 2  # Percentage of masked tokens that are left unchanged
        lm_mask = (rand <= self.mask_p) & (~padding_mask)  # Mask tokens with certain prob but not padding tokens.

        if not lm_mask.any(): # There is no tokens masked!
            row_i, col_i = (~ padding_mask).nonzero(as_tuple=True)  # Get indices of tokens which are not PAD
            rand_index = torch.randint(high=row_i.shape[0], size=(1,))  # Choose random one.
            lm_mask[row_i[rand_index], col_i[rand_index]] = True  # Make sure at least one token is masked.
            assert lm_mask.any()

        rand[~lm_mask] = torch.inf # Don't consider indices where lm_mask is False
        rand = rand / self.mask_p # Rescale according to masking percentage
        mask_random = (rand <= p_random) & (rand > p_unchanged)
        mask_unchanged = rand <= p_unchanged

        return lm_mask, padding_mask, mask_random, mask_unchanged

    def _process_batch(self, batch):
        """Mask batches for masked token prediction."""
        input_ids = batch["input_ids"].to(DEVICE)  # Shift IDs to device
        # Create masks
        mask_out = self._create_masks(input_ids)
        lm_mask, padding_mask, mask_random, mask_unchanged = mask_out
        # Mask IDs
        masked = input_ids.detach().clone()
        masked[lm_mask] = self.tokenizer.mask_token_id
        masked[mask_random] = random.randrange(1, self.tokenizer.vocab_size) #Sample random tokens
        masked[mask_unchanged] = input_ids[mask_unchanged]
        return {
            "original_sequence": input_ids,
            "masked_sequence": masked,
            "padding_mask": padding_mask,
            "lm_mask": lm_mask
        }

    def validate(self, dataloader, n=3):
        """Validation loop: Get current results on validation data.

        Args:
            dataloader: the validation dataloader
            n: n for the accuracy_at_n calculation, default is 3

        Returns:
            Average validation loss over validation data
            Average accuracy at n over validation data
        """
        # Set to eval mode.
        self.model.eval()
        # Set up counters for loss, accuracy and number of batches
        sum_loss = 0
        sum_acc = 0
        n_batch = 0
        # Validation loop
        for batch in dataloader:
            n_batch += 1
            with torch.no_grad():
                batch = self._process_batch(batch)
                batch_input_ids = batch["masked_sequence"].to(DEVICE) # Here randomly some tokens have been replace by [MASK].
                batch_padding_mask = batch["padding_mask"].to(DEVICE) # Masks which specifies which tokens are [PAD]
                batch_lm_mask = batch["lm_mask"].to(DEVICE) # Masks which specifies where tokens have been replace by [MASK]
                batch_y_true = batch["original_sequence"].to(DEVICE) # True tokens without mask
                batch_y_true = torch.flatten(batch_y_true[batch_lm_mask])

                # Make predictions
                batch_out = self.model(
                batch_input_ids,
                key_padding_mask=batch_padding_mask,
                pred_mask=torch.flatten(batch_lm_mask))
                batch_pred = batch_out["masked_preds"]

                val_loss = self.loss_fn(batch_pred, batch_y_true).item()
                sum_loss += val_loss

                acc = self.accuracy_at_n(y_pred=batch_pred, y_true=batch_y_true, n=n)
                sum_acc += acc

        self.model.train() # Set back to train mode.
        # Calculate average loss & accuracy.
        val_avg_loss = sum_loss/n_batch
        val_avg_acc = sum_acc/n_batch
        return val_avg_loss, val_avg_acc

    def accuracy_at_n(self, y_pred, y_true, n=3):
        """Calculate the accuracy at n, default is 3.

        This means that if the correct prediction for a token is among the
        top-n predictions of the model, the given position is counted as correct.
        """
        top_n = torch.topk(y_pred, k=n)[1] # Get n token indices where score is the highest. 
        correct = (top_n ==  y_true.unsqueeze(1)).any(1) # Any correct predictions in top n tokens?
        acc = torch.sum(correct) / correct.shape[0] # Calculaze accuracy at n.
        return acc

    def train(self, out_dir, save_steps=1, start_from_chp=False, chp_path=None):
        """The training loop."""
        # Get the dataloaders for training and validation data
        val_loader = self._get_val_dataloader()
        dataloader = self._get_train_dataloader()
        # Set up step counter
        start_step = 0
        # Start from step 1 in epoch
        if start_from_chp is False:
            if chp_path is None:
                # Initialize model if no checkpoint path has been provided
                self.model = FullModel(
                    vocab_size=self.tokenizer.vocab_size,
                    max_len=self.max_len, 
                    **self.model_params
                )
                self.model.to(DEVICE)  # Shift model to device
                # Set up optimizer
                self.optimizer = self.optimizer(
                    params=self.model.parameters(),
                    **self.optim_params
                )
                # Set up scheduler
                if self.scheduler is not None:
                    self.scheduler = self.scheduler(
                        self.optimizer,
                        **self.scheduler_params
                    )
            else:
                # Load checkpoint
                _ = self._load_checkpoint(path=chp_path)
        else:
            # In this case, the checkpoint path needs to be provided
            if chp_path is None:
                raise ValueError("No path for checkpoint given!")
            last_epoch, start_step = self._load_checkpoint(path=chp_path)
            print(f"Last epoch {last_epoch} start_step {start_step}")
            self.start_epoch = last_epoch

        # Training loop
        loss = None
        print("The range:", range(self.start_epoch, self.n_epochs))
        for epoch in range(self.start_epoch, self.n_epochs):

            torch.cuda.empty_cache()
            print(f"Epoch {epoch}")
            self.model.train()
            step = 0

            for batch in dataloader:
                step += 1
                start_t = time.time()
                
                if step <= start_step:  # Start from right batch
                    if step == start_step:
                        start_step = 0
                    continue
                
                batch = self._process_batch(batch)
                batch_y_true = torch.flatten(batch["original_sequence"][batch["lm_mask"]])

                # Make predictions
                batch = self.model(
                    batch["masked_sequence"],
                    key_padding_mask=batch["padding_mask"],
                    pred_mask=torch.flatten(batch["lm_mask"])
                )

                # Calculate loss
                loss = self.loss_fn(batch["masked_preds"], batch_y_true)
                loss.backward()
                self.optimizer.step()
                self.optimizer.zero_grad(set_to_none=True)
                # Calculate accuracy at 3
                acc_at3 = self.accuracy_at_n(batch["masked_preds"], batch_y_true, n=3)
                print(f"Batch loss on step {step}: {loss:.2f}\t Batch accuracy@3: {acc_at3:.2f}", end="\t")
                # Calculate time it took to process the batch
                end_t = time.time()
                batch_time = end_t - start_t
                print(f"ms/batch: {batch_time*1000.0:.2f}")

                if step % save_steps == 0:
                    print(f"Step {step}: Validate...")
                    val_loss, val_acc = self.validate(val_loader) # Do validation
                    print(f"Validation loss: {val_loss:.2f}\t Validation accuracy@3 {val_acc:.2f}")
                     # Back to train mode
                    self._save_checkpoint(
                        epoch=epoch,
                        loss=loss, 
                        step=step, 
                        out=out_dir
                    ) 
            
            # Don't trigger validation before reaching right moment
            if loss is None:
                continue

            if self.scheduler is not None:
                print(f"Current Learning Rate: {self.scheduler.get_lr()[0]:.2e}")
                self.scheduler.step()

            # Do validation after every epoch.
            print(f"Finished epoch {epoch}: Validate...")
            val_loss, val_acc = self.validate(val_loader)
            self._save_checkpoint(
                epoch=epoch,
                loss=loss, 
                step=step, 
                out=out_dir
            )
            print(f"Validation loss: {val_loss:.2f}\t Validation accuracy@3 {val_acc:.2f}")
    
    def _save_checkpoint(self, epoch, loss, step, out):
        """Save a checkpoint to a file."""
        # Create filepath
        file_name = f"checkpoint-epoch{epoch}-step{step}.pt"
        print(f"Saving checkpoint to {out}/{file_name} ...", end="")
        out_path = os.path.join(out, file_name)
        if self.scheduler is not None:
            scheduler_state_dict = self.scheduler.state_dict()
        else:
            scheduler_state_dict = None
        torch.save({
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': scheduler_state_dict,
            'loss': loss,
            'step': step
            }, out_path)
        print("Done")
    
    def _load_checkpoint(self, path):
        """Load a model from a saved checkpoint."""
        print(f"Loading checkpoint from {path}")
        # Initialize model, optimizer and scheduler
        self.model = FullModel(
            vocab_size=self.tokenizer.vocab_size,
            max_len=self.max_len, 
            **self.model_params
        )
        self.model.to(DEVICE)
        self.optimizer = self.optimizer(
            params=self.model.parameters(),
            **self.optim_params)
        
        if self.scheduler is not None:
            self.scheduler = self.scheduler(
                optimizer=self.optimizer,
                **self.scheduler_params
            )

        # Load checkpoint
        checkpoint = torch.load(path, weights_only=True, map_location=DEVICE)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        if self.scheduler is not None:
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        epoch = checkpoint['epoch']
        step = checkpoint["step"]
        return epoch, step

