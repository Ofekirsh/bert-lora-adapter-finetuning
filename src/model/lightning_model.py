import torch
import torch.nn.functional as F
import pytorch_lightning as pl
from sklearn.metrics import matthews_corrcoef
from sklearn.metrics import accuracy_score


class LightningPEFTModel(pl.LightningModule):
    def __init__(self, model, logger):
        super().__init__()
        self.model = model
        self.experiment_logger = logger
        self.all_val_probs = []
        self.all_val_labels = []

        # Per-epoch accumulators
        self.train_losses = []
        self.val_losses = []
        self.train_preds = []
        self.train_labels = []
        self.val_preds = []
        self.val_labels = []

    def forward(self, input_ids, attention_mask=None, token_type_ids=None, labels=None):
        return self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            labels=labels
        )

    def training_step(self, batch, batch_idx):
        labels = batch.pop("labels")
        outputs = self(**batch, labels=labels)
        loss = outputs.loss
        logits = outputs.logits
        preds = torch.argmax(logits, dim=1)

        # Accumulate for epoch-end logging
        self.train_losses.append(loss.item())
        self.train_preds.append(preds.detach())
        self.train_labels.append(labels.detach())

        return loss

    def validation_step(self, batch, batch_idx):
        labels = batch.pop("labels")
        outputs = self(**batch, labels=labels)
        loss = outputs.loss
        logits = outputs.logits
        probs = torch.softmax(logits, dim=1)[:, 1]
        preds = torch.argmax(logits, dim=1)

        # Accumulate for epoch-end logging
        self.val_losses.append(loss.item())
        self.all_val_probs.extend(probs.detach().cpu().tolist())
        self.all_val_labels.extend(labels.detach().cpu().tolist())
        self.val_preds.extend(preds.detach().cpu().tolist())
        self.val_labels.extend(labels.detach().cpu().tolist())

        return loss

    def configure_optimizers(self):
        return torch.optim.AdamW(self.parameters(), lr=2e-5)

    def on_train_epoch_end(self):
        # Calculate epoch averages and metrics
        avg_train_loss = sum(self.train_losses) / len(self.train_losses) if self.train_losses else 0

        all_preds = torch.cat(self.train_preds)
        all_labels = torch.cat(self.train_labels)
        train_mcc = matthews_corrcoef(all_labels.cpu(), all_preds.cpu())
        train_accuracy = accuracy_score(all_labels.cpu(), all_preds.cpu())

        # Log all training metrics for this epoch
        epoch_metrics = {
            "train_loss": avg_train_loss,
            "train_mcc": train_mcc,
            "train_accuracy": train_accuracy
        }

        # Add validation metrics if available (they should be from on_validation_epoch_end)
        if hasattr(self, '_current_val_metrics'):
            epoch_metrics.update(self._current_val_metrics)
            delattr(self, '_current_val_metrics')

        self.experiment_logger.log(self.current_epoch, epoch_metrics)

        # Clear accumulators
        self.train_losses.clear()
        self.train_preds.clear()
        self.train_labels.clear()

    def on_validation_epoch_end(self):
        if not self.val_losses:
            return

        # Calculate epoch averages and metrics
        avg_val_loss = sum(self.val_losses) / len(self.val_losses)
        val_mcc = matthews_corrcoef(self.val_labels, self.val_preds)
        val_accuracy = accuracy_score(self.val_labels, self.val_preds)

        # Store validation metrics to be logged with training metrics
        self._current_val_metrics = {
            "val_loss": avg_val_loss,
            "val_mcc": val_mcc,
            "val_accuracy": val_accuracy
        }

        # Clear accumulators
        self.val_losses.clear()
        self.val_preds.clear()
        self.val_labels.clear()

    def on_train_end(self):
        self.experiment_logger.accumulate_probs(self.all_val_probs, self.all_val_labels)
        self.experiment_logger.finalize()