# src/train.py
import pytorch_lightning as pl
from src.model.peft_model import get_model
from src.model.lightning_model import LightningPEFTModel
from src.utils.logger import ExperimentLogger
from src.utils.seed import set_seed
from src.utils.dataloading import load_and_tokenize_cola, get_dataloaders


def train_one_run(run_id, seed=42):
    set_seed(seed)
    run_name = f"run_{run_id}"

    tokenized_dataset, tokenizer = load_and_tokenize_cola("bert-base-uncased")
    train_loader, val_loader = get_dataloaders(tokenized_dataset, tokenizer)

    peft_model = get_model()
    logger = ExperimentLogger(run_name=run_name)
    model = LightningPEFTModel(peft_model, logger)

    trainer = pl.Trainer(
        max_epochs=1,
        logger=False,
        enable_checkpointing=False,
        enable_model_summary=False,
    )

    trainer.fit(model, train_loader, val_loader)


if __name__ == "__main__":
    for run in range(1):
        train_one_run(run_id=run, seed=42 + run)
