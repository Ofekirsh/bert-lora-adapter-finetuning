# src/train.py
import pytorch_lightning as pl
from model.model import get_model
from src.model.lightning_model import LightningPEFTModel
from src.utils.logger import ExperimentLogger
from src.utils.seed import set_seed
from src.utils.dataloading import load_and_tokenize_cola, get_dataloaders
import torch


def train_one_run(run_id, seed=42):
    set_seed(seed)
    run_name = f"run_{run_id}"

    tokenized_dataset, tokenizer = load_and_tokenize_cola("bert-base-uncased")
    train_loader, val_loader = get_dataloaders(tokenized_dataset, tokenizer)

    peft_model = get_model()
    logger = ExperimentLogger(run_name=run_name)
    model = LightningPEFTModel(peft_model, logger)

    trainer = pl.Trainer(
        max_epochs=2,
        logger=False,
        enable_checkpointing=False,
        enable_model_summary=False,
        strategy='ddp_find_unused_parameters_true',
    )

    trainer.fit(model, train_loader, val_loader)

    # Run classification examples on 5 sentences from CoLA
    example_sentences = [
        "The boy is playing in the park.",
        "She can sings beautifully.",
        "Dogs bark loudly.",
        "He going to school every day.",
        "This is a test sentence."
    ]
    # Tokenize the sentences
    inputs = tokenizer(
        example_sentences,
        padding=True,
        truncation=True,
        return_tensors="pt"
    )
    # Move inputs to the same device as the model
    device = model.device if hasattr(model, "device") else "cpu"
    inputs = {k: v.to(device) for k, v in inputs.items()}

    # Put model in eval mode and run predictions
    peft_model.eval()
    with torch.no_grad():
        outputs = peft_model(**inputs)
        logits = outputs.logits if hasattr(outputs, "logits") else outputs[0]
        predictions = logits.argmax(dim=-1).cpu().numpy()

    print("Classification results on example sentences:")
    for sent, pred in zip(example_sentences, predictions):
        print(f"Sentence: '{sent}' => Prediction: {pred}")

if __name__ == "__main__":
    for run in range(1):
        train_one_run(run_id=run, seed=42 + run)
