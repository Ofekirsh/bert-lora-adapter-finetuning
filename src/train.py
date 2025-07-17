# src/train.py
import random
import pytorch_lightning as pl
from model.model import get_model
from src.model.lightning_model import LightningPEFTModel
from src.utils.logger import ExperimentLogger
from src.utils.seed import set_seed
from src.utils.dataloading import load_and_tokenize_cola, get_dataloaders
import json
import time
from pathlib import Path
from huggingface_hub import login
login()


def count_parameters(model):
    """Count trainable parameters in the model"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def train_one_run(config_path, run_id=0, seed=42, train_loader=None, val_loader=None):
    set_seed(seed)
    # Load config
    with open(config_path, 'r') as f:
        config = json.load(f)

    # Set global config for model loading
    import model.model as model_module
    model_module.CONFIG = config

    # Extract method and create run name
    method = config.get("method", "unknown")
    config_name = Path(config_path).stem
    run_name = f"{config_name}_run_{run_id}"

    print(f"\n{'=' * 60}")
    print(f"STARTING RUN: {run_name}")
    print(f"Method: {method}")
    print(f"Config: {config_path}")
    print(f"{'=' * 60}")

    if train_loader is None or val_loader is None:
        raise ValueError("Dataloaders must be passed to train_one_run()")

    peft_model = get_model(config)
    trainable_params = count_parameters(peft_model)
    print(f"Trainable parameters: {trainable_params:,}")
    logger = ExperimentLogger(run_name=run_name)
    model = LightningPEFTModel(peft_model, logger)

    trainer = pl.Trainer(
        max_epochs=8,
        logger=False,
        enable_checkpointing=False,
        enable_model_summary=False,
        accelerator="gpu",
        devices=1

    )

    trainer.fit(model, train_loader, val_loader)
    # Get validation accuracy
    trainer.validate(model, val_loader)
    val_accuracy = getattr(model, 'val_accuracy', None)
    val_mcc = getattr(model, 'val_mcc', None)


    # Prepare results
    results = {
        "run_id": run_id,
        "config_path": config_path,
        "config_name": config_name,
        "method": method,
        "trainable_parameters": trainable_params,
        "validation_accuracy": float(val_accuracy) if val_accuracy is not None else None,
        "validation_mcc": float(val_mcc) if val_mcc is not None else None,
        "seed": seed,
        "config": config,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "model_config": str(peft_model.config),
        "model_string": str(peft_model),
    }

    # Add method-specific details
    if method == "lora":
        results["lora_rank"] = config["lora"]["r"]
        results["lora_alpha"] = config["lora"]["alpha"]
    elif method in ["houlsby", "adapter+"]:
        results["reduction_factor"] = config[method]["reduction_factor"]
        results["leave_out"] = config[method]["leave_out"]

    # Save results
    results_dir = Path("results")
    results_dir.mkdir(exist_ok=True)
    results_file = results_dir / f"{config_name}_seed{seed}_results.json"

    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\nResults saved to: {results_file}")
    return results


if __name__ == "__main__":

    tokenized_dataset, tokenizer = load_and_tokenize_cola("bert-base-uncased")
    train_loader, val_loader = get_dataloaders(tokenized_dataset, tokenizer)

    # Set how many config files to run (change this number)
    k = 7

    config_dir = Path("../config")
    config_files = list(config_dir.glob("config_*.json"))
    print(f"Found {len(config_files)} config files total")
    print(f"Running {k} random config files:")
    # Take k random files
    selected_configs = random.sample(config_files, k)

    for config_file in selected_configs:
        print(f"  - {config_file.name}")

    print(f"\nStarting experiments...")

    for i, config_file in enumerate(selected_configs):
        try:
            print(f"\n[{i + 1}/{k}] Running: {config_file.name}")
            results = train_one_run(str(config_file), run_id=i, seed=42, train_loader=train_loader, val_loader=val_loader)
            print(f"Completed: {config_file.stem}")
        except Exception as e:
            print(f"ERROR in {config_file}: {str(e)}")

    print(f"\nFinished running {len(selected_configs)} experiments")
