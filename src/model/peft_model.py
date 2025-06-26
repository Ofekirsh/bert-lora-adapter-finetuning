import json
from pathlib import Path
from transformers import BertForSequenceClassification
from peft import get_peft_model, LoraConfig, PrefixTuningConfig, TaskType

# Load config
config_path = Path(__file__).resolve().parents[2] / "config" / "peft_config.json"
with open(config_path, "r") as f:
    PEFT_HYPERPARAMS = json.load(f)


def get_model():
    """
    Load a BERT-based model and apply PEFT (LoRA or PrefixTuning) based on config.
    """
    method = PEFT_HYPERPARAMS["method"]
    model = BertForSequenceClassification.from_pretrained(
        PEFT_HYPERPARAMS["model_name"],
        num_labels=PEFT_HYPERPARAMS["num_labels"]
    )

    task_type = TaskType[PEFT_HYPERPARAMS[method]["task_type"]]

    if method == "lora":
        config = LoraConfig(
            r=PEFT_HYPERPARAMS["lora"]["r"],
            lora_alpha=PEFT_HYPERPARAMS["lora"]["alpha"],
            lora_dropout=PEFT_HYPERPARAMS["lora"]["dropout"],
            bias=PEFT_HYPERPARAMS["lora"]["bias"],
            task_type=task_type
        )
    elif method == "adapter":
        config = PrefixTuningConfig(
            num_virtual_tokens=PEFT_HYPERPARAMS["adapter"]["num_virtual_tokens"],
            task_type=task_type
        )
    else:
        raise ValueError(f"Unsupported method: {method}")

    return get_peft_model(model, config)
