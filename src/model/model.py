import json
from pathlib import Path
from transformers import BertForSequenceClassification
from peft import get_peft_model, LoraConfig, PrefixTuningConfig, TaskType
from adapters import BertAdapterModel
from adapters.configuration import AdapterPlusConfig
            

# Load config
config_path = Path(__file__).resolve().parents[2] / "config" / "config.json"
with open(config_path, "r") as f:
    CONFIG = json.load(f)


def get_model():
    """
    Load a BERT-based model and apply PEFT (LoRA, PrefixTuning) or adapters library (Houlsby) based on config.
    """
    method = CONFIG["method"]
    
    if method in ["lora", "prefix_tuning"]:
        # Use PEFT library
        model = BertForSequenceClassification.from_pretrained(
            CONFIG["model_name"],
            num_labels=CONFIG["num_labels"]
        )
        
        if method == "lora":
            task_type = TaskType[CONFIG["lora"]["task_type"]]
            config = LoraConfig(
                r=CONFIG["lora"]["r"],
                lora_alpha=CONFIG["lora"]["alpha"],
                lora_dropout=CONFIG["lora"]["dropout"],
                bias=CONFIG["lora"]["bias"],
                task_type=task_type
            )
        elif method == "prefix_tuning":
            task_type = TaskType[CONFIG["prefix_tuning"]["task_type"]]
            config = PrefixTuningConfig(
                num_virtual_tokens=CONFIG["prefix_tuning"]["num_virtual_tokens"],
                task_type=task_type
            )
        
        return get_peft_model(model, config)
    
    elif method == "houlsby":
        # Use adapters library for Houlsby bottleneck adapters
        
        # Load the base model for sequence classification
        model = BertAdapterModel.from_pretrained(
            CONFIG["model_name"]
        )
        
        # Add a classification head
        model.add_classification_head(
            "cola",
            num_labels=CONFIG["num_labels"]
        )
        
        # Add Houlsby adapter configuration
        adapter_config = AdapterPlusConfig.load(
            "houlsby",
            leave_out=CONFIG["houlsby"]["leave_out"],
        )
        
        # Add and activate the adapter
        adapter_name = "cola_adapter"
        model.add_adapter(adapter_name, config=adapter_config)
        model.set_active_adapters(adapter_name)
        
        # Set the active head
        model.active_head = "cola"
        
        return model
            

    
    else:
        raise ValueError(f"Unsupported method: {method}. Supported methods: 'lora', 'prefix_tuning', 'houlsby'")
