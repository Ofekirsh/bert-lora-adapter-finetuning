import json
from transformers import BertForSequenceClassification
from peft import get_peft_model, LoraConfig, PrefixTuningConfig, TaskType
from adapters import BertAdapterModel
from adapters.configuration import AdapterPlusConfig, DoubleSeqBnConfig


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


# Count total parameters
def count_total_parameters(model):
    return sum(p.numel() for p in model.parameters())


def get_model(conf):
    """
    Load a BERT-based model and apply PEFT (LoRA, PrefixTuning) or adapters library (Houlsby) based on config.

    Args:
        config (dict): Configuration dictionary containing model settings

    Returns:
        model: Configured model with the specified method applied
    """

    method = conf["method"]

    if method in ["lora", "prefix_tuning"]:
        # Use PEFT library
        model = BertForSequenceClassification.from_pretrained(
            conf["model_name"],
            num_labels=conf["num_labels"]
        )

        if method == "lora":
            task_type = TaskType[conf["lora"]["task_type"]]
            config = LoraConfig(
                r=conf["lora"]["r"],
                lora_alpha=conf["lora"]["alpha"],
                lora_dropout=conf["lora"]["dropout"],
                bias=conf["lora"]["bias"],
                task_type=task_type
            )
        elif method == "prefix_tuning":
            task_type = TaskType[conf["prefix_tuning"]["task_type"]]
            config = PrefixTuningConfig(
                num_virtual_tokens=conf["prefix_tuning"]["num_virtual_tokens"],
                task_type=task_type
            )

        return get_peft_model(model, config)

    elif method == "adapter+":

        # Load the base model for sequence classification
        model = BertAdapterModel.from_pretrained(
            conf["model_name"]
        )
        for param in model.bert.parameters():
            param.requires_grad = False

        # Add a classification head
        model.add_classification_head(
            "cola",
            num_labels=conf["num_labels"]
        )

        # Add Houlsby adapter configuration
        adapter_config = AdapterPlusConfig(

            leave_out=conf["adapter+"]["leave_out"],
            reduction_factor=conf["adapter+"]["reduction_factor"]
        )

        with open("adapter_architecture.json", "w") as f:
            json.dump(adapter_config.to_dict(), f, indent=2)

        # Add and activate the adapter
        adapter_name = "cola_adapter"
        model.add_adapter(adapter_name, config=adapter_config)
        model.set_active_adapters(adapter_name)

        # Set the active head
        model.active_head = "cola"

        return model

    elif method == "houlsby":
        # Use adapters library for Houlsby bottleneck adapters

        # Load the base model for sequence classification
        model = BertAdapterModel.from_pretrained(
            conf["model_name"]
        )
        for param in model.bert.parameters():
            param.requires_grad = False

        # Add a classification head
        model.add_classification_head(
            "cola",
            num_labels=conf["num_labels"]
        )

        # Add Houlsby adapter configuration
        adapter_config = DoubleSeqBnConfig(
            leave_out=conf["houlsby"]["leave_out"],
            reduction_factor=conf["houlsby"]["reduction_factor"]
        )

        # Add and activate the adapter
        adapter_name = "cola_adapter"
        model.add_adapter(adapter_name, config=adapter_config)
        model.set_active_adapters(adapter_name)

        with open("adapter_architecture.json", "w") as f:
            json.dump(adapter_config.to_dict(), f, indent=2)

        # Set the active head
        model.active_head = "cola"

        return model

    else:
        raise ValueError(f"Unsupported method: {method}. Supported methods: 'lora', 'prefix_tuning', 'houlsby'")
