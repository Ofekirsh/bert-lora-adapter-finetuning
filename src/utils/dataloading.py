from datasets import load_dataset
from transformers import BertTokenizer
from torch.utils.data import DataLoader
from transformers import DataCollatorWithPadding
import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"


def load_and_tokenize_cola(model_name="bert-base-uncased", max_length=128):
    """
    Loads and tokenizes the CoLA dataset using the BERT tokenizer.

    Adds input_ids, token_type_ids, and attention_mask for each sentence.
    Returns a DatasetDict with train, validation, and test splits,
    ready for input into a BERT model.
    """
    dataset = load_dataset("glue", "cola")
    tokenizer = BertTokenizer.from_pretrained(model_name)

    def tokenize_fn(example):
        return tokenizer(example["sentence"], truncation=True, padding="max_length", max_length=max_length)

    tokenized_dataset = dataset.map(tokenize_fn, batched=True, remove_columns=["sentence", "idx"])
    return tokenized_dataset, tokenizer


def get_dataloaders(tokenized_dataset, tokenizer, batch_size=16):
    """
    Creates PyTorch DataLoaders for the tokenized CoLA dataset using dynamic padding.

    Returns:
        train_loader: DataLoader for training data with shuffling.
        val_loader: DataLoader for validation data without shuffling.
    """
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
    train_loader = DataLoader(tokenized_dataset["train"], shuffle=True, batch_size=batch_size, collate_fn=data_collator)
    val_loader = DataLoader(tokenized_dataset["validation"], batch_size=batch_size, collate_fn=data_collator)
    return train_loader, val_loader

if __name__ == '__main__':
    load_and_tokenize_cola()