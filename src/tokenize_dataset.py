from datasets import load_dataset
from transformers import AutoTokenizer

# Load processed CSVs
dataset = load_dataset("csv", data_files={
    "train": "data/processed/train.csv",
    "validation": "data/processed/val.csv",
    "test": "data/processed/test.csv"
})

# Choose multilingual tokenizer
model_name = "xlm-roberta-base"
tokenizer = AutoTokenizer.from_pretrained(model_name)

# Tokenization function
def tokenize_function(examples):
    return tokenizer(
        examples["text"],
        padding="max_length",
        truncation=True,
        max_length=64
    )

# Apply tokenization
tokenized_datasets = dataset.map(tokenize_function, batched=True)
tokenized_datasets = tokenized_datasets.remove_columns(["text"])
tokenized_datasets = tokenized_datasets.rename_column("label", "labels")
tokenized_datasets.set_format("torch")

# Save locally for reuse
tokenized_datasets.save_to_disk("data/processed/tokenized")

print("✅ Tokenization complete")
print(tokenized_datasets)
