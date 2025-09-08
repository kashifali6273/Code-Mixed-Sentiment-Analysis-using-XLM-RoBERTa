from datasets import load_from_disk
from transformers import AutoTokenizer

# Load tokenized dataset
tokenized_datasets = load_from_disk("data/processed/tokenized")

# Load same tokenizer used earlier
model_name = "xlm-roberta-base"
tokenizer = AutoTokenizer.from_pretrained(model_name)

# Inspect first few samples
for i in range(5):
    item = tokenized_datasets["train"][i]
    input_ids = item["input_ids"]

    # Decode back to text
    decoded_text = tokenizer.decode(input_ids, skip_special_tokens=True)

    print(f"Sample {i+1}")
    print("Original (cleaned):", decoded_text)
    print("Label:", item["labels"])
    print("Token IDs:", input_ids[:20], "...\n")
