import pandas as pd
import re
import os

# Urdu diacritics to remove
urdu_diacritics = re.compile(r"[\u064B-\u0652]")

# Map string labels to numeric
label2id = {"negative": 0, "neutral": 1, "positive": 2}

def clean_text(text):
    # Lowercase English/Roman Urdu part
    text = text.lower()
    # Remove Urdu diacritics
    text = urdu_diacritics.sub("", text)
    # Remove unwanted symbols, keep letters, digits, ?, !, .
    text = re.sub(r"[^a-zA-Z0-9\u0600-\u06FF\s\?\!\.]", "", text)
    # Remove extra spaces
    text = re.sub(r"\s+", " ", text).strip()
    return text

if __name__ == "__main__":
    os.makedirs("data/processed", exist_ok=True)

    combined_df = []

    # Clean splits individually
    for split in ["train", "val", "test"]:
        try:
            df = pd.read_csv(f"data/processed/{split}.csv")
            # Clean text
            df["text"] = df["text"].astype(str).apply(clean_text)
            # Convert string labels to numeric
            df["label"] = df["label"].map(label2id)
            df.to_csv(f"data/processed/{split}.csv", index=False, encoding="utf-8")
            print(f"✅ Cleaned {split}.csv with numeric labels, shape:", df.shape)
            combined_df.append(df)
        except FileNotFoundError:
            print(f"⚠️ {split}.csv not found, skipping.")

    # Save merged dataset for debug
    if combined_df:
        full_df = pd.concat(combined_df, ignore_index=True)
        full_df.to_csv("data/processed/clean_dataset.csv", index=False, encoding="utf-8")
        print("✅ Combined cleaned dataset saved at data/processed/clean_dataset.csv, shape:", full_df.shape)
