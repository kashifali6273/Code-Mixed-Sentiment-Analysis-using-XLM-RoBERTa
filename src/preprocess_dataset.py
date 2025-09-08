import pandas as pd
from sklearn.model_selection import train_test_split

# Load dataset
df = pd.read_csv("data/raw/ur_en_generated.csv")

# Split train (80%), temp (20%)
train_df, temp_df = train_test_split(df, test_size=0.2, random_state=42, stratify=df["label"])

# Split temp into val (10%) and test (10%)
val_df, test_df = train_test_split(temp_df, test_size=0.5, random_state=42, stratify=temp_df["label"])

# Save files
train_df.to_csv("data/processed/train.csv", index=False, encoding="utf-8")
val_df.to_csv("data/processed/val.csv", index=False, encoding="utf-8")
test_df.to_csv("data/processed/test.csv", index=False, encoding="utf-8")

print("✅ Dataset split completed")
print("Train:", train_df.shape, "Val:", val_df.shape, "Test:", test_df.shape)
