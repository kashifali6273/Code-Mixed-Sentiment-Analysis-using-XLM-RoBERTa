import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# Path to the folder where you extracted the model
model_path = "models/xlm_roberta_sentiment_model"   

# Load tokenizer and model
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForSequenceClassification.from_pretrained(model_path)

# Use GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
model.eval()

# Function to predict sentiment
def predict_sentiment(text):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True).to(device)
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
        pred = torch.argmax(logits, dim=1).item()
    labels = ["negative", "neutral", "positive"]  # make sure the order matches training
    return labels[pred]

# Test some examples
examples = [
    "Mujhe ye app bohat pasand aayi, service amazing thi!",
    "Worst experience ever, bilkul time waste hua.",
    "Kal mausam thoda sa cloudy tha."
]

for text in examples:
    print(f"Text: {text}")
    print(f"Predicted Sentiment: {predict_sentiment(text)}\n")
