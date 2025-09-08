import streamlit as st
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import pandas as pd
import torch.nn.functional as F

# ----------------------------
# Model path
# ----------------------------
model_path = r"F:\multilingual-llm-codemixed\models\xlm_roberta_sentiment_model"

# ----------------------------
# Device setup
# ----------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ----------------------------
# Cache model loading
# ----------------------------
@st.cache_resource
def load_model(model_path):
    tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
    model = AutoModelForSequenceClassification.from_pretrained(
        model_path,
        local_files_only=True,
        trust_remote_code=False,
        torch_dtype=torch.float32
    )
    model.to(device)
    model.eval()
    return tokenizer, model

tokenizer, model = load_model(model_path)

# ----------------------------
# Prediction function with confidence
# ----------------------------
def predict_sentiment(text):
    encoding = tokenizer(
        text,
        padding='max_length',
        truncation=True,
        max_length=128,
        return_tensors='pt'
    ).to(device)

    with torch.no_grad():
        outputs = model(**encoding)
        logits = outputs.logits
        probs = F.softmax(logits, dim=-1)
        pred_label_id = torch.argmax(probs, dim=-1).item()
        confidence = probs[0, pred_label_id].item()

    id2label = model.config.id2label
    return id2label[pred_label_id], confidence

# ----------------------------
# Streamlit UI
# ----------------------------
st.title("Code-Mixed Sentiment Analysis Demo")
st.markdown("Enter Urdu/English or Roman-Urdu text and get the predicted sentiment with confidence score.")

# --- Single text input ---
st.subheader("Single Text Prediction")
user_input = st.text_area("Enter text here:")

if st.button("Predict Sentiment"):
    if user_input.strip() != "":
        result, conf = predict_sentiment(user_input)
        color = "green" if result=="positive" else "red" if result=="negative" else "gray"
        st.markdown(f"<h3 style='color:{color}'>{result.upper()} ({conf*100:.2f}%)</h3>", unsafe_allow_html=True)
    else:
        st.error("Please enter some text.")

# --- Manual multiple-text input ---
st.markdown("---")
st.subheader("Multiple Texts (Paste manually)")
multi_text_input = st.text_area("Enter multiple texts, one per line:")

if st.button("Predict Multiple"):
    if multi_text_input.strip() != "":
        texts = [line.strip() for line in multi_text_input.strip().split("\n") if line.strip()]
        results = [predict_sentiment(txt) for txt in texts]
        for txt, (pred, conf) in zip(texts, results):
            color = "green" if pred=="positive" else "red" if pred=="negative" else "gray"
            st.markdown(f"**Text:** {txt}\n**Prediction:** <span style='color:{color}'>{pred.upper()} ({conf*100:.2f}%)</span>", unsafe_allow_html=True)
    else:
        st.error("Please enter some texts.")

# --- Bulk CSV upload ---
st.markdown("---")
st.subheader("Bulk Prediction via CSV Upload")
st.markdown("Upload a CSV file with a column named `text` containing sentences to predict.")

uploaded_file = st.file_uploader("Upload CSV", type=["csv"])

if uploaded_file is not None:
    try:
        df = pd.read_csv(uploaded_file)
        if 'text' not in df.columns:
            st.error("CSV must have a column named 'text'.")
        else:
            df[['prediction', 'confidence']] = df['text'].apply(lambda x: pd.Series(predict_sentiment(x)))

            # Color-coded display
            def colorize(pred):
                return f"<span style='color:green'>{pred}</span>" if pred=="positive" else \
                       f"<span style='color:red'>{pred}</span>" if pred=="negative" else \
                       f"<span style='color:gray'>{pred}</span>"

            st.markdown("### Predictions")
            for i, row in df.iterrows():
                st.markdown(f"**Text:** {row['text']}  \n**Prediction:** {colorize(row['prediction'])} ({row['confidence']*100:.2f}%)", unsafe_allow_html=True)

            # Download results
            csv_result = df.to_csv(index=False).encode('utf-8')
            st.download_button(
                label="Download Predictions CSV",
                data=csv_result,
                file_name="predictions_with_confidence.csv",
                mime="text/csv"
            )
    except Exception as e:
        st.error(f"Error reading CSV: {e}")

# --- Quick test block ---
st.markdown("---")
st.subheader("Quick Test Multiple Texts")

test_texts = [
    "Main bazar gaya aur sab theek tha.",
    "Kal mausam thoda sa cloudy tha.",
    "Office me sab log kaam kar rahe the.",
    "Mujhe library me kuch books mil gayi.",
    "School me students morning assembly me khade the.",
    "Worst experience ever, bilkul time waste hua.",
    "Mujhe ye app bohat pasand aayi, service amazing thi!"
]

for txt in test_texts:
    pred, conf = predict_sentiment(txt)
    color = "green" if pred=="positive" else "red" if pred=="negative" else "gray"
    st.markdown(f"**Text:** {txt}\n**Prediction:** <span style='color:{color}'>{pred.upper()} ({conf*100:.2f}%)</span>", unsafe_allow_html=True)
