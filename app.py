import streamlit as st
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from preprocess import clean_tweet

# 1. Page Configuration for a professional look
st.set_page_config(
    page_title="GOP Debate Sentiment Analyzer",
    page_icon="🎬",
    layout="centered"
)

# 2. Cache the model loading so it stays in memory and runs lightning-fast
@st.cache_resource
def load_custom_model():
    model_path = "./best_model"
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForSequenceClassification.from_pretrained(model_path)
    return tokenizer, model

# Load the model and tokenizer
try:
    tokenizer, model = load_custom_model()
    model_loaded = True
except Exception as e:
    st.error(f"Error loading model from './best_model'. Make sure files are in the right folder! Details: {e}")
    model_loaded = False

# 3. Design the UI Interface
st.title("🎬 2015 GOP Debate Sentiment Analyzer")
st.markdown("""
This full-stack Capstone application deploys a **Fine-Tuned DistilBERT Transformer** model. 
It interprets context-specific sentiment (Positive, Negative, or Neutral) from political debate tweets.
""")

st.write("---")

# User Input Text Area
user_tweet = st.text_area(
    "Enter a tweet to analyze live:", 
    placeholder="Type something like: 'The candidates discussed economic policy for forty minutes last night...'"
)

# 4. Run Live Prediction on Button Click
if st.button("Analyze Sentiment", type="primary"):
    if not model_loaded:
        st.error("Model is not loaded. Cannot run prediction.")
    elif not user_tweet.strip():
        st.warning("Please enter some text before analyzing!")
    else:
        with st.spinner("Analyzing text tokens and computing weights..."):
            # A. Clean the input using your preprocessing logic
            cleaned_text = clean_tweet(user_tweet)
            
            # B. Tokenize the cleaned text
            inputs = tokenizer(cleaned_text, return_tensors="pt", padding=True, truncation=True, max_length=128)
            
            # C. Inference (Turn off gradients for speed)
            with torch.no_grad():
                outputs = model(**inputs)
            
            # D. Process probabilities
            logits = outputs.logits
            probabilities = torch.nn.functional.softmax(logits, dim=-1).flatten().tolist()
            
            # Our fine-tuning labels mapping: 0 -> Negative, 1 -> Neutral, 2 -> Positive
            labels = ["Negative", "Neutral", "Positive"]
            predicted_index = torch.argmax(logits).item()
            prediction = labels[predicted_index]
            confidence = probabilities[predicted_index]
            
            # 5. Display the Visual Results
            st.write("### 📊 Prediction Result")
            
            # Color coding the output card metric for impact
            if prediction == "Positive":
                st.success(f"**Predicted Sentiment:** {prediction} (Confidence: {confidence:.2%})")
            elif prediction == "Negative":
                st.error(f"**Predicted Sentiment:** {prediction} (Confidence: {confidence:.2%})")
            else:
                st.info(f"**Predicted Sentiment:** {prediction} (Confidence: {confidence:.2%})")
                
            # Expandable debug info to show off your pipeline metrics during evaluation
            with st.expander("🔍 See NLP Pipeline Details"):
                st.write(f"**Original Text:** `{user_tweet}`")
                st.write(f"**Cleaned Text Matrix Input:** `{cleaned_text}`")
                st.write("**Raw Class Metrics Logits:**")
                for lbl, prob in zip(labels, probabilities):
                    st.write(f"- {lbl}: {prob:.2%}")