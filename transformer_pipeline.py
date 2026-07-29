import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from preprocess import clean_tweet

# Using a robust Twitter-specific model built for 3 classes: Negative, Neutral, Positive
MODEL_NAME = "cardiffnlp/twitter-roberta-base-sentiment-latest"

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME)

def analyze_tweet_sentiment(raw_text):
    # Pre-process the text using custom clean function
    cleaned_text = clean_tweet(raw_text)
    
    if not cleaned_text.strip():
        return {"prediction": "Neutral", "confidence": 1.0}
        
    # Tokenize the text
    inputs = tokenizer(cleaned_text, return_tensors="pt", padding=True, truncation=True)
    
    # Run inference without tracking gradients
    with torch.no_grad():
        outputs = model(**inputs)
    
    logits = outputs.logits
    probabilities = torch.nn.functional.softmax(logits, dim=-1).flatten().tolist()
    
    # This specific model maps index 0 -> Negative, 1 -> Neutral, 2 -> Positive
    labels = ["Negative", "Neutral", "Positive"]
    predicted_index = torch.argmax(logits).item()
    
    return {
        "original_text": raw_text,
        "cleaned_text": cleaned_text,
        "prediction": labels[predicted_index],
        "confidence": probabilities[predicted_index]
    }

if __name__ == "__main__":
    # test it with a classic neutral sentence from a debate environment
    test_tweet = "RT @DebateCentral: The candidates discussed economic policy for forty minutes last night. http://link.com"
    
    print("\n--- Running Multi-Class (3-Label) Inference ---")
    result = analyze_tweet_sentiment(test_tweet)
    
    print(f"Original: {result['original_text']}")
    print(f"Cleaned : {result['cleaned_text']}")
    print(f"Predicted Sentiment: {result['prediction']} (Confidence: {result['confidence']:.2%})")