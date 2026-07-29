import pandas as pd
from sklearn.metrics import accuracy_score, classification_report
from transformer_pipeline import analyze_tweet_sentiment
from data_downloader import load_real_data
import visualize  # Using your custom filename!

def run_dataset_evaluation(sample_size=300):
    df = load_real_data()
    
    # 2. Take a random sample to make live training inference fast and responsive
    print(f"\nSampling {sample_size} random rows for rapid model evaluation...")
    sample_df = df.sample(n=sample_size, random_state=42).copy()
    
    predictions = []
    confidences = []
    
    print("Running batch inference through Twitter-roBERTa")
    
    # 3. Loop through the sample rows and get predictions
    for idx, row in sample_df.iterrows():
        result = analyze_tweet_sentiment(row['text'])
        predictions.append(result['prediction'])
        confidences.append(result['confidence'])
        
    sample_df['predicted_sentiment'] = predictions
    sample_df['confidence'] = confidences
    
    # 4. Calculate metrics by comparing the True column vs Predicted column
    print("\n================ EVALUATION METRICS ================")
    accuracy = accuracy_score(sample_df['sentiment'], sample_df['predicted_sentiment'])
    print(f"Overall Batch Accuracy: {accuracy:.2%}")
    
    print("\nDetailed Classification Report:")
    print(classification_report(sample_df['sentiment'], sample_df['predicted_sentiment']))
    
    # Save results to review
    sample_df.to_csv("evaluation_results.csv", index=False)
    print("Saved evaluation spreadsheet matrix to 'evaluation_results.csv'")

if __name__ == "__main__":
    run_dataset_evaluation()