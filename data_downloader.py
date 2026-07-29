import pandas as pd

def load_real_data():
    
    df = pd.read_csv("twitter.csv")
    
    clean_df = df[['text', 'sentiment']].copy()
    
    print(f"Dataset loaded successfully. Total tweets found: {len(clean_df)}")
    
    print("\nReal Sentiment Class Distribution:")
    print(clean_df['sentiment'].value_counts())
    
    return clean_df

if __name__ == "__main__":
    tweet_df = load_real_data()
    print("\nFirst 5 rows of your real dataset:")
    print(tweet_df.head())