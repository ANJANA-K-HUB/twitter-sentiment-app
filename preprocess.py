import re

def clean_tweet(text):
    # If the text is empty or missing, return an empty string
    if not isinstance(text, str):
        return ""
        
    # 1. Convert text to lowercase so 'HATE' and 'hate' are treated the same
    text = text.lower()
    
    # 2. Remove Twitter handles/usernames (e.g., @realDonaldTrump, @NancyLeeGrahn)
    text = re.sub(r'@\w+', '', text)
    
    # 3. Remove the classic retweet marker 'rt ' at the start of tweets
    text = re.sub(r'\brt\b', '', text)
    
    # 4. Remove URLs/Hyperlinks
    text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
    
    # 5. Remove special characters and numbers (keep only letters and spaces)
    text = re.sub(r'[^a-z\s]', '', text)
    
    # 6. Clean up extra whitespace/spaces left behind from deletions
    text = ' '.join(text.split())
    
    return text

if __name__ == "__main__":
    #  test the script with a messy sample tweet from the debate
    sample_tweet = "RT @DanScavino: #GOPDebate w/ @realDonaldTrump delivered highest ratings!! http://t.co/abc123"
    print("--- Testing Text Cleaning ---")
    print("Original:", sample_tweet)
    print("Cleaned :", clean_tweet(sample_tweet))