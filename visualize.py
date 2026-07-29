import matplotlib.pyplot as plt
from wordcloud import WordCloud
from data_downloader import load_real_data
from preprocess import clean_tweet

def generate_sentiment_wordclouds():
    df = load_real_data()
    # 2. Apply your cleaning function to every row in the 'text' column
    df['cleaned_text'] = df['text'].apply(clean_tweet)
    
    # 3. Separate the dataset into positive and negative slices
    positive_tweets = " ".join(df[df['sentiment'] == 'Positive']['cleaned_text'])
    negative_tweets = " ".join(df[df['sentiment'] == 'Negative']['cleaned_text'])
    
    # 4. Configure the layout for the side-by-side plots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 7))
    
    # 5. Generate Positive Word Cloud
    pos_wordcloud = WordCloud(width=800, height=800, background_color='white', 
                              colormap='Greens', max_words=100).generate(positive_tweets)
    ax1.imshow(pos_wordcloud, interpolation='bilinear')
    ax1.set_title('Positive Tweets Key Words', fontsize=16, pad=10)
    ax1.axis('off')
    
    # 6. Generate Negative Word Cloud
    neg_wordcloud = WordCloud(width=800, height=800, background_color='white', 
                              colormap='Reds', max_words=100).generate(negative_tweets)
    ax2.imshow(neg_wordcloud, interpolation='bilinear')
    ax2.set_title('Negative Tweets Key Words', fontsize=16, pad=10)
    ax2.axis('off')
    
    # 7. Save the final chart visual as an image inside your project folder
    output_image = "sentiment_wordclouds.png"
    plt.tight_layout()
    plt.savefig(output_image)
    print(f"Success. Visual saved inside your project folder as '{output_image}'")

if __name__ == "__main__":
    generate_sentiment_wordclouds()