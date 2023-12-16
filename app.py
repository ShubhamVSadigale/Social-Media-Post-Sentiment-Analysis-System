from flask import Flask, render_template, request, redirect
import pandas as pd
import nltk
from nltk.sentiment import SentimentIntensityAnalyzer

app = Flask(__name__)

# Load the dataset
df = pd.read_csv('Reviews.csv')
df = df.head(500)  # Limit to the first 500 rows for faster processing

# Initialize NLTK
nltk.download('punkt')

# Initialize Sentiment Intensity Analyzer
sia = SentimentIntensityAnalyzer()

@app.route('/')
def index():
    return render_template('index.html')


@app.route('/sentiment_model')
def sentiment_model():
    return render_template('sentiment_model.html')

@app.route('/upload_dataset')
def upload_dataset():
    return render_template('upload_dataset.html')

@app.route('/contact')
def contact():
    return render_template('contact.html')


@app.route('/analyze', methods=['POST'])
def analyze():
    if request.method == 'POST':
        review_text = request.form['review_text']

        # Tokenize and perform sentiment analysis
        tokens = nltk.word_tokenize(review_text)
        tagged = nltk.pos_tag(tokens)
        entities = nltk.chunk.ne_chunk(tagged)
        sentiment_score = sia.polarity_scores(review_text)

        if sentiment_score["neg"] > 0.5:
            return render_template('sentiment_model.html', spost=review_text, review="Negative", sentiment_score=sentiment_score['neg'])
        elif sentiment_score["pos"] > 0.5:
            return render_template('sentiment_model.html', spost=review_text, review="Positive", sentiment_score=sentiment_score['pos'])
        elif sentiment_score["neu"] > 0.5:
            return render_template('sentiment_model.html', spost=review_text, review="Neutral", sentiment_score=sentiment_score['neu'])
        else:
            return render_template('sentiment_model.html', spost=review_text, review="Neutral", sentiment_score=sentiment_score['neu'])




if __name__ == '__main__':
    app.run(debug=True)
