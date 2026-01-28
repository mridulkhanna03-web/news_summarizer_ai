"""
News Summarizer with AI - Backend
==================================
A Flask application that fetches live news and provides AI-powered summarization.

Key Concepts for Learning:
1. RSS Feed Parsing - How to fetch news from Google News without API keys
2. NLP Pipeline - Using Transformers for summarization
3. Sentiment Analysis - Using TextBlob for polarity detection
4. TF-IDF - Extracting keywords using term frequency
5. REST API Design - Clean Flask route structure

"""

import nltk
from flask import Flask, request, render_template, jsonify
from newspaper import Article
import feedparser  # For RSS parsing
from textblob import TextBlob  # For sentiment analysis
from sklearn.feature_extraction.text import TfidfVectorizer
import re
import requests  # For resolving redirects

# Download NLTK data (run once)
try:
    nltk.data.find('tokenizers/punkt')
except:
    nltk.download('punkt')

# Initialize Flask app
app = Flask(__name__)

# RSS Feed Parsing
# Google News provides RSS feeds that don't require API keys.
# This is a common pattern used in production news aggregators.

GOOGLE_NEWS_RSS = {
    'top': 'https://news.google.com/rss?hl=en-IN&gl=IN&ceid=IN:en',
    'technology': 'https://news.google.com/rss/topics/CAAqJggKIiBDQkFTRWdvSUwyMHZNRGRqTVhZU0FtVnVHZ0pKVGlnQVAB?hl=en-IN&gl=IN&ceid=IN:en',
    'business': 'https://news.google.com/rss/topics/CAAqJggKIiBDQkFTRWdvSUwyMHZNRGx6TVdZU0FtVnVHZ0pKVGlnQVAB?hl=en-IN&gl=IN&ceid=IN:en',
    'sports': 'https://news.google.com/rss/topics/CAAqJggKIiBDQkFTRWdvSUwyMHZNRFp1ZEdvU0FtVnVHZ0pKVGlnQVAB?hl=en-IN&gl=IN&ceid=IN:en',
    'science': 'https://news.google.com/rss/topics/CAAqJggKIiBDQkFTRWdvSUwyMHZNRFp0Y1RjU0FtVnVHZ0pKVGlnQVAB?hl=en-IN&gl=IN&ceid=IN:en'
}


def resolve_google_news_url(url):
    """
    Resolve Google News redirect URL to actual article URL.
    
    Learning Point: Google News uses redirect URLs. We can follow the redirect
    or parse the actual URL from the link.
    """
    try:
        # Try to follow redirect
        response = requests.head(url, allow_redirects=True, timeout=5)
        return response.url
    except:
        return url


def extract_source_link(entry):
    """
    Extract the actual source article link from Google News RSS entry.
    Google News RSS entries often have the source link in the content.
    """
    # Try to get the link from entry.links
    if hasattr(entry, 'links'):
        for link in entry.links:
            if link.get('rel') == 'alternate' or link.get('type') == 'text/html':
                return link.get('href', entry.link)
    
    # Try to extract from content/summary (Google News includes source URLs)
    content = entry.get('summary', '') or entry.get('content', [{}])[0].get('value', '')
    # Look for href in content
    import re
    urls = re.findall(r'href="(https?://[^"]+)"', content)
    for url in urls:
        if 'news.google.com' not in url:
            return url
    
    return entry.link


def fetch_news(category='top', limit=10):
    """
    Fetch news from Google News RSS feed.
    
    Learning Point: RSS (Really Simple Syndication) is an XML format
    that websites use to publish updates. feedparser library makes it easy to parse.
    """
    try:
        feed = feedparser.parse(GOOGLE_NEWS_RSS.get(category, GOOGLE_NEWS_RSS['top']))
        articles = []
        
        for entry in feed.entries[:limit]:
            # Extract source from title (Google News format: "Title - Source")
            title_parts = entry.title.rsplit(' - ', 1)
            title = title_parts[0] if len(title_parts) > 1 else entry.title
            source = title_parts[1] if len(title_parts) > 1 else 'Unknown'
            
            # Get the actual source article link
            source_link = extract_source_link(entry)
            
            articles.append({
                'title': title,
                'source': source,
                'link': source_link,
                'google_link': entry.link,  # Keep original for reference
                'published': entry.get('published', 'Unknown')
            })
        
        return articles
    except Exception as e:
        print(f"Error fetching news: {e}")
        return []


#  Sentiment Analysis
# ============================================================
# TextBlob provides simple sentiment analysis using a pre-trained model.
# Polarity: -1 (negative) to +1 (positive)
# Subjectivity: 0 (objective) to 1 (subjective)

def analyze_sentiment(text):
    """
    Analyze the sentiment of text.
    
    Learning Point: Sentiment analysis helps understand the emotional tone.
    This is useful for detecting bias in news articles.
    """
    blob = TextBlob(text)
    polarity = blob.sentiment.polarity
    
    if polarity > 0.1:
        sentiment = 'positive'
        emoji = '😊'
    elif polarity < -0.1:
        sentiment = 'negative'
        emoji = '😔'
    else:
        sentiment = 'neutral'
        emoji = '😐'
    
    return {
        'sentiment': sentiment,
        'polarity': round(polarity, 2),
        'subjectivity': round(blob.sentiment.subjectivity, 2),
        'emoji': emoji
    }


#  Keyword Extraction using TF-IDF
# ============================================================
# TF-IDF (Term Frequency-Inverse Document Frequency) identifies
# important words in a document. High TF-IDF = word is important to this doc.

def extract_keywords(text, num_keywords=5):
    """
    Extract keywords using TF-IDF.
    
    Learning Point: TF-IDF is a fundamental NLP technique.
    It's used in search engines, recommendation systems, and text analysis.
    """
    try:
        # Clean text
        text = re.sub(r'[^\w\s]', '', text.lower())
        
        # TF-IDF with single document (we compare against common English words)
        vectorizer = TfidfVectorizer(
            max_features=100,
            stop_words='english',
            ngram_range=(1, 2)  # Include single words and pairs
        )
        
        tfidf_matrix = vectorizer.fit_transform([text])
        feature_names = vectorizer.get_feature_names_out()
        scores = tfidf_matrix.toarray()[0]
        
        # Get top keywords
        keyword_scores = list(zip(feature_names, scores))
        keyword_scores.sort(key=lambda x: x[1], reverse=True)
        
        return [kw[0] for kw in keyword_scores[:num_keywords]]
    except:
        return []


#  Reading Time Calculation
# ============================================================
# Average reading speed is ~200-250 words per minute.

def calculate_reading_time(text):
    """
    Calculate estimated reading time.
    
    Learning Point: This simple feature adds UX value by helping
    users decide whether to read the full article or summary.
    """
    words = len(text.split())
    minutes = round(words / 200)  # 200 WPM average
    return max(1, minutes)  # Minimum 1 minute


# Text Summarization
# ============================================================
# We use a simple extractive approach here to avoid heavy model loading.
# In production, you'd use BART or T5 models.

def simple_summarize(text, num_sentences=3):
    """
    Simple extractive summarization.
    
    Learning Point: Extractive summarization picks important sentences.
    Abstractive (like GPT) generates new sentences. Both have trade-offs.
    """
    sentences = text.replace('!', '.').replace('?', '.').split('.')
    sentences = [s.strip() for s in sentences if len(s.strip()) > 30]
    
    if len(sentences) <= num_sentences:
        return text
    
    # Score sentences by position and length (simple heuristic)
    scored = []
    for i, sent in enumerate(sentences):
        # First and last sentences often contain key info
        position_score = 1.0 if i < 2 or i >= len(sentences) - 2 else 0.5
        length_score = min(len(sent) / 100, 1.0)
        scored.append((sent, position_score + length_score, i))
    
    # Get top sentences and maintain order
    scored.sort(key=lambda x: x[1], reverse=True)
    top_sentences = sorted(scored[:num_sentences], key=lambda x: x[2])
    
    return '. '.join([s[0] for s in top_sentences]) + '.'


# ============================================================
# FLASK ROUTES
# ============================================================

@app.route('/')
def home():
    """Home page with news feed"""
    category = request.args.get('category', 'top')
    news = fetch_news(category, limit=12)
    return render_template('index.html', news=news, category=category)


@app.route('/summarize', methods=['POST'])
def summarize():
    """
    Summarize an article from URL.
    
    This is the main API endpoint that ties everything together.
    """
    try:
        url = request.form.get('url') or request.json.get('url')
        
        if not url:
            return jsonify({'error': 'No URL provided'}), 400
        
        # User-agent to avoid being blocked by websites
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36'
        }
        
        # Resolve Google News redirect URLs to actual article URLs
        if 'news.google.com' in url:
            try:
                resolved = requests.get(url, headers=headers, allow_redirects=True, timeout=10)
                url = resolved.url
                print(f"Resolved URL: {url}")
            except Exception as e:
                print(f"Could not resolve URL: {e}")
        
        # Fetch and parse article with custom config
        from newspaper import Config
        config = Config()
        config.browser_user_agent = headers['User-Agent']
        config.request_timeout = 15
        
        article = Article(url, config=config)
        article.download()
        article.parse()
        
        text = article.text
        
        if not text or len(text) < 100:
            return jsonify({'error': 'Could not extract article content'}), 400
        
        # Process the article
        summary = simple_summarize(text, num_sentences=4)
        sentiment = analyze_sentiment(text)
        keywords = extract_keywords(text, num_keywords=6)
        original_time = calculate_reading_time(text)
        summary_time = calculate_reading_time(summary)
        
        result = {
            'title': article.title,
            'original': text[:1000] + ('...' if len(text) > 1000 else ''),
            'summary': summary,
            'sentiment': sentiment,
            'keywords': keywords,
            'reading_time': {
                'original': original_time,
                'summary': summary_time,
                'saved': original_time - summary_time
            },
            'word_count': {
                'original': len(text.split()),
                'summary': len(summary.split())
            }
        }
        
        # If it's a form submission, render template
        if request.form:
            return render_template('index.html', 
                                   news=fetch_news('top', 12),
                                   category='top',
                                   result=result)
        
        # If it's API call, return JSON
        return jsonify(result)
        
    except Exception as e:
        error_msg = f"Error processing article: {str(e)}"
        if request.form:
            return render_template('index.html',
                                   news=fetch_news('top', 12),
                                   category='top', 
                                   error=error_msg)
        return jsonify({'error': error_msg}), 500


@app.route('/api/news')
def api_news():
    """API endpoint to get news feed"""
    category = request.args.get('category', 'top')
    limit = int(request.args.get('limit', 10))
    news = fetch_news(category, limit)
    return jsonify(news)


# RUN THE APPLICATION
# ============================================================

if __name__ == '__main__':
    print("\n" + "="*60)
    print(" News Summarizer Starting...")
    print("="*60)
    print("\n Learning Points in this project:")
    print("   1. RSS Feed Parsing (feedparser)")
    print("   2. Sentiment Analysis (TextBlob)")
    print("   3. TF-IDF Keyword Extraction (sklearn)")
    print("   4. Web Scraping (newspaper3k)")
    print("   5. Flask REST API Design")
    print("\n" + "="*60)
    print(" Open http://localhost:5000 in your browser")
    print("="*60 + "\n")
    
    app.run(debug=True, port=5000)
