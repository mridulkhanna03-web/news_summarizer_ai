#  AI News Summarizer

![Python](https://img.shields.io/badge/Python-3.9+-blue.svg)
![Flask](https://img.shields.io/badge/Flask-2.0+-green.svg)
![NLP](https://img.shields.io/badge/AI-NLP-orange.svg)

An intelligent news aggregator that fetches trending stories from Google News and uses Natural Language Processing (NLP) to generate concise, readable summaries. This project demonstrates backend API development, sentiment analysis, and modern UI design.

##  Features

- ** Live News Feed**: Aggregates trending news from Google News RSS feeds without requiring paid APIs.
- ** AI Summarization**: Uses NLP to extract key sentences and generate concise summaries.
- ** Sentiment Analysis**: Analyzes article tone (Positive/Negative/Neutral) using TextBlob.
- ** Keyword Extraction**: Automatically identifies key topics using TF-IDF algorithms.
- ** Smart Reading Time**: Calculates estimated reading time and shows "Time Saved" stats.
- ** Modern UI**: Fully responsive Dark Mode interface with Glassmorphism design.

##  Tech Stack

- **Backend:** Python, Flask
- **NLP:** TextBlob, NLTK, Scikit-learn (TF-IDF), Newspaper3k
- **Frontend:** HTML5, CSS3 (Glassmorphism), JavaScript
- **Data:** RSS Feed Parsing (XML)

##  How to Run locally

1. **Clone the repository**
   ```bash
   git clone https://github.com/YOUR-USERNAME/news-summarizer-ai.git
   cd news-summarizer-ai
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Run the application**
   ```bash
   python app.py
   ```

4. **Open in Browser**
   Navigate to `http://localhost:5000`


