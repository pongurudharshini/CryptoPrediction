"""
Integration test for the Sentiment Analysis Pipeline.
Validates text cleaning, FinBERT/VADER models, and news fetching.
"""

import sys
from pathlib import Path

# Add project root to sys.path automatically
ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from sentiment.analyzers.preprocessor import TextCleaner
from sentiment.analyzers.models import SentimentEngine
from sentiment.fetchers.news import NewsFetcher

def main():
    print("=== 1. Testing Text Cleaning ===")
    sample_text = "HODL! $BTC is going to the moon 🚀 check out https://example.com @elonmusk bearish dip!"
    cleaned = TextCleaner.clean(sample_text)
    print(f"Raw:     {sample_text}")
    print(f"Cleaned: {cleaned}\n")

    print("=== 2. Initializing Sentiment Engine ===")
    engine = SentimentEngine(use_gpu=False) # Set True if you have CUDA setup
    
    print("\n=== 3. Comparing Models on Sample Sentences ===")
    sentences = [
        "Bitcoin surges past $90,000 as institutional adoption reaches all-time high.",
        "SEC launches investigation into crypto exchange due to severe regulatory violations.",
        "Ethereum price remains flat in quiet trading session."
    ]
    
    for text in sentences:
        clean_text = TextCleaner.clean(text)
        res = engine.analyze_all(clean_text)
        print(f"\nText: \"{text}\"")
        print(f"  └─ VADER Score:    {res['vader_score']:.3f}")
        print(f"  └─ FinBERT Score:  {res['finbert_score']:.3f} ({res['finbert_label']})")
        print(f"  └─ RoBERTa Score:  {res['roberta_score']:.3f} ({res['roberta_label']})")
        print(f"  └─ Ensemble Score: {res['ensemble_score']:.3f}")

    print("\n=== 4. Fetching Live News & Scoring ===")
    news_df = NewsFetcher.fetch_google_news("Bitcoin", limit=3)
    if not news_df.empty:
        print("Live News Sentiment Sample:")
        for idx, row in news_df.iterrows():
            clean_title = TextCleaner.clean(row['title'])
            scores = engine.analyze_all(clean_title)
            print(f"  [{idx+1}] {row['title'][:60]}...")
            print(f"      Ensemble Sentiment Score: {scores['ensemble_score']:.3f}")

if __name__ == "__main__":
    main()