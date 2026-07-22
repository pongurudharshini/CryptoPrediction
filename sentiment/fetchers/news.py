"""
Crypto News Fetcher.
Scrapes news headlines using RSS feeds and Google News.
"""

import requests
from bs4 import BeautifulSoup
import pandas as pd
import logging
from datetime import datetime

logger = logging.getLogger(__name__)

class NewsFetcher:
    """Fetches real-time financial news headlines."""

    @staticmethod
    def fetch_google_news(query: str = "Bitcoin", limit: int = 20) -> pd.DataFrame:
        """
        Scrapes Google News RSS for a given query (e.g., 'Bitcoin', 'Ethereum').
        """
        url = f"https://news.google.com/rss/search?q={query}+crypto+when:1d&hl=en-US&gl=US&ceid=US:en"
        logger.info(f"Fetching news RSS from: {url}")
        
        try:
            response = requests.get(url, timeout=10)
            response.raise_for_status()
            
            soup = BeautifulSoup(response.content, features="xml")
            items = soup.find_all("item")[:limit]
            
            news_data = []
            for item in items:
                title = item.find("title").text if item.find("title") else ""
                pub_date = item.find("pubDate").text if item.find("pubDate") else ""
                
                news_data.append({
                    "title": title,
                    "published_at": pub_date,
                    "source": "Google News"
                })
                
            return pd.DataFrame(news_data)
        except Exception as e:
            logger.error(f"Error fetching news: {e}")
            return pd.DataFrame()