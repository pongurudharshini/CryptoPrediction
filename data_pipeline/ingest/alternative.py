"""
Alternative Data Ingestion.
Fetches external market signals like the Crypto Fear & Greed Index.
"""

import requests
import pandas as pd
import logging

logger = logging.getLogger(__name__)

class FearAndGreedFetcher:
    """Fetches the Crypto Fear & Greed Index from alternative.me."""
    
    API_URL = "https://api.alternative.me/fng/?limit=0"
    
    @classmethod
    def fetch_historical(cls) -> pd.DataFrame:
        """
        Retrieves all available Fear and Greed index data.
        
        Returns:
            pd.DataFrame: Contains 'timestamp' and 'fng_value' (0-100).
        """
        logger.info("Fetching Crypto Fear & Greed Index...")
        response = requests.get(cls.API_URL)
        response.raise_for_status()
        
        data = response.json()["data"]
        df = pd.DataFrame(data)
        
        # Explicitly convert string timestamp to numeric integer first
        df['timestamp'] = pd.to_numeric(df['timestamp'], errors='coerce')
        
        # Now convert Unix epoch seconds to UTC datetime
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='s', utc=True)
        
        # Numerical fear & greed value
        df['fng_value'] = pd.to_numeric(df['value'], errors='coerce')
        
        # Drop any failed parses and sort by date ascending
        df = df.dropna(subset=['timestamp', 'fng_value'])
        return df[['timestamp', 'fng_value']].sort_values('timestamp').reset_index(drop=True)