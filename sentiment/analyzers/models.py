"""
Multi-Model Sentiment Analyzer.
Evaluates text using FinBERT, VADER, and Twitter-RoBERTa.
"""

import logging
import torch
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from transformers import pipeline
from typing import Dict, Any

logger = logging.getLogger(__name__)

class SentimentEngine:
    """Combines FinBERT, VADER, and RoBERTa for financial text sentiment scoring."""

    def __init__(self, use_gpu: bool = False):
        self.device = 0 if use_gpu and torch.cuda.is_available() else -1
        
        # 1. Initialize VADER (Fast, rule-based)
        self.vader = SentimentIntensityAnalyzer()
        
        # 2. Initialize FinBERT (Financial domain transformer)
        logger.info("Loading FinBERT model...")
        self.finbert_model_name = "ProsusAI/finbert"
        try:
            self.finbert_pipe = pipeline(
                "text-classification", 
                model=self.finbert_model_name, 
                device=self.device, 
                top_k=None
            )
        except Exception as e:
            logger.error(f"Failed to load FinBERT: {e}")
            self.finbert_pipe = None

        # 3. Initialize RoBERTa (General social media transformer)
        logger.info("Loading Twitter-RoBERTa model...")
        self.roberta_model_name = "cardiffnlp/twitter-roberta-base-sentiment-latest"
        try:
            self.roberta_pipe = pipeline(
                "text-classification", 
                model=self.roberta_model_name, 
                device=self.device, 
                top_k=None
            )
        except Exception as e:
            logger.error(f"Failed to load RoBERTa: {e}")
            self.roberta_pipe = None

    def analyze_vader(self, text: str) -> float:
        """Returns VADER compound score (-1.0 to 1.0)."""
        scores = self.vader.polarity_scores(text)
        return scores['compound']

    def analyze_finbert(self, text: str) -> Dict[str, Any]:
        """
        Runs FinBERT inference.
        Returns mapped score: positive_prob - negative_prob (-1.0 to 1.0).
        """
        if not self.finbert_pipe or not text.strip():
            return {"score": 0.0, "label": "neutral"}
        
        results = self.finbert_pipe(text[:512])[0]
        probs = {item['label'].lower(): item['score'] for item in results}
        
        score = probs.get('positive', 0.0) - probs.get('negative', 0.0)
        dominant_label = max(probs, key=probs.get)
        return {"score": score, "label": dominant_label}

    def analyze_roberta(self, text: str) -> Dict[str, Any]:
        """
        Runs Twitter-RoBERTa inference.
        Returns mapped score: positive_prob - negative_prob (-1.0 to 1.0).
        """
        if not self.roberta_pipe or not text.strip():
            return {"score": 0.0, "label": "neutral"}
        
        results = self.roberta_pipe(text[:512])[0]
        probs = {item['label'].lower(): item['score'] for item in results}
        
        score = probs.get('positive', 0.0) - probs.get('negative', 0.0)
        dominant_label = max(probs, key=probs.get)
        return {"score": score, "label": dominant_label}

    def analyze_all(self, text: str) -> Dict[str, Any]:
        """Runs all three models and returns a consolidated report."""
        vader_score = self.analyze_vader(text)
        finbert_res = self.analyze_finbert(text)
        roberta_res = self.analyze_roberta(text)

        # Ensemble weighted score: FinBERT (50%), RoBERTa (30%), VADER (20%)
        ensemble_score = (finbert_res['score'] * 0.5) + (roberta_res['score'] * 0.3) + (vader_score * 0.2)

        return {
            "text": text,
            "vader_score": vader_score,
            "finbert_score": finbert_res['score'],
            "finbert_label": finbert_res['label'],
            "roberta_score": roberta_res['score'],
            "roberta_label": roberta_res['label'],
            "ensemble_score": ensemble_score
        }