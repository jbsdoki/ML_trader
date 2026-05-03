"""Sentiment scoring (FinBERT and related)."""

from .finbert_scorer import FinBERTScorer, score_texts_batch

__all__ = ["FinBERTScorer", "score_texts_batch"]
