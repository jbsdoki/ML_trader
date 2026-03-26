"""
FinBERT inference (base model, no fine-tuning) for financial sentiment.

Uses ``ProsusAI/finbert`` by default (same as ``notebooks/FinBert.ipynb``).
Returns per-text probabilities and a scalar ``score = prob_pos - prob_neg`` in [-1, 1]
for alignment with ``storage.sentiment_repo.upsert_article_sentiment`` (caller adds
``dedupe_key``, ``model_id``, ``text_hash``).

References
----------
- Model card: https://huggingface.co/ProsusAI/finbert
- ``transformers``: https://huggingface.co/docs/transformers/index
- PyTorch no_grad: https://pytorch.org/docs/stable/generated/torch.no_grad.html
"""

from __future__ import annotations

import logging
import os
from typing import Any

logger = logging.getLogger(__name__)

_DEFAULT_MODEL = "ProsusAI/finbert"


def _optional_imports():
    try:
        import torch
        from transformers import AutoModelForSequenceClassification, AutoTokenizer
    except ImportError as e:  # pragma: no cover
        raise ImportError(
            "FinBERT requires torch and transformers. "
            "Install with: pip install torch transformers"
        ) from e
    return torch, AutoModelForSequenceClassification, AutoTokenizer


def _resolve_device(torch: Any, device: str | None) -> Any:
    if device and device.lower() != "auto":
        return torch.device(device)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _probs_from_logits(
    id2label: dict[int, str],
    logits: Any,
) -> tuple[float | None, float | None, float | None]:
    """
    Map softmax probabilities to positive / negative / neutral floats.

    Keys are matched case-insensitively on label text (FinBERT uses
    ``positive``, ``negative``, ``neutral``).
    """
    import torch

    probs = torch.softmax(logits, dim=-1)[0]
    name_to_p: dict[str, float] = {}
    for idx, p in enumerate(probs.tolist()):
        label = str(id2label.get(idx, f"label_{idx}")).lower().strip()
        name_to_p[label] = float(p)

    def _get(*aliases: str) -> float | None:
        for a in aliases:
            if a in name_to_p:
                return name_to_p[a]
        return None

    pos = _get("positive", "pos", "label_0")
    neg = _get("negative", "neg", "label_1")
    neu = _get("neutral", "neu", "label_2")
    return pos, neg, neu


class FinBERTScorer:
    """
    Load tokenizer + model once, then score text batches on CPU or CUDA.

    Parameters
    ----------
    model_name
        Hugging Face model id (default ``ProsusAI/finbert`` or env ``FINBERT_MODEL``).
    device
        ``"cuda"``, ``"cpu"``, or ``None`` / ``"auto"`` to pick CUDA if available.
    """

    def __init__(
        self,
        model_name: str | None = None,
        device: str | None = None,
    ) -> None:
        self.model_name = (model_name or os.getenv("FINBERT_MODEL") or _DEFAULT_MODEL).strip()
        self._device_str = device
        self._torch = None
        self._tokenizer = None
        self._model = None
        self._device = None

    def _load(self) -> None:
        if self._model is not None:
            return
        torch, AutoModelForSequenceClassification, AutoTokenizer = _optional_imports()
        self._torch = torch
        self._device = _resolve_device(torch, self._device_str)
        logger.info("Loading FinBERT model=%s device=%s", self.model_name, self._device)
        self._tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self._model = AutoModelForSequenceClassification.from_pretrained(self.model_name)
        self._model.to(self._device)
        self._model.eval()

    def score_texts(
        self,
        texts: list[str],
        *,
        batch_size: int = 8,
    ) -> list[dict[str, Any]]:
        """
        Run FinBERT on each string; return one result dict per input (same order).

        Each dict has:
        - ``prob_pos``, ``prob_neg``, ``prob_neutral`` (float or None)
        - ``score``: ``prob_pos - prob_neg`` when both set, else None
        - ``error``: str if the row was skipped (e.g. empty text), else None

        Empty or whitespace-only strings are not passed to the model; they get
        ``error="empty_text"`` and null probabilities.
        """
        return self._score_texts_impl(texts, batch_size=batch_size)

    def _score_texts_impl(
        self,
        texts: list[str],
        *,
        batch_size: int,
    ) -> list[dict[str, Any]]:
        self._load()
        assert self._model is not None and self._tokenizer is not None and self._torch is not None
        torch = self._torch
        id2label: dict[int, str] = dict(self._model.config.id2label.items())

        out: list[dict[str, Any]] = []
        batch: list[str] = []

        def run_batch(strings: list[str]) -> list[dict[str, Any]]:
            enc = self._tokenizer(
                strings,
                truncation=True,
                padding=True,
                max_length=512,
                return_tensors="pt",
            )
            enc = {k: v.to(self._device) for k, v in enc.items()}
            rows: list[dict[str, Any]] = []
            with torch.no_grad():
                logits = self._model(**enc).logits
            for i in range(logits.size(0)):
                pos, neg, neu = _probs_from_logits(id2label, logits[i : i + 1])
                sc = (pos - neg) if pos is not None and neg is not None else None
                rows.append(
                    {
                        "prob_pos": pos,
                        "prob_neg": neg,
                        "prob_neutral": neu,
                        "score": sc,
                        "error": None,
                    }
                )
            return rows

        for t in texts:
            s = (t or "").strip()
            if not s:
                out.append(
                    {
                        "prob_pos": None,
                        "prob_neg": None,
                        "prob_neutral": None,
                        "score": None,
                        "error": "empty_text",
                    }
                )
                continue
            batch.append(s)
            if len(batch) >= max(1, batch_size):
                out.extend(run_batch(batch))
                batch = []

        if batch:
            out.extend(run_batch(batch))

        return out


def score_texts_batch(
    texts: list[str],
    *,
    model_name: str | None = None,
    device: str | None = None,
    batch_size: int = 8,
) -> list[dict[str, Any]]:
    """
    One-shot scoring using a fresh :class:`FinBERTScorer` (loads model each call).

    Prefer reusing a single :class:`FinBERTScorer` instance in long-running jobs.
    """
    return FinBERTScorer(model_name=model_name, device=device).score_texts(
        texts, batch_size=batch_size
    )
