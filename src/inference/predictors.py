"""Model-specific sentiment predictors."""

from pathlib import Path
from typing import Optional, Protocol, runtime_checkable

import torch

from src.config import MODEL_BASELINE, MODEL_BILSTM, MODEL_DISTILBERT, PRED_THRESHOLD
from src.data.preprocess import clean_text

from .loaders import load_baseline_model, load_bilstm_model, load_distilbert_model


@runtime_checkable
class Predictor(Protocol):
    """Single-text sentiment predictor interface."""

    def predict(self, text: str) -> dict:
        """Return {"label": str, "confidence": float, "model": str}."""
        ...


class BaselinePredictor:
    """TF-IDF + Logistic Regression predictor."""

    def predict(self, text: str, path: Optional[Path] = None) -> dict:
        pipeline = load_baseline_model(path)
        cleaned = clean_text(text)
        proba = pipeline.predict_proba([cleaned])[0]
        pred_idx = int(proba.argmax())
        confidence = float(proba[pred_idx])
        label = "Positive review" if pred_idx == 1 else "Negative review"
        return {
            "label": label,
            "confidence": round(confidence, 4),
            "model": MODEL_BASELINE,
        }


class BiLSTMPredictor:
    """BiLSTM + GloVe predictor."""

    def predict(
        self,
        text: str,
        checkpoint_path: Optional[Path] = None,
        vocab_path: Optional[Path] = None,
    ) -> dict:
        from src.tokenization.sequence import MAX_LEN, tokenize_and_pad

        model, vocab, device = load_bilstm_model(checkpoint_path, vocab_path)
        cleaned = clean_text(text)
        tokens = tokenize_and_pad([cleaned], vocab, max_len=MAX_LEN).to(device)

        model.eval()
        with torch.no_grad():
            prob = torch.sigmoid(model(tokens)).item()

        pred_idx = int(prob >= PRED_THRESHOLD)
        confidence = prob if pred_idx == 1 else 1.0 - prob
        label = "Positive review" if pred_idx == 1 else "Negative review"
        return {
            "label": label,
            "confidence": round(confidence, 4),
            "model": MODEL_BILSTM,
        }


class DistilBERTPredictor:
    """Hugging Face DistilBERT predictor."""

    def predict(self, text: str, checkpoint_path: Optional[Path] = None) -> dict:
        model, tokenizer, checkpoint, device = load_distilbert_model(checkpoint_path)
        cleaned = clean_text(text)
        max_len = int(checkpoint.get("model_config", {}).get("max_len", 256))

        encoded = tokenizer(
            [cleaned],
            padding="max_length",
            truncation=True,
            max_length=max_len,
            return_tensors="pt",
        )
        encoded = {key: value.to(device) for key, value in encoded.items()}

        model.eval()
        with torch.no_grad():
            prob = torch.sigmoid(
                model(
                    input_ids=encoded["input_ids"],
                    attention_mask=encoded["attention_mask"],
                )
            ).item()

        pred_idx = int(prob >= PRED_THRESHOLD)
        confidence = prob if pred_idx == 1 else 1.0 - prob
        label = "Positive review" if pred_idx == 1 else "Negative review"
        return {
            "label": label,
            "confidence": round(confidence, 4),
            "model": MODEL_DISTILBERT,
        }
