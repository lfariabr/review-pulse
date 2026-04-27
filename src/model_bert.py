"""Hugging Face DistilBERT sentiment classifier for ReviewPulse."""

import torch
import torch.nn as nn

try:
    from transformers import DistilBertForSequenceClassification
except ImportError as exc:  # pragma: no cover - optional dependency
    DistilBertForSequenceClassification = None
    _TRANSFORMERS_IMPORT_ERROR = exc
else:
    _TRANSFORMERS_IMPORT_ERROR = None


PRETRAINED_DISTILBERT_MODEL_NAME = "distilbert-base-uncased"
DISTILBERT_MODEL_NAME = PRETRAINED_DISTILBERT_MODEL_NAME
BERT_DROPOUT = 0.2


class DistilBERTSentiment(nn.Module):
    """Binary sentiment classifier backed by Hugging Face DistilBERT.

    The encoder is frozen by default so the first training stage only updates the
    classification head. Pass ``freeze_encoder=False`` when full fine-tuning is
    desired.
    """

    def __init__(
        self,
        model_name: str = PRETRAINED_DISTILBERT_MODEL_NAME,
        dropout: float = BERT_DROPOUT,
        freeze_encoder: bool = True,
        local_files_only: bool = False,
    ) -> None:
        super().__init__()

        if DistilBertForSequenceClassification is None:
            raise ImportError(
                "transformers is required for DistilBERT support. "
                "Install it in the active environment to use this model."
            ) from _TRANSFORMERS_IMPORT_ERROR

        self.model_name = model_name
        self.freeze_encoder = freeze_encoder
        self.model = DistilBertForSequenceClassification.from_pretrained(
            model_name,
            num_labels=1,
            seq_classif_dropout=dropout,
            local_files_only=local_files_only,
        )
        self.encoder = self.model.distilbert

        if freeze_encoder:
            self.freeze_distilbert_encoder()

    def freeze_distilbert_encoder(self) -> None:
        """Freeze the pretrained DistilBERT encoder parameters."""
        for param in self.encoder.parameters():
            param.requires_grad = False
        self.freeze_encoder = True

    def unfreeze_distilbert_encoder(self) -> None:
        """Unfreeze the pretrained DistilBERT encoder for full fine-tuning."""
        for param in self.encoder.parameters():
            param.requires_grad = True
        self.freeze_encoder = False

    def unfreeze_last_encoder_layers(self, n_layers: int) -> list[int]:
        """Unfreeze only the final DistilBERT transformer layers."""
        if n_layers < 1:
            raise ValueError("n_layers must be at least 1")

        layers = self.encoder.transformer.layer
        layer_count = len(layers)
        n_layers = min(n_layers, layer_count)
        start_idx = layer_count - n_layers
        trainable_indexes = list(range(start_idx, layer_count))

        self.freeze_distilbert_encoder()
        for idx in trainable_indexes:
            for param in layers[idx].parameters():
                param.requires_grad = True

        self.freeze_encoder = False
        return trainable_indexes

    @property
    def classifier(self) -> nn.Module:
        return self.model.classifier

    @property
    def pre_classifier(self) -> nn.Module:
        return self.model.pre_classifier

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Return one binary-classification logit per review."""
        outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
        return outputs.logits.squeeze(-1)


PretrainedDistilBERTSentiment = DistilBERTSentiment
