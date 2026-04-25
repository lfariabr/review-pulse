"""Local and pretrained transformer sentiment classifiers for ReviewPulse.

The repo keeps the historical ``DistilBERTSentiment`` name for the local,
non-Hugging-Face model so existing imports continue to work. An optional
``PretrainedDistilBERTSentiment`` wrapper is also provided for direct
comparison against a cached pretrained DistilBERT encoder.
"""

from typing import Optional

import numpy as np
import torch
import torch.nn as nn

try:
    from transformers import AutoModel
except ImportError as exc:  # pragma: no cover - optional dependency
    AutoModel = None
    _TRANSFORMERS_IMPORT_ERROR = exc
else:
    _TRANSFORMERS_IMPORT_ERROR = None


DISTILBERT_MODEL_NAME = "reviewpulse-local-transformer"
PRETRAINED_DISTILBERT_MODEL_NAME = "distilbert-base-uncased"
BERT_DROPOUT = 0.2
BERT_EMBEDDING_DIM = 128
BERT_HEADS = 4
BERT_LAYERS = 3
BERT_FF_DIM = 256
DEFAULT_VOCAB_SIZE = 30_000
DEFAULT_MAX_LEN = 256


class DistilBERTSentiment(nn.Module):
    """Local transformer encoder with a binary classification head."""

    def __init__(
        self,
        model_name: str = DISTILBERT_MODEL_NAME,
        dropout: float = BERT_DROPOUT,
        freeze_encoder: bool = False,
        local_files_only: bool = False,
        vocab_size: int = DEFAULT_VOCAB_SIZE,
        embedding_dim: int = BERT_EMBEDDING_DIM,
        token_embedding_dim: Optional[int] = None,
        n_heads: int = BERT_HEADS,
        n_layers: int = BERT_LAYERS,
        ff_dim: int = BERT_FF_DIM,
        max_len: int = DEFAULT_MAX_LEN,
        pretrained_embeddings: Optional[np.ndarray] = None,
    ) -> None:
        super().__init__()

        if embedding_dim % n_heads != 0:
            raise ValueError(
                f"embedding_dim={embedding_dim} must be divisible by n_heads={n_heads}"
            )
        if vocab_size < 2:
            raise ValueError("vocab_size must be at least 2")

        del local_files_only

        self.model_name = model_name
        self.max_len = max_len

        token_embedding_dim = token_embedding_dim or embedding_dim
        if pretrained_embeddings is not None:
            if pretrained_embeddings.shape[0] != vocab_size:
                raise ValueError(
                    f"pretrained_embeddings shape {pretrained_embeddings.shape} "
                    f"does not match vocab_size={vocab_size}"
                )
            token_embedding_dim = int(pretrained_embeddings.shape[1])

        self.token_embedding = nn.Embedding(
            vocab_size,
            token_embedding_dim,
            padding_idx=0,
        )
        if pretrained_embeddings is not None:
            self.token_embedding.weight.data.copy_(
                torch.from_numpy(pretrained_embeddings).float()
            )

        self.embedding_projection = (
            nn.Identity()
            if token_embedding_dim == embedding_dim
            else nn.Linear(token_embedding_dim, embedding_dim, bias=False)
        )
        self.position_embedding = nn.Embedding(max_len, embedding_dim)
        self.embedding_norm = nn.LayerNorm(embedding_dim)
        self.embedding_dropout = nn.Dropout(dropout)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embedding_dim,
            nhead=n_heads,
            dim_feedforward=ff_dim,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=n_layers,
            enable_nested_tensor=False,
        )

        self.dropout = nn.Dropout(dropout)
        self.mlm_head = nn.Sequential(
            nn.Linear(embedding_dim, embedding_dim),
            nn.GELU(),
            nn.LayerNorm(embedding_dim),
            nn.Linear(embedding_dim, vocab_size),
        )
        self.fc = nn.Linear(embedding_dim * 2, 1)

        if freeze_encoder:
            for module in (
                self.token_embedding,
                self.embedding_projection,
                self.position_embedding,
                self.embedding_norm,
                self.encoder,
            ):
                for param in module.parameters():
                    param.requires_grad = False

    def encode(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Encode tokens into contextual embeddings."""
        if attention_mask is None:
            attention_mask = (input_ids != 0).long()

        seq_len = input_ids.size(1)
        if seq_len > self.max_len:
            raise ValueError(
                f"Sequence length {seq_len} exceeds model max_len={self.max_len}"
            )

        position_ids = torch.arange(seq_len, device=input_ids.device).unsqueeze(0)
        x = self.token_embedding(input_ids)
        x = self.embedding_projection(x)
        x = x + self.position_embedding(position_ids)
        x = self.embedding_dropout(self.embedding_norm(x))

        encoded = self.encoder(
            x,
            src_key_padding_mask=(attention_mask == 0),
        )
        return encoded, attention_mask

    def forward_mlm(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Predict token ids for masked-language-model pretraining."""
        encoded, _ = self.encode(input_ids=input_ids, attention_mask=attention_mask)
        return self.mlm_head(self.dropout(encoded))

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Forward pass for binary classification."""
        encoded, attention_mask = self.encode(
            input_ids=input_ids,
            attention_mask=attention_mask,
        )

        mask = attention_mask.unsqueeze(-1).to(encoded.dtype)
        mean_pooled = (encoded * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1.0)

        masked_encoded = encoded.masked_fill(mask == 0, -1e9)
        max_pooled = masked_encoded.max(dim=1).values
        max_pooled = torch.where(
            mask.sum(dim=1) > 0,
            max_pooled,
            torch.zeros_like(max_pooled),
        )

        pooled = torch.cat([mean_pooled, max_pooled], dim=1)
        logits = self.fc(self.dropout(pooled))
        return logits.squeeze(1)


class PretrainedDistilBERTSentiment(nn.Module):
    """Optional wrapper around a cached pretrained DistilBERT encoder."""

    def __init__(
        self,
        model_name: str = PRETRAINED_DISTILBERT_MODEL_NAME,
        dropout: float = BERT_DROPOUT,
        freeze_encoder: bool = False,
        local_files_only: bool = True,
    ) -> None:
        super().__init__()

        if AutoModel is None:
            raise ImportError(
                "transformers is required for pretrained DistilBERT support. "
                "Install it in the active environment to use this path."
            ) from _TRANSFORMERS_IMPORT_ERROR

        self.encoder = AutoModel.from_pretrained(
            model_name,
            local_files_only=local_files_only,
        )
        hidden_size = int(self.encoder.config.hidden_size)

        if freeze_encoder:
            for param in self.encoder.parameters():
                param.requires_grad = False

        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_size, 1)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        cls_embedding = outputs.last_hidden_state[:, 0]
        logits = self.fc(self.dropout(cls_embedding))
        return logits.squeeze(1)
