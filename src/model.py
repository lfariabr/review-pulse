"""BiLSTM sentiment classifier for ReviewPulse.

Architecture:
    Embedding(vocab_size, embedding_dim, padding_idx=0)
    → Dropout
    → BiLSTM(hidden_dim, num_layers=2, bidirectional=True)
    → concat final forward + backward hidden state
    → Dropout
    → Linear(hidden_dim * 2, 1)

Output is a raw logit; apply sigmoid at inference time or use
BCEWithLogitsLoss during training.
"""

from typing import Optional

import numpy as np
import torch
import torch.nn as nn


class BiLSTMSentiment(nn.Module):
    """Bidirectional LSTM for binary sentiment classification.

    Args:
        vocab_size:           Number of tokens in the vocabulary (including <pad> and <unk>).
        embedding_dim:        Dimension of token embeddings (default: 100).
        hidden_dim:           Hidden size per LSTM direction (default: 256).
        n_layers:             Number of stacked LSTM layers (default: 2).
        dropout:              Dropout probability applied after embedding and before
                              the final linear layer (default: 0.5).
        pretrained_embeddings: Optional numpy array of shape (vocab_size, embedding_dim)
                              to initialise the embedding matrix (e.g. from load_glove()).
                              When provided the weights are copied and left trainable.
    """

    def __init__(
        self,
        vocab_size: int,
        embedding_dim: int = 100,
        hidden_dim: int = 256,
        n_layers: int = 2,
        dropout: float = 0.5,
        pretrained_embeddings: Optional[np.ndarray] = None,
    ) -> None:
        super().__init__()

        self.embedding = nn.Embedding(
            vocab_size, embedding_dim, padding_idx=0
        )

        if pretrained_embeddings is not None:
            if pretrained_embeddings.shape != (vocab_size, embedding_dim):
                raise ValueError(
                    f"pretrained_embeddings shape {pretrained_embeddings.shape} "
                    f"does not match (vocab_size={vocab_size}, embedding_dim={embedding_dim})"
                )
            self.embedding.weight.data.copy_(
                torch.from_numpy(pretrained_embeddings).float()
            )

        self.embedding_dropout = nn.Dropout(dropout)

        self.lstm = nn.LSTM(
            input_size=embedding_dim,
            hidden_size=hidden_dim,
            num_layers=n_layers,
            bidirectional=True,
            dropout=dropout if n_layers > 1 else 0.0,
            batch_first=True,
        )

        self.fc_dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_dim * 2, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x: LongTensor of token indices, shape (batch_size, seq_len).

        Returns:
            Raw logit tensor, shape (batch_size,).
        """
        # Compute real (non-pad) length for each sequence; clamp to ≥1 so
        # pack_padded_sequence never receives a length of zero.
        lengths = (x != 0).sum(dim=1).cpu()
        lengths = torch.clamp(lengths, min=1)

        embedded = self.embedding_dropout(self.embedding(x))
        # embedded: (batch, seq_len, embedding_dim)

        packed = nn.utils.rnn.pack_padded_sequence(
            embedded, lengths, batch_first=True, enforce_sorted=False
        )
        _, (hidden, _) = self.lstm(packed)
        # hidden: (n_layers * 2, batch, hidden_dim)
        # hidden[-2] = top forward direction, hidden[-1] = top backward direction
        # With packed sequences these correspond to the real sequence endpoints.
        final_hidden = torch.cat([hidden[-2], hidden[-1]], dim=1)
        # final_hidden: (batch, hidden_dim * 2)

        out = self.fc(self.fc_dropout(final_hidden))
        # out: (batch, 1) → squeeze to (batch,)
        return out.squeeze(1)
