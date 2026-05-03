"""Training loop for BiLSTMSentiment.

Trains the BiLSTM model, tracks val F1 per epoch, and saves the best
checkpoint to outputs/bilstm.pt.

Usage:
    python -m src.training.bilstm
"""

from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn
from sklearn.metrics import f1_score, accuracy_score
import time

from src.config import BILSTM_CHECKPOINT_PATH, OUTPUTS_DIR, PRED_THRESHOLD, VOCAB_PATH
from src.tokenization.sequence import (
    BATCH_SIZE,
    MAX_LEN,
    make_dataloaders,
)
from src.tokenization.vocab import (
    EMBEDDINGS_DIR,
    EMBEDDING_DIM,
    build_vocab,
    load_glove,
    save_vocab,
)
from src.models.bilstm import BiLSTMSentiment

# ---------------------------------------------------------------------------
# Hyperparameters
# ---------------------------------------------------------------------------

EPOCHS     = 10
LR         = 1e-3
CLIP       = 5.0      # gradient clipping max norm
HIDDEN_DIM = 256
N_LAYERS   = 2
DROPOUT    = 0.5
SEED       = 42

CHECKPOINT_PATH = BILSTM_CHECKPOINT_PATH


# ---------------------------------------------------------------------------
# Per-epoch helpers
# ---------------------------------------------------------------------------

def train_one_epoch(
    model: BiLSTMSentiment,
    loader: torch.utils.data.DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    clip: float = CLIP,
    device: torch.device = torch.device("cpu"),
) -> dict:
    """Run one training epoch.

    Returns:
        Dict with 'loss' (mean batch loss over the epoch).
    """
    model.train()
    total_loss = 0.0

    for tokens, labels in loader:
        tokens = tokens.to(device)
        labels = labels.float().to(device)

        optimizer.zero_grad()
        logits = model(tokens)
        loss = criterion(logits, labels)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), clip)
        optimizer.step()

        total_loss += loss.item()

    return {"loss": total_loss / len(loader)}


def evaluate_epoch(
    model: BiLSTMSentiment,
    loader: torch.utils.data.DataLoader,
    criterion: nn.Module,
    device: torch.device = torch.device("cpu"),
) -> dict:
    """Evaluate model on a dataloader (val or test).

    Returns:
        Dict with 'loss', 'accuracy', and 'f1'.
    """
    model.eval()
    total_loss = 0.0
    all_preds  = []
    all_labels = []

    with torch.no_grad():
        for tokens, labels in loader:
            tokens = tokens.to(device)
            labels_device = labels.float().to(device)

            logits = model(tokens)
            loss   = criterion(logits, labels_device)
            total_loss += loss.item()

            preds = (torch.sigmoid(logits) >= PRED_THRESHOLD).long().cpu()
            all_preds.extend(preds.tolist())
            all_labels.extend(labels.tolist())

    acc = accuracy_score(all_labels, all_preds)
    f1  = f1_score(all_labels, all_preds, zero_division=0)

    return {
        "loss":     total_loss / len(loader),
        "accuracy": round(acc, 4),
        "f1":       round(f1, 4),
    }


# ---------------------------------------------------------------------------
# Main training function
# ---------------------------------------------------------------------------

def train(
    train_df,
    val_df,
    vocab: dict,
    epochs: int = EPOCHS,
    lr: float = LR,
    clip: float = CLIP,
    hidden_dim: int = HIDDEN_DIM,
    n_layers: int = N_LAYERS,
    dropout: float = DROPOUT,
    batch_size: int = BATCH_SIZE,
    max_len: int = MAX_LEN,
    embedding_dim: int = EMBEDDING_DIM,
    pretrained_embeddings=None,
    checkpoint_path: Optional[Path] = None,
    seed: int = SEED,
    device: Optional[torch.device] = None,
) -> dict:
    """Train BiLSTMSentiment and save the best checkpoint by val F1.

    Args:
        train_df:             DataFrame with 'text' and 'label' columns.
        val_df:               DataFrame with 'text' and 'label' columns.
        vocab:                Token-to-index mapping from build_vocab().
        epochs:               Number of training epochs.
        lr:                   Adam learning rate.
        clip:                 Gradient clipping max norm.
        hidden_dim:           LSTM hidden size per direction.
        n_layers:             Number of stacked LSTM layers.
        dropout:              Dropout probability.
        batch_size:           DataLoader batch size.
        max_len:              Sequence length after padding.
        embedding_dim:        Embedding dimension.
        pretrained_embeddings: Optional GloVe numpy array.
        checkpoint_path:      Where to save the best checkpoint.
        seed:                 Random seed for DataLoader shuffling.
        device:               Torch device (defaults to CPU).

    Returns:
        Dict with 'best_val_f1', 'best_epoch', and 'history' (list of
        per-epoch metric dicts).
    """
    if device is None:
        if torch.cuda.is_available():
            device = torch.device("cuda")
        elif torch.backends.mps.is_available():
            device = torch.device("mps")
        else:
            device = torch.device("cpu")

    checkpoint_path = checkpoint_path or CHECKPOINT_PATH
    Path(checkpoint_path).parent.mkdir(parents=True, exist_ok=True)

    torch.manual_seed(seed)

    # Build dataloaders (test_df not needed during training — pass val as placeholder)
    train_loader, val_loader, _ = make_dataloaders(
        train_df, val_df, val_df,
        vocab=vocab,
        batch_size=batch_size,
        max_len=max_len,
        seed=seed,
    )

    vocab_size = len(vocab)
    model = BiLSTMSentiment(
        vocab_size=vocab_size,
        embedding_dim=embedding_dim,
        hidden_dim=hidden_dim,
        n_layers=n_layers,
        dropout=dropout,
        pretrained_embeddings=pretrained_embeddings,
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.BCEWithLogitsLoss()

    best_val_f1  = -1.0
    best_epoch   = 0
    history      = []

    print(f"Training on {device} | epochs={epochs} | vocab={vocab_size:,} | hidden={hidden_dim}", flush=True)
    print("-" * 60, flush=True)

    for epoch in range(1, epochs + 1):
        train_metrics = train_one_epoch(model, train_loader, optimizer, criterion, clip, device)
        val_metrics   = evaluate_epoch(model, val_loader, criterion, device)

        history.append({
            "epoch":      epoch,
            "train_loss": train_metrics["loss"],
            **{f"val_{k}": v for k, v in val_metrics.items()},
        })

        print(
            f"Epoch {epoch:>2}/{epochs} | "
            f"train_loss={train_metrics['loss']:.4f} | "
            f"val_loss={val_metrics['loss']:.4f} | "
            f"val_acc={val_metrics['accuracy']:.4f} | "
            f"val_f1={val_metrics['f1']:.4f}",
            flush=True,
        )

        if val_metrics["f1"] > best_val_f1:
            best_val_f1 = val_metrics["f1"]
            best_epoch  = epoch
            torch.save(
                {
                    "model_state": model.state_dict(),
                    "model_config": {
                        "vocab_size":    vocab_size,
                        "embedding_dim": embedding_dim,
                        "hidden_dim":    hidden_dim,
                        "n_layers":      n_layers,
                        "dropout":       dropout,
                    },
                    "vocab_path":   str(VOCAB_PATH),
                    "best_val_f1":  best_val_f1,
                    "best_epoch":   best_epoch,
                    "history":      history,
                },
                checkpoint_path,
            )
            print(f"  ✓ checkpoint saved (val_f1={best_val_f1:.4f})", flush=True)

    print("-" * 60, flush=True)
    print(f"Best val F1: {best_val_f1:.4f} at epoch {best_epoch}", flush=True)
    return {"best_val_f1": best_val_f1, "best_epoch": best_epoch, "history": history}


def main() -> None:
    """Train the BiLSTM model from the CLI."""
    start_time = time.perf_counter()
    from src.data.parser import load_all_domains
    from src.data.preprocess import preprocess

    raw = load_all_domains()
    train_df, val_df, _ = preprocess(raw)

    vocab = build_vocab(train_df["text"])
    save_vocab(vocab, VOCAB_PATH)

    glove_path = EMBEDDINGS_DIR / "glove.6B.100d.txt"
    pretrained = load_glove(vocab, glove_path) if glove_path.exists() else None
    load_time = time.perf_counter()
    print(f"Data loaded and preprocessed in {load_time - start_time:.2f} seconds", flush=True)
    train(train_df, val_df, vocab, pretrained_embeddings=pretrained)
    end_time = time.perf_counter()
    elapsed = end_time - load_time
    print(f"Training-only time (excluding data load): {elapsed:.2f} seconds", flush=True)


if __name__ == "__main__":
    main()
