"""Compatibility wrapper for ReviewPulse tokenization and dataset helpers."""

from src.tokenization.sequence import *  # noqa: F401,F403
from src.tokenization.vocab import *  # noqa: F401,F403


def main() -> None:
    """Run a quick tokenization and DataLoader smoke test."""
    from src.parser import load_all_domains
    from src.preprocess import preprocess

    raw = load_all_domains()
    train, val, test = preprocess(raw)

    vocab = build_vocab(train["text"].tolist())
    save_vocab(vocab)

    embeddings = load_glove(vocab)
    print(f"Embedding matrix: {embeddings.shape}")

    train_loader, val_loader, test_loader = make_dataloaders(train, val, test, vocab)
    tokens, labels = next(iter(train_loader))
    print(f"Batch - tokens: {tokens.shape}, labels: {labels.shape}")


if __name__ == "__main__":
    main()
