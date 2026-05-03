"""Tests for removing legacy flat-module compatibility wrappers."""

from pathlib import Path


REMOVED_WRAPPERS = (
    "src/model.py",
    "src/model_bert.py",
    "src/baseline.py",
    "src/train.py",
    "src/train_bert.py",
    "src/dataset.py",
    "src/dataset_bert.py",
    "src/parser.py",
    "src/preprocess.py",
    "src/features.py",
    "src/app_service.py",
    "src/evaluate.py",
    "src/inference.py",
)


def test_legacy_flat_wrapper_files_are_removed():
    project_root = Path(__file__).parent.parent
    remaining = [path for path in REMOVED_WRAPPERS if (project_root / path).exists()]
    assert remaining == []


def test_canonical_data_paths_importable():
    from src.data.features import class_balance
    from src.data.parser import load_all_domains, parse_review_file
    from src.data.preprocess import clean_text, preprocess

    assert callable(class_balance)
    assert callable(load_all_domains)
    assert callable(parse_review_file)
    assert callable(clean_text)
    assert callable(preprocess)


def test_canonical_model_and_tokenization_paths_importable():
    from src.models.bert import DistilBERTSentiment
    from src.models.bilstm import BiLSTMSentiment
    from src.tokenization.bert import BertReviewDataset
    from src.tokenization.sequence import ReviewDataset, make_dataloaders
    from src.tokenization.vocab import build_vocab

    assert callable(DistilBERTSentiment)
    assert callable(BiLSTMSentiment)
    assert callable(BertReviewDataset)
    assert callable(ReviewDataset)
    assert callable(make_dataloaders)
    assert callable(build_vocab)


def test_canonical_training_and_evaluation_paths_importable():
    from src.evaluation import run_evaluation
    from src.training.baseline import train_baseline
    from src.training.bert import train_bert
    from src.training.bilstm import train

    assert callable(run_evaluation)
    assert callable(train_baseline)
    assert callable(train_bert)
    assert callable(train)
