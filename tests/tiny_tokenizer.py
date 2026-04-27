"""Reusable tokenizer test double for transformer tests."""

from pathlib import Path

import torch


class TinyTokenizer:
    """Small Hugging Face-style tokenizer for deterministic unit tests."""

    def __init__(self, vocab: dict[str, int] | None = None) -> None:
        self.vocab = vocab or {"[PAD]": 0, "[UNK]": 1}
        self.pad_token_id = self._special_token_id("[PAD]", "<pad>", default=0)
        self.unk_token_id = self._special_token_id("[UNK]", "<unk>", default=1)

    def _special_token_id(
        self,
        hf_name: str,
        project_name: str,
        *,
        default: int,
    ) -> int:
        return self.vocab.get(hf_name, self.vocab.get(project_name, default))

    def __call__(
        self,
        texts,
        padding="max_length",
        truncation=True,
        max_length=16,
        return_tensors="pt",
        **kwargs,
    ):
        del kwargs
        assert padding == "max_length"
        assert truncation is True
        assert return_tensors == "pt"

        rows = []
        masks = []
        for text in texts:
            ids = [
                self.vocab.get(token, self.unk_token_id)
                for token in text.split()
            ][:max_length]
            mask = [1] * len(ids)
            ids += [self.pad_token_id] * (max_length - len(ids))
            mask += [0] * (max_length - len(mask))
            rows.append(ids)
            masks.append(mask)

        return {
            "input_ids": torch.tensor(rows, dtype=torch.long),
            "attention_mask": torch.tensor(masks, dtype=torch.long),
        }

    def save_pretrained(self, save_directory):
        path = Path(save_directory)
        path.mkdir(parents=True, exist_ok=True)
        tokenizer_path = path / "tokenizer.json"
        config_path = path / "tokenizer_config.json"
        tokenizer_path.write_text('{"test": true}')
        config_path.write_text('{"model_max_length": 16}')
        return (str(tokenizer_path), str(config_path))
