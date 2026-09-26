"""Prepare in-memory LIAR data loaders for DistilBERT LoRA fine-tuning."""

import json
from pathlib import Path

from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer, DataCollatorWithPadding

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data" / "liar" / "deduplicated"
TOKENIZER_DIR = ROOT / "models" / "distilbert-liar-initialized"
SPLITS = ("train", "validation", "test")
MAX_LENGTH = 128
DEFAULT_BATCH_SIZE = 8
NUM_LABELS = 6


class LiarTokenizedDataset(Dataset):
    """Tokenize LIAR statements and retain their numeric class labels in memory."""

    def __init__(self, path, tokenizer):
        self.examples = []
        with path.open("r", encoding="utf-8") as file:
            for line_number, line in enumerate(file, start=1):
                if not line.strip():
                    continue
                try:
                    row = json.loads(line)
                except json.JSONDecodeError as error:
                    raise ValueError(f"Invalid JSON in {path}, line {line_number}: {error}") from error

                statement = row.get("statement")
                label = row.get("label")
                if not isinstance(statement, str) or not statement.strip():
                    raise ValueError(f"Missing or empty statement in {path}, line {line_number}.")
                if not isinstance(label, int) or not 0 <= label < NUM_LABELS:
                    raise ValueError(f"Invalid label {label!r} in {path}, line {line_number}.")

                encoded = tokenizer(
                    statement,
                    max_length=MAX_LENGTH,
                    truncation=True,
                )
                self.examples.append({
                    "input_ids": encoded["input_ids"],
                    "attention_mask": encoded["attention_mask"],
                    "labels": label,
                })

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, index):
        return self.examples[index]


def build_data_loaders(batch_size=DEFAULT_BATCH_SIZE):
    """Create split data loaders; padding happens dynamically within each batch."""
    if batch_size < 1:
        raise ValueError("batch_size must be at least 1.")
    if not TOKENIZER_DIR.is_dir():
        raise FileNotFoundError(f"Local DistilBERT tokenizer not found: {TOKENIZER_DIR}")

    missing = [DATA_DIR / f"{split}.jsonl" for split in SPLITS
               if not (DATA_DIR / f"{split}.jsonl").is_file()]
    if missing:
        raise FileNotFoundError("Missing dataset split(s): " + ", ".join(map(str, missing)))

    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_DIR, local_files_only=True)
    collator = DataCollatorWithPadding(tokenizer=tokenizer)
    loaders = {}
    for split in SPLITS:
        dataset = LiarTokenizedDataset(DATA_DIR / f"{split}.jsonl", tokenizer)
        loaders[split] = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=(split == "train"),
            collate_fn=collator,
        )
    return loaders


def main():
    loaders = build_data_loaders()
    print("LIAR data loaders ready. No model training was performed.")
    print(f"Max length: {MAX_LENGTH}; truncation: enabled; batch size: {DEFAULT_BATCH_SIZE}")
    print("Padding: dynamic, to the longest sequence in each batch")
    for split, loader in loaders.items():
        print(f"{split}: {len(loader.dataset)} claims, {len(loader)} batches")
        if len(loader):
            batch = next(iter(loader))
            print(f"  First batch input shape: {tuple(batch['input_ids'].shape)}")
            print(f"  First batch labels shape: {tuple(batch['labels'].shape)}")


if __name__ == "__main__":
    main()
