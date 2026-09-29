"""Evaluate a saved DistilBERT LoRA adapter once on the held-out LIAR test split."""

import argparse
import json
from pathlib import Path

import torch
from peft import PeftModel
from sklearn.metrics import accuracy_score, classification_report, f1_score
from torch.utils.data import DataLoader
from transformers import AutoModelForSequenceClassification, AutoTokenizer, DataCollatorWithPadding

from training.prepare_lora_data import DATA_DIR, DEFAULT_BATCH_SIZE, LiarTokenizedDataset, TOKENIZER_DIR
from training.setup_model import LABELS


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_ADAPTER_DIR = ROOT / "models" / "distilbert-liar-lora"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--adapter-dir", type=Path, default=DEFAULT_ADAPTER_DIR)
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument("--output-json", type=Path)
    args = parser.parse_args()
    adapter_dir = args.adapter_dir.resolve()
    test_path = DATA_DIR / "test.jsonl"

    if args.batch_size < 1:
        parser.error("--batch-size must be at least 1")
    for path in (TOKENIZER_DIR, adapter_dir):
        if not path.is_dir():
            raise FileNotFoundError(f"Model directory not found: {path}")
    if not test_path.is_file():
        raise FileNotFoundError(f"LIAR test split not found: {test_path}")

    torch.set_num_threads(min(4, torch.get_num_threads()))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_DIR, local_files_only=True)
    test_data = LiarTokenizedDataset(test_path, tokenizer)
    loader = DataLoader(
        test_data,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=DataCollatorWithPadding(tokenizer=tokenizer),
    )

    base_model = AutoModelForSequenceClassification.from_pretrained(
        TOKENIZER_DIR, local_files_only=True
    )
    model = PeftModel.from_pretrained(base_model, adapter_dir, is_trainable=False)
    model.to(device)
    model.eval()

    y_true = []
    y_pred = []
    with torch.inference_mode():
        for batch in loader:
            batch = {name: values.to(device) for name, values in batch.items()}
            labels = batch.pop("labels")
            predictions = model(**batch).logits.argmax(dim=-1)
            y_true.extend(labels.cpu().tolist())
            y_pred.extend(predictions.cpu().tolist())

    report = classification_report(
        y_true,
        y_pred,
        labels=list(range(len(LABELS))),
        target_names=list(LABELS),
        output_dict=True,
        zero_division=0,
    )
    result = {
        "dataset": "LIAR deduplicated held-out test split",
        "examples": len(y_true),
        "device": str(device),
        "accuracy": accuracy_score(y_true, y_pred),
        "macro_f1": f1_score(y_true, y_pred, average="macro", zero_division=0),
        "weighted_f1": f1_score(y_true, y_pred, average="weighted", zero_division=0),
        "per_class": {label: report[label] for label in LABELS},
    }
    print(json.dumps(result, indent=2))

    if args.output_json:
        output_path = args.output_json.resolve()
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
        print(f"Saved metrics to {output_path}")


if __name__ == "__main__":
    main()
