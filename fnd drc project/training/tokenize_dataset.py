"""Offline DistilBERT tokenization and dynamically padded data loaders (no training)."""

import argparse
import hashlib
from importlib.metadata import version
import json
import math
from pathlib import Path
import statistics

from training.setup_model import DEFAULT_OUTPUT as MODEL_DIR, LABEL2ID, MAX_LENGTH, ROOT, SEED

DATA_DIR = ROOT / "data" / "liar"
SPLITS = ("train", "validation", "test")
TRAIN_BATCH_SIZE = 16
EVAL_BATCH_SIZE = 32


def sha256(data):
    return hashlib.sha256(data).hexdigest()


def read_rows(path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def length_summary(lengths, limit):
    ordered = sorted(lengths)
    if not ordered:
        raise ValueError("Cannot tokenize an empty split")
    truncated = sum(n > limit for n in lengths)
    return {"rows": len(lengths), "min": ordered[0], "median": statistics.median(ordered),
            "p95": ordered[math.ceil(.95 * len(ordered)) - 1],
            "p99": ordered[math.ceil(.99 * len(ordered)) - 1], "max": ordered[-1],
            "truncated_rows": truncated, "truncated_percent": 100 * truncated / len(lengths),
            "tokens_removed": sum(max(0, n - limit) for n in lengths)}


def encode_rows(rows, tokenizer, max_length=MAX_LENGTH):
    texts = [r["statement"] for r in rows]
    full = tokenizer(texts, padding=False, truncation=False, add_special_tokens=True)
    encoded = tokenizer(texts, padding=False, truncation=True, max_length=max_length,
                        add_special_tokens=True, return_attention_mask=True)
    result = []
    for i, row in enumerate(rows):
        if row["label"] != LABEL2ID[row["label_name"]]:
            raise ValueError(f"Label mismatch for {row['id']}")
        result.append({"id": row["id"], "input_ids": encoded["input_ids"][i],
                       "attention_mask": encoded["attention_mask"][i], "labels": row["label"]})
    lengths = [len(ids) for ids in full["input_ids"]]
    summary = length_summary(lengths, max_length)
    summary["truncated_ids"] = [r["id"] for r, n in zip(rows, lengths) if n > max_length]
    return result, summary


def make_dataloader(rows, tokenizer, *, training=False, batch_size=None, seed=SEED):
    """Only tensors consumed by the model enter batches; IDs stay in stored rows."""
    import torch
    from torch.utils.data import DataLoader
    from transformers import DataCollatorWithPadding

    if batch_size is None:
        batch_size = TRAIN_BATCH_SIZE if training else EVAL_BATCH_SIZE
    if batch_size < 1:
        raise ValueError("batch_size must be positive")
    features = [{k: row[k] for k in ("input_ids", "attention_mask", "labels")} for row in rows]
    return DataLoader(features, batch_size=batch_size, shuffle=training, drop_last=False,
                      num_workers=0, generator=torch.Generator().manual_seed(seed),
                      collate_fn=DataCollatorWithPadding(tokenizer, padding="longest", return_tensors="pt"))


def load_dataloader(split, *, output_dir=DATA_DIR / "tokenized", model_dir=MODEL_DIR,
                    batch_size=None, seed=SEED):
    """Load a prepared split; only train is shuffled. Test loading is explicit."""
    from transformers import AutoTokenizer

    if split not in SPLITS:
        raise ValueError(f"Unknown split: {split}")
    report = json.loads((output_dir / "report.json").read_text())
    for name, expected in report["tokenizer_sha256"].items():
        if sha256((model_dir / name).read_bytes()) != expected:
            raise ValueError(f"Tokenizer file changed: {name}")
    data = output_dir / f"{split}.jsonl"
    if sha256(data.read_bytes()) != report["output_sha256"][data.name]:
        raise ValueError(f"Tokenized data changed: {data}")
    tokenizer = AutoTokenizer.from_pretrained(model_dir, local_files_only=True)
    tokenizer.padding_side = "right"
    return make_dataloader(read_rows(data), tokenizer, training=split == "train",
                           batch_size=batch_size, seed=seed)


def prepare_tokens(data_dir=DATA_DIR, model_dir=MODEL_DIR, output_dir=None):
    from transformers import AutoTokenizer

    output_dir = output_dir or data_dir / "tokenized"
    source_report = json.loads((data_dir / "report.json").read_text())
    config = json.loads((model_dir / "config.json").read_text())
    metadata = json.loads((model_dir / "setup_metadata.json").read_text())
    if config["model_type"] != "distilbert" or config["label2id"] != LABEL2ID:
        raise ValueError("Expected the six-class DistilBERT checkpoint")
    if MAX_LENGTH > config["max_position_embeddings"]:
        raise ValueError("Token limit exceeds model capacity")
    tokenizer = AutoTokenizer.from_pretrained(model_dir, local_files_only=True)
    tokenizer.truncation_side = "right"
    tokenizer.padding_side = "right"
    tokenizer_files = [p for p in model_dir.iterdir()
                       if p.is_file() and (p.name.startswith("tokenizer")
                                           or p.name in ("vocab.txt", "special_tokens_map.json", "added_tokens.json"))]
    report = {"base_model": metadata["base_model"], "revision": metadata["resolved_revision"],
              "tokenizer_sha256": {p.name: sha256(p.read_bytes()) for p in sorted(tokenizer_files)},
              "packages": {name: version(name) for name in ("transformers", "tokenizers", "torch")},
              "label2id": LABEL2ID, "max_length": MAX_LENGTH,
              "lengths_include_special_tokens": True,
              "limit_policy": "128 fixed from model setup; judge suitability on train lengths. Held-out lengths are diagnostics only.",
              "padding": "right, longest sequence in each batch; no padding stored",
              "truncation": "right, preserving tokenizer special tokens",
              "train_batch_size": TRAIN_BATCH_SIZE, "eval_batch_size": EVAL_BATCH_SIZE,
              "shuffle_train_only": True, "drop_last": False, "seed": SEED,
              "input_sha256": {}, "splits": {}, "output_sha256": {}}
    files = {}
    for split in SPLITS:
        path = data_dir / "deduplicated" / f"{split}.jsonl"
        checksum = sha256(path.read_bytes())
        if checksum != source_report["sha256"][f"deduplicated/{split}.jsonl"]:
            raise ValueError(f"Prepared source data changed: {path}")
        report["input_sha256"][split] = checksum
        rows, stats = encode_rows(read_rows(path), tokenizer)
        report["splits"][split] = stats
        payload = "".join(json.dumps(row) + "\n" for row in rows).encode()
        files[f"{split}.jsonl"] = payload
        report["output_sha256"][f"{split}.jsonl"] = sha256(payload)
    report["train_fits_128_without_truncation"] = report["splits"]["train"]["truncated_rows"] == 0
    files["report.json"] = (json.dumps(report, indent=2) + "\n").encode()
    for name, payload in files.items():
        path = output_dir / name
        if path.exists() and path.read_bytes() != payload:
            raise ValueError(f"Refusing to overwrite different contents: {path}")
    output_dir.mkdir(parents=True, exist_ok=True)
    for name, payload in files.items():
        path = output_dir / name
        if not path.exists():
            path.write_bytes(payload)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=DATA_DIR)
    parser.add_argument("--model-dir", type=Path, default=MODEL_DIR)
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()
    report = prepare_tokens(args.data_dir, args.model_dir, args.output_dir)
    print(json.dumps({"max_length": report["max_length"], "splits": report["splits"],
                      "train_fits_128_without_truncation": report["train_fits_128_without_truncation"]}, indent=2))


if __name__ == "__main__":
    main()
