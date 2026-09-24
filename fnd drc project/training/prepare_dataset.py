"""Prepare the original LIAR archive without third-party dependencies or training."""

import argparse
from collections import Counter, defaultdict
import csv
import hashlib
import io
import json
from pathlib import Path
import unicodedata
from urllib.request import urlopen
import zipfile

from training.setup_model import LABELS, LABEL2ID

ROOT = Path(__file__).resolve().parents[1]
SOURCE_URL = "https://sites.cs.ucsb.edu/~william/data/liar_dataset.zip"
ARCHIVE_SHA256 = "611c1addad919743dde15822b87a60bfb760d8f85597f25289e34621800654c7"
FILES = {"train": "train.tsv", "validation": "valid.tsv", "test": "test.tsv"}
EXPECTED_COUNTS = {"train": 10269, "validation": 1284, "test": 1283}
DEFAULT_OUTPUT = ROOT / "data" / "liar"


def digest(data):
    return hashlib.sha256(data).hexdigest()


def claim_key(text):
    """Normalize only for duplicate detection, never for model input."""
    return " ".join(unicodedata.normalize("NFKC", text).casefold().split())


def parse_split(data, split):
    # These TSVs have literal quotes, not CSV-style quoted fields.
    reader = csv.reader(io.StringIO(data.decode("utf-8")), delimiter="\t", quoting=csv.QUOTE_NONE)
    records = []
    ids = set()
    for line, columns in enumerate(reader, 1):
        if len(columns) != 14:
            raise ValueError(f"{split}:{line}: expected 14 columns, got {len(columns)}")
        statement_id, label, statement = columns[:3]
        if not statement_id.strip() or not statement.strip() or label not in LABEL2ID:
            raise ValueError(f"{split}:{line}: missing ID/text or invalid label")
        if statement_id in ids:
            raise ValueError(f"{split}:{line}: repeated statement ID {statement_id}")
        ids.add(statement_id)
        records.append({"id": statement_id, "statement": statement, "label": LABEL2ID[label],
                        "label_name": label, "source_split": split, "source_line": line})
    return records


def deduplicate(splits):
    """Keep test before validation before train; earliest row wins within a split.

    Decisions depend only on text and split/row order, never on labels.
    """
    groups = defaultdict(list)
    kept = {split: [] for split in FILES}
    excluded = []
    for split in ("test", "validation", "train"):
        for record in splits[split]:
            key = claim_key(record["statement"])
            if groups[key]:
                winner = groups[key][0]
                excluded.append({"id": record["id"], "split": split,
                                 "kept_id": winner["id"], "kept_split": winner["source_split"],
                                 "reason": "duplicate_normalized_statement"})
            else:
                kept[split].append(record)
            groups[key].append(record)
    duplicates = []
    for key, records in groups.items():
        if len(records) > 1:
            duplicates.append({
                "normalized_text_sha256": digest(key.encode()),
                "cross_split": len({r["source_split"] for r in records}) > 1,
                "conflicting_labels": len({r["label"] for r in records}) > 1,
                "members": [{k: r[k] for k in ("id", "source_split", "source_line", "label_name")}
                            for r in records],
            })
    return kept, duplicates, excluded


def summarize(records):
    lengths = sorted(len(r["statement"].split()) for r in records)
    counts = Counter(r["label_name"] for r in records)
    return {"rows": len(records), "class_counts": {label: counts[label] for label in LABELS},
            "whitespace_word_lengths": {
                "min": lengths[0] if lengths else 0,
                "median": lengths[len(lengths) // 2] if lengths else 0,
                "p95": lengths[min(len(lengths) - 1, int(len(lengths) * .95))] if lengths else 0,
                "max": lengths[-1] if lengths else 0,
            }}


def prepare(archive, output):
    archive_hash = digest(archive)
    if archive_hash != ARCHIVE_SHA256:
        raise ValueError("Archive checksum differs from the verified original release; refusing to prepare it.")
    raw = {}
    with zipfile.ZipFile(io.BytesIO(archive)) as bundle:
        for name in (*FILES.values(), "README"):
            raw[name] = bundle.read(name)
    splits = {split: parse_split(raw[name], split) for split, name in FILES.items()}
    if {split: len(rows) for split, rows in splits.items()} != EXPECTED_COUNTS:
        raise ValueError("Unexpected original split sizes")
    all_ids = [r["id"] for rows in splits.values() for r in rows]
    if len(set(all_ids)) != len(all_ids):
        raise ValueError("Statement IDs overlap across splits")
    cleaned, duplicates, excluded = deduplicate(splits)
    files = {"raw/liar_dataset.zip": archive}
    files.update({f"raw/{name}": data for name, data in raw.items()})
    for variant, data in (("original", splits), ("deduplicated", cleaned)):
        for split, records in data.items():
            files[f"{variant}/{split}.jsonl"] = "".join(
                json.dumps(r, ensure_ascii=False) + "\n" for r in records).encode("utf-8")
    report = {
        "source_url": SOURCE_URL, "archive_sha256": archive_hash,
        "usage": "Research purposes only; original sources retain copyright. See raw/README.",
        "label2id": LABEL2ID, "model_input": "statement only",
        "original": {s: summarize(r) for s, r in splits.items()},
        "deduplicated": {s: summarize(r) for s, r in cleaned.items()},
        "validation": {"missing_required_fields": 0, "invalid_labels": 0,
                       "malformed_rows": 0, "duplicate_ids": 0},
        "duplicate_policy": "NFKC + casefold + collapsed whitespace for matching only. Keep test, then validation, then train; first source row wins. No label-based decisions or relabeling.",
        "duplicate_groups": duplicates, "excluded": excluded,
        "limitations": ["No semantic/near-duplicate detection or speaker-disjoint split.",
                        "Conflicting labels are flagged, not resolved; retained label is from the winning row.",
                        "Deduplicated results are not directly comparable to the original benchmark.",
                        "Word lengths are not tokenizer lengths; measure truncation before training.",
                        "Six LIAR labels are not the application's three evidence statuses."],
        "sha256": {name: digest(data) for name, data in files.items()},
    }
    files["report.json"] = (json.dumps(report, indent=2) + "\n").encode()
    # Validate all data and existing files before writing; repeat runs are idempotent.
    for name, data in files.items():
        target = output / name
        if target.exists() and target.read_bytes() != data:
            raise ValueError(f"Refusing to overwrite different contents: {target}")
    for name, data in files.items():
        target = output / name
        target.parent.mkdir(parents=True, exist_ok=True)
        if not target.exists():
            target.write_bytes(data)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", type=Path, help="Use an existing original ZIP offline")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    cached = args.output_dir / "raw" / "liar_dataset.zip"
    archive_path = args.archive or (cached if cached.exists() else None)
    if archive_path:
        archive = archive_path.read_bytes()
    else:
        with urlopen(SOURCE_URL, timeout=60) as response:
            archive = response.read()
    report = prepare(archive, args.output_dir)
    print(json.dumps({"original": report["original"], "deduplicated": report["deduplicated"],
                      "duplicate_groups": len(report["duplicate_groups"]),
                      "cross_split_groups": sum(g["cross_split"] for g in report["duplicate_groups"]),
                      "conflicting_label_groups": sum(g["conflicting_labels"] for g in report["duplicate_groups"]),
                      "excluded_rows": len(report["excluded"]),
                      "report": str(args.output_dir / "report.json")}, indent=2))


if __name__ == "__main__":
    main()
