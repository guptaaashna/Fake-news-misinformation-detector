"""Print LIAR claims and their DistilBERT tokens for a quick inspection."""

import json
from pathlib import Path

from transformers import AutoTokenizer

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data" / "liar" / "deduplicated"
TOKENIZER_DIR = ROOT / "models" / "distilbert-liar-initialized"
SPLITS = ("train", "validation", "test")
EXAMPLES_TO_PRINT = 3


def load_claims(path):
    """Read statement text from a JSONL split."""
    claims = []
    with path.open("r", encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(f"Invalid JSON in {path}, line {line_number}: {error}") from error
            statement = row.get("statement")
            if not isinstance(statement, str):
                raise ValueError(f"Missing string 'statement' in {path}, line {line_number}.")
            claims.append(statement)
    return claims


def main():
    missing_files = [DATA_DIR / f"{split}.jsonl" for split in SPLITS
                     if not (DATA_DIR / f"{split}.jsonl").is_file()]
    if missing_files:
        for path in missing_files:
            print(f"Dataset file not found: {path}")
        return

    claims_by_split = {
        split: load_claims(DATA_DIR / f"{split}.jsonl")
        for split in SPLITS
    }
    print("Loaded claim counts:")
    for split, claims in claims_by_split.items():
        print(f"  {split}: {len(claims)}")

    if not TOKENIZER_DIR.is_dir():
        print(f"Local DistilBERT tokenizer not found: {TOKENIZER_DIR}")
        print("Run training/setup_model.py first to create the local checkpoint and tokenizer.")
        return

    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_DIR, local_files_only=True)
    for split, claims in claims_by_split.items():
        print(f"\n{split} examples:")
        if not claims:
            print("  No claims in this split.")
            continue
        for claim in claims[:EXAMPLES_TO_PRINT]:
            tokens = tokenizer.tokenize(claim)
            token_ids = tokenizer.convert_tokens_to_ids(tokens)
            print(f"  Claim: {claim}")
            print(f"  Tokens: {tokens}")
            print(f"  Token IDs: {token_ids}")


if __name__ == "__main__":
    main()
