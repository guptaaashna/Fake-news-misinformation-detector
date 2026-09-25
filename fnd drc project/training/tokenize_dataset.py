import json
from pathlib import Path

from transformers import AutoTokenizer

ROOT = Path(__file__).resolve().parents[1]

MODEL_ID = "distilbert-base-uncased"
MAX_LENGTH = 128

INPUT_DIR = ROOT / "data" / "liar" / "deduplicated"
OUTPUT_DIR = ROOT / "data" / "liar" / "tokenized"

tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)


def tokenize_split(split):
    input_file = INPUT_DIR / f"{split}.jsonl"
    output_file = OUTPUT_DIR / f"{split}.jsonl"

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    with input_file.open("r", encoding="utf-8") as infile, \
         output_file.open("w", encoding="utf-8") as outfile:

        for line in infile:
            record = json.loads(line)

            encoded = tokenizer(
                record["statement"],
                truncation=True,
                max_length=MAX_LENGTH
            )

            tokenized_record = {
                "id": record["id"],
                "statement": record["statement"],
                "label": record["label"],
                "label_name": record["label_name"],
                "input_ids": encoded["input_ids"],
                "attention_mask": encoded["attention_mask"]
            }

            outfile.write(json.dumps(tokenized_record) + "\n")

    print(f"Tokenized {split}: {output_file}")


def main():
    for split in ("train", "validation", "test"):
        tokenize_split(split)


if __name__ == "__main__":
    main()