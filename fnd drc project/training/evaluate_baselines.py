"""Compare majority and TF-IDF baselines on the held-out LIAR test split."""

import argparse
import json
from pathlib import Path

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.pipeline import Pipeline

from training.prepare_lora_data import DATA_DIR


def load_split(path):
    texts, labels = [], []
    with path.open("r", encoding="utf-8") as source:
        for line_number, line in enumerate(source, start=1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(f"Invalid JSON in {path}, line {line_number}: {error}") from error
            texts.append(row["statement"])
            labels.append(row["label"])
    if not texts:
        raise ValueError(f"No examples found in {path}")
    return texts, labels


def metrics(y_true, y_pred):
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "macro_f1": f1_score(y_true, y_pred, average="macro", zero_division=0),
        "weighted_f1": f1_score(y_true, y_pred, average="weighted", zero_division=0),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-json", type=Path)
    args = parser.parse_args()

    train_texts, train_labels = load_split(DATA_DIR / "train.jsonl")
    test_texts, test_labels = load_split(DATA_DIR / "test.jsonl")

    majority_label = max(set(train_labels), key=train_labels.count)
    majority_predictions = [majority_label] * len(test_labels)
    tfidf = Pipeline([
        ("tfidf", TfidfVectorizer(ngram_range=(1, 2), min_df=2, max_features=200_000)),
        ("classifier", LogisticRegression(max_iter=1000, random_state=42)),
    ])
    tfidf.fit(train_texts, train_labels)
    tfidf_predictions = tfidf.predict(test_texts)

    result = {
        "dataset": "LIAR deduplicated held-out test split",
        "train_examples": len(train_texts),
        "test_examples": len(test_texts),
        "majority_label_from_train": majority_label,
        "majority": metrics(test_labels, majority_predictions),
        "tfidf_logistic_regression": metrics(test_labels, tfidf_predictions),
    }
    print(json.dumps(result, indent=2))

    if args.output_json:
        output_path = args.output_json.resolve()
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
        print(f"Saved metrics to {output_path}")


if __name__ == "__main__":
    main()
