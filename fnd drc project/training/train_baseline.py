"""Train TF-IDF + logistic regression and evaluate validation only."""
import argparse
import hashlib
from importlib.metadata import version
import json
from pathlib import Path
import warnings

import joblib
from sklearn.exceptions import ConvergenceWarning
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from sklearn.pipeline import Pipeline
from training.setup_model import ROOT, LABELS, LABEL2ID, SEED


def load_data(path):
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    for row in rows:
        if not row["statement"].strip() or LABEL2ID[row["label_name"]] != row["label"]:
            raise ValueError(f"Invalid statement or label in {path}")
    return [r["statement"] for r in rows], [r["label"] for r in rows]


def build_model():
    return Pipeline([
        ("tfidf", TfidfVectorizer()),
        ("classifier", LogisticRegression(max_iter=1000, random_state=SEED)),
    ])


def evaluate(labels, predictions):
    ids = list(range(len(LABELS)))
    return {"accuracy": accuracy_score(labels, predictions),
            "macro_f1": f1_score(labels, predictions, labels=ids, average="macro", zero_division=0),
            "per_class": classification_report(labels, predictions, labels=ids,
                                               target_names=LABELS, output_dict=True, zero_division=0),
            "confusion_matrix": confusion_matrix(labels, predictions, labels=ids).tolist(),
            "confusion_matrix_order": list(LABELS)}


def run(data_dir, output_dir):
    if output_dir.exists() and any(output_dir.iterdir()):
        raise ValueError("Output directory is not empty; choose a new --output-dir to preserve prior runs.")
    manifest = json.loads((data_dir / "report.json").read_text())
    inputs, hashes = {}, {}
    for split in ("train", "validation"):
        path = data_dir / "deduplicated" / f"{split}.jsonl"
        hashes[split] = hashlib.sha256(path.read_bytes()).hexdigest()
        if hashes[split] != manifest["sha256"][f"deduplicated/{split}.jsonl"]:
            raise ValueError(f"Prepared data checksum mismatch: {split}")
        inputs[split] = load_data(path)
    train_texts, train_labels = inputs["train"]
    validation_texts, validation_labels = inputs["validation"]
    model = build_model()
    with warnings.catch_warnings():
        warnings.simplefilter("error", ConvergenceWarning)
        model.fit(train_texts, train_labels)
    predictions = model.predict(validation_texts)
    report = {"model": "TF-IDF + LogisticRegression", "evaluation_split": "validation",
              "test_evaluated": False, "seed": SEED, "label2id": LABEL2ID,
              "train_rows": len(train_texts), "validation_rows": len(validation_texts),
              "input_sha256": hashes,
              "packages": {name: version(name) for name in ("scikit-learn", "numpy", "joblib")},
              "configuration": {"tfidf": "sklearn defaults, fitted on train only",
                                "logistic_regression": "sklearn defaults, max_iter=1000, random_state=42"},
              "validation": evaluate(validation_labels, predictions)}
    output_dir.mkdir(parents=True, exist_ok=True)
    model_path = output_dir / "model.joblib"
    joblib.dump(model, model_path)
    reloaded = joblib.load(model_path)
    if not (reloaded.predict(validation_texts) == predictions).all():
        raise RuntimeError("Saved baseline predictions changed after reload")
    report["model_sha256"] = hashlib.sha256(model_path.read_bytes()).hexdigest()
    report["save_reload_verified"] = True
    (output_dir / "validation_metrics.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=ROOT / "data" / "liar")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "models" / "baseline-validation-v1")
    args = parser.parse_args()
    print(json.dumps(run(args.data_dir, args.output_dir), indent=2))


if __name__ == "__main__":
    main()
