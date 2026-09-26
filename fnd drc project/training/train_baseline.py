import json
import os

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
import joblib


def load_data(path):
    texts = []
    labels = []

    with open(path, "r", encoding="utf-8") as file:
        for line in file:
            row = json.loads(line)
            texts.append(row["statement"])
            labels.append(row["label"])

    return texts, labels


train_path = "data/liar/deduplicated/train.jsonl"
validation_path = "data/liar/deduplicated/validation.jsonl"

train_texts, train_labels = load_data(train_path)
validation_texts, validation_labels = load_data(validation_path)

model = Pipeline([
    ("tfidf", TfidfVectorizer()),
    ("classifier", LogisticRegression(max_iter=1000))
])

print("Training baseline model...")

model.fit(train_texts, train_labels)

print("Training complete.")
print("Training examples:", len(train_texts))
print("Validation examples:", len(validation_texts))

os.makedirs("models", exist_ok=True)

joblib.dump(model, "models/tfidf_logistic_baseline.joblib")

print("Baseline model saved.")