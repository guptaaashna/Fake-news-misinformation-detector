# TruthShield AI

A Python and Streamlit research prototype for examining news article claims, model predictions, related news, and basic website-security indicators.

## What it does

- Extracts readable text from an article URL.
- Displays text statistics and word-frequency charts.
- Selects up to five factual-looking candidate claims using spaCy-assisted rules.
- Predicts six LIAR labels with a DistilBERT classifier adapted using LoRA.
- Retrieves related news through GDELT and displays evidence status separately from the model prediction.
- Checks HTTPS, DNS resolution, certificate availability, and suspicious URL patterns.
- Saves analyses in SQLite and exports HTML, CSV, and PDF reports.

**Scope:** This is an experimental analysis tool, not a reliable automated fact-checking service. Current GDELT results are marked neutral; related news does not automatically establish support or contradiction. Model predictions do not feed into the heuristic scores.

## Saved evaluation results

Results on the deduplicated LIAR held-out test split (1,283 statements):

| Method | Accuracy | Macro-F1 | Weighted-F1 |
| --- | ---: | ---: | ---: |
| Majority baseline | 20.81% | 0.0574 | 0.0717 |
| TF-IDF + logistic regression | 24.55% | 0.2121 | 0.2342 |
| DistilBERT + LoRA | 28.37% | 0.2684 | 0.2769 |

These saved results show improvement over the evaluated baselines, but limited overall classification performance. They do not measure the accuracy of the complete application on arbitrary news articles.

## Quick start (PowerShell)

From the repository root, create an environment and install dependencies:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
cd "fnd drc project"
python -m pip install -r requirements.txt
python -m spacy download en_core_web_sm
python -c "import nltk; [nltk.download(resource) for resource in ('punkt', 'punkt_tab', 'stopwords', 'wordnet', 'omw-1.4')]"
python -m streamlit run app.py
```

Open the local address printed by Streamlit, usually `http://localhost:8501`. The app loads the included local adapter; the public DistilBERT base checkpoint may be downloaded on first use.

For full setup, architecture, training, tests, and limitations, see the [project README](fnd%20drc%20project/README.md).

## Repository layout

- [Application](fnd%20drc%20project/app.py): Streamlit entry point.
- [Services](fnd%20drc%20project/services): extraction, claims, inference, evidence, security, scoring, storage, and exports.
- [Training](fnd%20drc%20project/training): LIAR preparation, baselines, LoRA training, and evaluation.
- [Tests](fnd%20drc%20project/tests): automated behavior checks.
- [Saved LoRA metrics](fnd%20drc%20project/models/distilbert-liar-lora/test_metrics.json): recorded test results.

The LoRA adapter and tokenizer are included in this repository. Downloaded base checkpoints, LIAR data, caches, and generated databases remain local. The current dashboard accepts URLs; pasted-text input and image/OCR analysis are not implemented as working dashboard inputs.
