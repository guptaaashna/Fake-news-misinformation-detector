# TruthShield AI

TruthShield AI is a local research prototype built with Python and Streamlit. It extracts an article from a URL, selects factual-looking claims, displays a trained classifier's assessment, retrieves related news, and checks basic website-security indicators. Analyses can be saved locally and downloaded as reports.

**The model assessment, evidence status, and security indicators are different signals.** The current application does not combine them into a validated final true/false verdict.

## Current functionality

| Component | Implemented behavior |
| --- | --- |
| Input | Article URL through the Streamlit dashboard |
| Extraction | Requests and BeautifulSoup extract readable article text, with structured-data and Times of India fallbacks |
| Text processing | NLTK tokenization, stop-word filtering, lemmatization, statistics, and frequency charts |
| Candidate claims | spaCy-assisted heuristic ranking; up to five sentences, with a minimum length of 35 characters |
| Classifier | DistilBERT base uncased with a local LoRA adapter; six LIAR labels |
| Evidence retrieval | GDELT related-news search; all retrieved items are labeled neutral |
| Evidence status | Rules inspect explicit supports/contradicts relationships; neutral-only, missing, or conflicting evidence is insufficient |
| Security | HTTPS, DNS resolution, certificate retrieval, and suspicious URL-pattern checks |
| Scores | Four manually weighted prototype indicators, with reasons |
| History | SQLite storage in `data/truthshield.db` |
| Reports | HTML, CSV, and PDF downloads |

The evidence module's legacy description mentions Fact Check Tools, but no active Google Fact Check Tools integration is implemented. Image/OCR dependencies are listed, but `services/image.py` is a placeholder. Some messages mention pasted text, but the current dashboard exposes URL input only.

## How the application works

```text
Article URL
  -> download and extract article text
  -> descriptive text processing
  -> select up to five candidate claims
       -> DistilBERT + LoRA prediction (displayed separately)
       -> GDELT contextual news -> evidence status
  -> basic URL security checks
  -> heuristic scores
  -> Streamlit dashboard + SQLite history + report downloads
```

`services/pipeline.py` coordinates the claim analyses and security checks. A model-loading failure is recorded without discarding available evidence results. Model predictions are not used by `services/scoring.py`.

### Model assessment versus evidence verification

The classifier learns from claim text and returns a label, the winning softmax score, and scores for all six classes. It does not retrieve facts while predicting.

The evidence verifier returns:

- **Supported:** at least one supporting item and no contradicting items.
- **Contradicted:** at least one contradicting item and no supporting items.
- **Insufficient:** no evidence, neutral-only evidence, or both supporting and contradicting items.

The live GDELT retrieval layer assigns `relation = "neutral"`. It does not download the returned articles' full text or infer semantic support/contradiction. Consequently, live retrieved news supplies context but does not establish supported/contradicted verdicts. Those verifier branches are available for explicitly labeled evidence supplied by another caller or tests.

A model prediction such as `mostly-true` can coexist with `insufficient` evidence. Neither missing evidence nor a model label establishes real-world truthfulness. Softmax confidence is not a calibrated probability that a claim is true.

## Model and dataset

- Base checkpoint: `distilbert/distilbert-base-uncased`.
- Task: supervised, single-label classification of claim text.
- Dataset: original UCSB LIAR release, prepared as deduplicated JSONL splits.
- Input: statement text only; speaker metadata and external evidence are not model inputs.
- Label IDs: `0=false`, `1=half-true`, `2=mostly-true`, `3=true`, `4=barely-true`, `5=pants-fire`.
- Sequence limit: 128 tokenizer tokens, with truncation and dynamic padding during training.
- LoRA: rank 8, alpha 16, dropout 0.1, targeting `q_lin` and `v_lin`.
- Training: 3 epochs, batch size 8, AdamW, learning rate 0.0002, weight decay 0.01, gradient clipping 1.0, seed 42.
- Checkpoint selection: best validation macro-F1; saved metadata records epoch 3 and macro-F1 0.2520.

LoRA adapts selected transformations while keeping pretrained base weights frozen; the saved adapter also includes classifier-related weights. `training.setup_model` initializes the task classifier but does not fine-tune it. Training is performed separately by `training.train_lora`.

### Prepared dataset sizes

| Split | Original | Deduplicated |
| --- | ---: | ---: |
| Train | 10,269 | 10,243 |
| Validation | 1,284 | 1,284 |
| Test | 1,283 | 1,283 |
| Total | 12,836 | 12,810 |

Preparation verifies the archive's SHA-256 checksum and validates required fields, label names, and row structure. Exact normalized duplicates are detected using Unicode normalization, casefolding, and collapsed whitespace, without changing model-input text. Retention prioritizes test, then validation, then training. The preparation report records 26 removed training rows; near-duplicates and speaker overlap are not eliminated.

See [the dataset preparation report](training/DATASET_REPORT.md). LIAR is an older political-statement benchmark, so performance on it does not establish performance on arbitrary contemporary news.

### Saved test results

These are recorded results, not a new evaluation run. All methods below were evaluated on the deduplicated held-out test split of 1,283 examples.

| Method | Accuracy | Macro-F1 | Weighted-F1 |
| --- | ---: | ---: | ---: |
| Majority baseline | 20.81% | 0.0574 | 0.0717 |
| TF-IDF + logistic regression | 24.55% | 0.2121 | 0.2342 |
| DistilBERT + LoRA | 28.37% | 0.2684 | 0.2769 |

The LoRA model improves accuracy over the evaluated TF-IDF baseline by approximately 3.82 percentage points, but its overall performance remains limited. Macro-F1 gives each class equal weight, which matters because LIAR classes are imbalanced.

- LoRA metrics: [test_metrics.json](models/distilbert-liar-lora/test_metrics.json).
- Training settings: [training_metadata.json](models/distilbert-liar-lora/training_metadata.json).
- The local baseline metrics were generated by `training.evaluate_baselines`, which trains a unigram/bigram TF-IDF baseline. Its configuration differs from the default TF-IDF configuration in `training.train_baseline`.

## Setup (PowerShell)

Use Python 3.11 or later and a compatible installation of the pinned ML dependencies. Internet access is required for article extraction, GDELT retrieval, NLP resource downloads, and the first base-model download.

### 1. Create an environment

Run from the repository root:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
cd "fnd drc project"
```

If an existing environment is already activated, use it instead of creating another.

### 2. Install dependencies and language resources

From `fnd drc project`:

```powershell
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m spacy download en_core_web_sm
python -c "import nltk; [nltk.download(resource) for resource in ('punkt', 'punkt_tab', 'stopwords', 'wordnet', 'omw-1.4')]"
```

For training and evaluation, also install:

```powershell
python -m pip install -r training/requirements.txt
```

### 3. Run the dashboard

```powershell
python -m streamlit run app.py
```

Open the address printed by Streamlit, usually `http://localhost:8501`.

If PowerShell activation is unavailable, invoke the environment's interpreter directly. With the environment at the repository root, run from `fnd drc project`:

```powershell
..\.venv\Scripts\python.exe -m pip install -r requirements.txt
..\.venv\Scripts\python.exe -m streamlit run app.py
```

### Model files and first use

The repository includes `models/distilbert-liar-lora/` with the adapter weights, adapter configuration, tokenizer files, and saved metrics. Runtime inference loads the tokenizer locally and attaches the adapter to the public DistilBERT base checkpoint. The base checkpoint may be downloaded from Hugging Face on first use; later use can reuse cached files.

The adapter configuration records the original training machine's checkpoint path. Runtime inference explicitly supplies the public base model, so that historical path is not the intended runtime dependency.

## Training and evaluation

Run the following from `fnd drc project` with the intended environment active. Dataset preparation and initialization are separate from training.

```powershell
python -m training.prepare_dataset
python -m training.setup_model
python -m training.train_baseline
```

The initializer refuses to overwrite a nonempty output directory. It creates `models/distilbert-liar-initialized/` for local training and evaluation. The included trained adapter directory is also protected against overwrite, so use a fresh output directory for a new experiment:

```powershell
python -m training.train_lora --output-dir models/distilbert-liar-lora-new
python -m training.evaluate_baselines --output-json models/baseline-test-metrics.json
python -m training.evaluate_lora --adapter-dir models/distilbert-liar-lora-new --output-json models/distilbert-liar-lora-new/test_metrics.json
```

To evaluate the included adapter after preparing the data and initializing the local base checkpoint:

```powershell
python -m training.evaluate_lora --adapter-dir models/distilbert-liar-lora
```

Training uses train and validation splits; its checkpoint selection does not use the test split. Keep test evaluation separate from tuning. Retraining into a new directory does not change the dashboard's default adapter path automatically.

## Tests

From the repository root or the application folder, using the environment with the required dependencies installed:

```powershell
python -m pytest
```

Tests cover extraction, text processing, claims, evidence relationships, security, scores, storage, exports, pipeline behavior, and model-inference interfaces. Many external calls and model outputs are mocked. Passing these tests checks software behavior; it does not establish live service availability, successful real-model loading, or prediction accuracy.

## Prototype scores and known limitations

The dashboard displays information risk, cyber risk, source credibility, and evidence strength on a 0-100 scale. Weights are manually specified, not learned or scientifically calibrated. Model predictions are not included in these calculations.

Known implementation limitations:

- The sensational-language scorer reads a claim's `text` field, while normal pipeline claim records use `claim`, so those checks generally receive empty text.
- Insufficient and neutral-only claims are counted separately and can overlap, inflating the weak-evidence component.
- A zero information-risk score with no extracted claims does not mean an article is verified.
- Distinct evidence URLs do not necessarily represent independent publishers.
- Source-credibility rules describe available evidence context, not a validated reputation assessment of the original publisher.
- Current retrieval cannot infer evidence support or contradiction; no active fact-check API or natural-language-inference component is implemented.
- Website checks do not perform malware scanning. Threat reputation remains `unknown`; HTTPS is not proof of trustworthy content.
- Long claims can be truncated, and selecting five candidates can omit important statements.
- Current reports do not explicitly include the classifier label/confidence, although the dashboard and stored full-result JSON contain model predictions.
- Article extraction can fail on blocked, paywalled, dynamically rendered, or unfamiliar pages.

## Troubleshooting

- **spaCy model missing:** install `en_core_web_sm` in the same environment used to run Streamlit. Basic sentence-segmentation fallbacks are available.
- **VS Code spaCy extension errors:** editor-extension errors are separate from whether the application can import spaCy and load its language model.
- **LoRA prediction unavailable:** inspect the displayed original error, adapter files, dependencies, base-model access, and environment restrictions. The pipeline preserves available evidence results after a model error.
- **Windows Application Control blocks a SciPy compiled module such as `_pava_pybind`:** this is an environment-policy restriction. It may require an administrator-approved environment or policy allowance; reinstalling spaCy does not resolve that restriction.
- **No evidence:** a failed request or no matching GDELT results produces insufficient evidence, not a false verdict.
- **Extraction fails:** the current dashboard has no working paste-text fallback, despite messages suggesting one.

## Project structure

```text
fnd drc project/
|-- app.py
|-- requirements.txt
|-- services/     # extraction, processing, claims, inference, evidence, security,
|                 # scoring, pipeline, storage, export, and OCR placeholder
|-- ui/           # dashboard and display components
|-- training/     # preparation, tokenization, baselines, LoRA, evaluation
|-- tests/        # automated checks
|-- models/       # included adapter; locally generated base checkpoints
|-- data/         # local LIAR preparation outputs and SQLite history
`-- README.md
```

Downloaded base checkpoints, datasets, caches, databases, and secrets should remain local. The LoRA adapter/tokenizer and their metadata are currently tracked in Git.

## Future work

- Correct and validate score calculations.
- Retrieve evidence passages and evaluate semantic support/contradiction.
- Add reliable fact-check and official-source integrations.
- Evaluate on relevant contemporary news and calibrate confidence.
- Improve smaller-class recall and compare additional training settings fairly.
- Implement working pasted-text input and OCR if those features are needed.
- Include classifier assessments in exported reports.
- Harden URL fetching before public deployment, including private-network and redirect protections.

## References

- [DistilBERT documentation](https://huggingface.co/docs/transformers/model_doc/distilbert)
- [LoRA paper](https://arxiv.org/abs/2106.09685)
- [Original LIAR paper](https://aclanthology.org/P17-2067/)
- [LIAR dataset](https://huggingface.co/datasets/ucsbai/liar)
- [scikit-learn F1 documentation](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.f1_score.html)
