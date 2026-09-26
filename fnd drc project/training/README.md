# DistilBERT model setup

This step downloads pretrained DistilBERT, initializes a six-class head, and saves a
local starting checkpoint. It does not train on LIAR, evaluate accuracy, or connect
the model to the dashboard. The new head has random weights, so its predictions
must not be displayed as truth assessments.

## Environment

Run commands from `fnd drc project`. A project-local `.venv` is used so the
system Python is not modified. The versions in `requirements.txt` are the model
environment dependencies; the application's dependencies are separate.

```powershell
# Only needed if .venv does not already exist:
python -m venv .venv

.\.venv\Scripts\python.exe -m pip install -r training/requirements.txt
.\.venv\Scripts\python.exe -m training.setup_model
```

The first setup requires internet access to Hugging Face. No account or API key is
required for this public model. CPU is sufficient for setup; CUDA availability is
recorded for the subsequent training step.

The script refuses to overwrite an existing checkpoint. To check an existing
setup entirely offline:

```powershell
.\.venv\Scripts\python.exe -m training.setup_model --verify-only
```

To reproduce the base weights, pass `--revision <resolved_revision>` from the
saved metadata, and use a different `--output-dir` for a fresh initialization.

## What is saved

`models/distilbert-liar-initialized/` contains model weights, model configuration,
tokenizer files, and `setup_metadata.json`. The metadata records the exact base
model revision, seed, package versions, parameter count, and completed checks.

The setup verifies a two-item forward pass, truncation at 128 tokens, gradients
reaching both the pretrained encoder and new head, and identical outputs after
saving/reloading. The synthetic labels used for the gradient check are only a
technical test. No optimizer updates are applied.

Model weights and the local Hugging Face cache are ignored by Git. Keep the scripts
and requirements in Git; regenerate the large files when needed.

## Label mapping

This order matches the original Hugging Face LIAR dataset definition:

| ID | Label |
|---|---|
| 0 | false |
| 1 | half-true |
| 2 | mostly-true |
| 3 | true |
| 4 | barely-true |
| 5 | pants-fire |

The data-preparation step must use exactly this mapping. These are dataset classes,
not the app's evidence statuses (supported, contradicted, insufficient).

## Dataset preparation

From `fnd drc project`, run:

```powershell
.\.venv\Scripts\python.exe -m training.prepare_dataset
```

This downloads the original UCSB LIAR ZIP, checks its pinned SHA-256, and writes
`data/liar/` (ignored by Git). Subsequent runs use the cached ZIP offline and
verify that generated contents are identical. A different existing output is
never overwritten. To supply an already downloaded archive:

```powershell
.\.venv\Scripts\python.exe -m training.prepare_dataset --archive C:\path\liar_dataset.zip
```

- `raw/`: unchanged ZIP, original TSVs, and the author's README.
- `original/`: original split membership as JSONL, with validated fields.
- `deduplicated/`: recommended working splits with repeated statements removed.
- `report.json`: provenance, checksums, class counts, word lengths, duplicate
  groups, conflicting labels, and every excluded record ID.

Each JSONL row contains `id`, `statement`, numeric `label`, `label_name`,
`source_split`, and `source_line`. Only `statement` should enter the model;
other fields support targets and traceability. Text is preserved exactly:
no stemming, stopword removal, punctuation removal, or lowercasing. These TSVs
must be read with `csv.QUOTE_NONE`; CSV quote handling incorrectly merges rows.

Duplicate matching uses Unicode NFKC, case folding, and collapsed whitespace,
without modifying saved text. Test takes priority over validation, then train;
the earliest source row wins within a split. This policy uses no labels.
Conflicting labels are reported without relabeling. Original files remain
available for benchmark comparisons; deduplicated results must be identified
as a modified benchmark. Semantic duplicates and speaker overlap are not checked.

The author's README permits **research use only** and retains the original
sources' copyright. Raw and prepared data remain local rather than being
redistributed in Git. See [the preparation results](DATASET_REPORT.md).

Run the preparation regression checks with:

```powershell
.\.venv\Scripts\python.exe -m unittest discover -s tests -p test_dataset_preparation.py
```

## Tokenization and batches

After model setup and dataset preparation, run offline:

```powershell
.\.venv\Scripts\python.exe -m training.tokenize_dataset
```

This verifies prepared input hashes and loads the tokenizer from
`models/distilbert-liar-initialized/` with `local_files_only=True`. The pretrained
uncased WordPiece tokenizer performs its own normalization and subword splitting;
we do not fit a new vocabulary or remove punctuation, negation, or numbers.
Lengths include `[CLS]` and `[SEP]`. The existing 128-token limit is checked
against training lengths; held-out lengths are diagnostics, not tuning criteria.
Right truncation preserves special tokens. No prepared claim exceeds 128 tokens.

Outputs in the ignored `data/liar/tokenized/` directory:

- `train.jsonl`, `validation.jsonl`, `test.jsonl`: ID, variable-length `input_ids`,
  `attention_mask`, and integer `labels`. No padding is stored.
- `report.json`: full-length statistics and truncation counts, source and output
  hashes, tokenizer file hashes and base revision, package versions, and batch settings.

The script refuses to overwrite different outputs; use `--output-dir` for a new
version. Identical reruns are safe. `--data-dir` and `--model-dir` accept alternate
prepared-data and local checkpoint directories.

To consume these files in a future training loop:

```python
from training.tokenize_dataset import load_dataloader

train_loader = load_dataloader("train")  # batch size 16, shuffled with seed 42
validation_loader = load_dataloader("validation")  # batch size 32, source order
# Test is loaded explicitly only for final evaluation.
# batch = next(iter(train_loader))
# outputs = model(**batch)
```

`DataCollatorWithPadding` pads to the longest sequence in each batch, on the right.
The attention mask is 1 for real tokens and 0 for padding. IDs are excluded from
model inputs. Labels are PyTorch integer tensors for six-class cross-entropy.
The final partial batch is retained, and `num_workers=0` works on Windows.
Batch sizes can be overridden with `batch_size=`; 16/32 are starting defaults,
not a measured guarantee of training memory capacity. Loaders verify tokenized
data and tokenizer hashes before use. Training reshuffles each epoch while
remaining reproducible across fresh runs with the same seed.

Run regression checks without network or downloaded model files:

```powershell
.\.venv\Scripts\python.exe -m unittest discover -s tests -p test_tokenization.py
```

See [the tokenization results](TOKENIZATION_REPORT.md). Tokenization does not train
the model or evaluate its predictions on any split.

## Next step

Train majority-class and TF-IDF baselines on `data/liar/deduplicated/train.jsonl`.
Tokenization and batch preparation are complete; then fine-tune the encoder and head.
Select checkpoints using validation macro-F1 and reserve test examples for final
evaluation. Word lengths in the report are not tokenizer lengths. Fit learned
preprocessing only on training data; do not rebalance validation or test sets.
No training or evaluation is performed by dataset preparation.

References:
- https://huggingface.co/distilbert/distilbert-base-uncased
- https://huggingface.co/docs/transformers/en/tasks/sequence_classification
- https://huggingface.co/datasets/ucsbai/liar/blob/main/liar.py
