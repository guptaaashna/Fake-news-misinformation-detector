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

## Next step

Train majority-class and TF-IDF baselines on `data/liar/deduplicated/train.jsonl`.
Measure tokenizer lengths/truncation before fine-tuning the encoder and head.
Select checkpoints using validation macro-F1 and reserve test examples for final
evaluation. Word lengths in the report are not tokenizer lengths. Fit learned
preprocessing only on training data; do not rebalance validation or test sets.
No training or evaluation is performed by dataset preparation.

References:
- https://huggingface.co/distilbert/distilbert-base-uncased
- https://huggingface.co/docs/transformers/en/tasks/sequence_classification
- https://huggingface.co/datasets/ucsbai/liar/blob/main/liar.py
