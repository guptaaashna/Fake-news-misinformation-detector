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

## Next step

Prepare the LIAR train/validation/test splits, check duplicates, train baselines,
then fine-tune the encoder and classification head. Select a checkpoint using
validation macro-F1 and reserve test data for final evaluation.

References:
- https://huggingface.co/distilbert/distilbert-base-uncased
- https://huggingface.co/docs/transformers/en/tasks/sequence_classification
- https://huggingface.co/datasets/ucsbai/liar/blob/main/liar.py
