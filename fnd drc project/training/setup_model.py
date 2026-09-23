"""Download DistilBERT and initialize (not fine-tune) a LIAR classifier."""
import argparse
from datetime import datetime, timezone
from importlib.metadata import version
import json
import os
from pathlib import Path
import platform

ROOT = Path(__file__).resolve().parents[1]
MODEL_ID = "distilbert/distilbert-base-uncased"
# Match the original ucsbai/liar ClassLabel order, not alphabetical order.
LABELS = ("false", "half-true", "mostly-true", "true", "barely-true", "pants-fire")
ID2LABEL = dict(enumerate(LABELS))
LABEL2ID = {label: index for index, label in ID2LABEL.items()}
DEFAULT_OUTPUT = ROOT / "models" / "distilbert-liar-initialized"
MAX_LENGTH = 128
SEED = 42


def check_model(model, tokenizer, backward=False):
    """Check tensor shapes and optional gradients without an optimizer update."""
    import torch

    if model.config.id2label != ID2LABEL or model.config.label2id != LABEL2ID:
        raise ValueError("The checkpoint does not have the expected six LIAR labels.")
    if model.config.model_type != "distilbert":
        raise ValueError("Expected a DistilBERT checkpoint.")
    if not all(parameter.requires_grad for parameter in model.parameters()):
        raise ValueError("Both the encoder and classification head must be trainable.")
    samples = [
        "The agency opened a new office in Delhi.",
        "This is a long input sentence. " * 200,
    ]
    inputs = tokenizer(
        samples, padding=True, truncation=True, max_length=MAX_LENGTH,
        return_tensors="pt",
    )
    model.eval()
    with torch.no_grad():
        logits = model(**inputs).logits
    if logits.shape != (2, len(LABELS)) or not torch.isfinite(logits).all():
        raise RuntimeError("Forward-pass check failed.")
    if inputs["input_ids"].shape[1] != MAX_LENGTH:
        raise RuntimeError("Long-input truncation check failed.")

    if backward:
        # Synthetic targets only test backpropagation; they are not dataset labels.
        model.train()
        loss = model(**inputs, labels=torch.tensor([0, 1])).loss
        if not torch.isfinite(loss):
            raise RuntimeError("Training loss is not finite.")
        loss.backward()
        for name, parameter in (
            ("encoder", model.distilbert.embeddings.word_embeddings.weight),
            ("classifier", model.classifier.weight),
        ):
            gradient = parameter.grad
            if gradient is None or not torch.isfinite(gradient).all() or not gradient.abs().sum() > 0:
                raise RuntimeError(f"No valid gradients reached the {name}.")
        model.zero_grad(set_to_none=True)
        model.eval()
    return logits


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--revision", default="main", help="Hugging Face commit or revision to download.")
    parser.add_argument("--verify-only", action="store_true", help="Check saved files offline; do not download or save.")
    args = parser.parse_args()
    output = args.output_dir.resolve()

    # Keep downloaded files inside an ignored project directory.
    os.environ.setdefault("HF_HOME", str(ROOT / ".cache" / "huggingface"))
    os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")
    if args.verify_only:
        os.environ["HF_HUB_OFFLINE"] = "1"

    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer, set_seed

    torch.set_num_threads(min(4, os.cpu_count() or 1))
    set_seed(SEED)
    if args.verify_only:
        metadata = json.loads((output / "setup_metadata.json").read_text(encoding="utf-8"))
        if metadata.get("status") != "initialized_not_fine_tuned":
            raise ValueError("This command verifies the initialized setup checkpoint only.")
        model = AutoModelForSequenceClassification.from_pretrained(output, local_files_only=True)
        tokenizer = AutoTokenizer.from_pretrained(output, local_files_only=True)
        check_model(model, tokenizer)
        print("PASS: offline reload, six labels, finite outputs, trainable weights, and truncation.")
        print("This classification head is not fine-tuned; its predictions are not meaningful.")
        return

    if output.exists() and any(output.iterdir()):
        parser.error("Output directory is not empty. Use --verify-only or choose a new --output-dir.")

    from huggingface_hub import HfApi
    # Resolve main once so tokenizer and weights come from the same immutable revision.
    revision = HfApi().model_info(MODEL_ID, revision=args.revision).sha
    print(f"Downloading {MODEL_ID} at {revision}", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, revision=revision)
    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_ID, revision=revision, num_labels=len(LABELS),
        id2label=ID2LABEL, label2id=LABEL2ID, use_safetensors=True,
    )
    model.config.problem_type = "single_label_classification"
    model.config.truthshield_training_status = "initialized_not_fine_tuned"
    model.config.truthshield_max_length = MAX_LENGTH
    original_logits = check_model(model, tokenizer, backward=True)

    output.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(output)
    tokenizer.save_pretrained(output)
    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    del model
    reloaded = AutoModelForSequenceClassification.from_pretrained(output, local_files_only=True)
    reloaded_tokenizer = AutoTokenizer.from_pretrained(output, local_files_only=True)
    saved_logits = check_model(reloaded, reloaded_tokenizer)
    if not torch.allclose(original_logits, saved_logits, atol=1e-6):
        raise RuntimeError("Saved checkpoint outputs differ from the initialized model.")

    metadata = {
        "status": "initialized_not_fine_tuned",
        "base_model": MODEL_ID,
        "resolved_revision": revision,
        "labels": list(LABELS),
        "seed": SEED,
        "max_length": MAX_LENGTH,
        "parameter_count": parameter_count,
        "all_parameters_trainable": True,
        "setup_device": "cpu",
        "cuda_available": torch.cuda.is_available(),
        "python": platform.python_version(),
        "packages": {name: version(name) for name in (
            "torch", "transformers", "accelerate", "datasets", "scikit-learn"
        )},
        "verified_at": datetime.now(timezone.utc).isoformat(),
        "checks": ["forward", "long_input_truncation", "encoder_and_head_gradients", "save_reload_equivalence"],
        "optimizer_steps": 0,
    }
    (output / "setup_metadata.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(metadata, indent=2))
    print(f"Saved initialized model and tokenizer to {output}")
    print("Setup complete. No fine-tuning or accuracy evaluation has been performed.")


if __name__ == "__main__":
    main()
