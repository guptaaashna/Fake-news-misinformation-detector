"""Cached local inference for the trained TruthShield LIAR LoRA adapter."""

from pathlib import Path

import streamlit as st

from training.setup_model import ID2LABEL, LABEL2ID, LABELS, MODEL_ID, MAX_LENGTH


ROOT = Path(__file__).resolve().parents[1]
ADAPTER_DIR = ROOT / "models" / "distilbert-liar-lora"


class ModelInferenceError(RuntimeError):
	"""Raised when the base checkpoint or trained adapter cannot be loaded."""


@st.cache_resource(show_spinner=False)
def _load_model_and_tokenizer():
	"""Load the public base model and local adapter once per Streamlit process."""
	if not ADAPTER_DIR.is_dir():
		raise ModelInferenceError(f"LoRA adapter directory is missing: {ADAPTER_DIR}")

	required_files = (
		"adapter_config.json",
		"adapter_model.safetensors",
		"tokenizer.json",
		"tokenizer_config.json",
	)
	missing = [name for name in required_files if not (ADAPTER_DIR / name).is_file()]
	if missing:
		raise ModelInferenceError(
			f"LoRA adapter files are missing from {ADAPTER_DIR}: {', '.join(missing)}"
		)

	try:
		import torch
		from peft import PeftModel
		from transformers import AutoModelForSequenceClassification, AutoTokenizer

		tokenizer = AutoTokenizer.from_pretrained(ADAPTER_DIR, local_files_only=True)
		base_model = AutoModelForSequenceClassification.from_pretrained(
			MODEL_ID,
			num_labels=len(LABELS),
			id2label=ID2LABEL,
			label2id=LABEL2ID,
			use_safetensors=True,
		)
		model = PeftModel.from_pretrained(base_model, ADAPTER_DIR, is_trainable=False)
		device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
		model.to(device)
		model.eval()
	except Exception as error:
		raise ModelInferenceError(
			f"Could not load the TruthShield LoRA classifier. The local adapter is expected at "
			f"{ADAPTER_DIR}, and its public base checkpoint is {MODEL_ID!r}. The base checkpoint "
			f"is downloaded automatically from Hugging Face on first use; check internet access, "
			f"the installed torch/transformers/peft packages, and the adapter files. "
			f"Original error: {error}"
		) from error

	return model, tokenizer, device


def predict_claim(text: str) -> dict:
	"""Return the top LIAR label, its softmax score, and scores for all classes."""
	if not isinstance(text, str) or not text.strip():
		raise ValueError("Claim text must be a non-empty string.")

	model, tokenizer, device = _load_model_and_tokenizer()
	import torch

	inputs = tokenizer(
		text,
		max_length=MAX_LENGTH,
		truncation=True,
		return_tensors="pt",
	)
	inputs = {name: values.to(device) for name, values in inputs.items()}
	with torch.inference_mode():
		logits = model(**inputs).logits[0]
		probabilities = torch.softmax(logits, dim=-1).cpu().tolist()

	if len(probabilities) != len(LABELS):
		raise ModelInferenceError(
			f"The LoRA classifier returned {len(probabilities)} scores; expected {len(LABELS)} LIAR classes."
		)
	class_scores = dict(zip(LABELS, map(float, probabilities), strict=True))
	label = max(class_scores, key=class_scores.__getitem__)
	return {
		"label": label,
		"confidence": class_scores[label],
		"class_scores": class_scores,
	}