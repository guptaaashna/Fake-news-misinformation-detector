"""Tests for local LoRA model loading and claim predictions."""

from types import SimpleNamespace
import unittest
from unittest.mock import patch

import torch

from services.model_inference import (
	ADAPTER_DIR,
	LABELS,
	MODEL_ID,
	_load_model_and_tokenizer,
	predict_claim,
)


class ModelInferenceTests(unittest.TestCase):
	def test_loader_uses_public_base_and_local_adapter(self):
		base_model = object()
		tokenizer = object()
		model = SimpleNamespace(to=lambda device: None, eval=lambda: None)
		_load_model_and_tokenizer.clear()
		try:
			with patch("transformers.AutoTokenizer.from_pretrained", return_value=tokenizer) as tokenizer_load:
				with patch(
					"transformers.AutoModelForSequenceClassification.from_pretrained",
					return_value=base_model,
				) as base_load:
					with patch("peft.PeftModel.from_pretrained", return_value=model) as adapter_load:
						with patch("torch.cuda.is_available", return_value=False):
							loaded_model, loaded_tokenizer, device = _load_model_and_tokenizer()
		finally:
			_load_model_and_tokenizer.clear()

		self.assertIs(loaded_model, model)
		self.assertIs(loaded_tokenizer, tokenizer)
		self.assertEqual(str(device), "cpu")
		self.assertEqual(tokenizer_load.call_args.args[0], ADAPTER_DIR)
		self.assertTrue(tokenizer_load.call_args.kwargs["local_files_only"])
		self.assertEqual(base_load.call_args.args[0], MODEL_ID)
		self.assertEqual(base_load.call_args.kwargs["num_labels"], len(LABELS))
		self.assertIs(adapter_load.call_args.args[0], base_model)
		self.assertEqual(adapter_load.call_args.args[1], ADAPTER_DIR)

	def test_prediction_returns_top_label_and_all_six_scores(self):
		class Tokenizer:
			def __call__(self, text, **kwargs):
				self.text = text
				self.kwargs = kwargs
				return {"input_ids": torch.tensor([[1, 2]]), "attention_mask": torch.tensor([[1, 1]])}

		class Model:
			def __call__(self, **inputs):
				return SimpleNamespace(logits=torch.tensor([[0.0, 4.0, 1.0, 0.0, 0.0, 0.0]]))

		tokenizer = Tokenizer()
		with patch(
			"services.model_inference._load_model_and_tokenizer",
			return_value=(Model(), tokenizer, torch.device("cpu")),
		):
			result = predict_claim("A factual claim.")

		self.assertEqual(result["label"], "half-true")
		self.assertAlmostEqual(result["confidence"], result["class_scores"]["half-true"])
		self.assertEqual(tuple(result["class_scores"]), LABELS)
		self.assertEqual(len(result["class_scores"]), 6)
		self.assertAlmostEqual(sum(result["class_scores"].values()), 1.0, places=6)
		self.assertTrue(tokenizer.kwargs["truncation"])

	def test_empty_claim_is_rejected_before_model_loading(self):
		with patch("services.model_inference._load_model_and_tokenizer") as load_model:
			with self.assertRaisesRegex(ValueError, "non-empty"):
				predict_claim("  ")
		load_model.assert_not_called()


if __name__ == "__main__":
	unittest.main()