"""Offline regression tests using a small WordPiece tokenizer fixture."""

import unittest

from tokenizers import Tokenizer
from tokenizers.models import WordPiece
from tokenizers.pre_tokenizers import Whitespace
from tokenizers.processors import TemplateProcessing
from transformers import PreTrainedTokenizerFast

from training.tokenize_dataset import encode_rows, length_summary, make_dataloader


class TokenizationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        backend = Tokenizer(WordPiece({"[PAD]": 0, "[UNK]": 1, "[CLS]": 2,
                                      "[SEP]": 3, "claim": 4, "not": 5}, unk_token="[UNK]"))
        backend.pre_tokenizer = Whitespace()
        backend.post_processor = TemplateProcessing(
            single="[CLS] $A [SEP]", special_tokens=[("[CLS]", 2), ("[SEP]", 3)])
        cls.tokenizer = PreTrainedTokenizerFast(tokenizer_object=backend, pad_token="[PAD]",
                                               unk_token="[UNK]", cls_token="[CLS]", sep_token="[SEP]")

    def row(self, statement, index=0):
        return {"id": str(index), "statement": statement, "label": 0, "label_name": "false"}

    def test_truncation_counts_special_tokens_and_preserves_separator(self):
        rows, stats = encode_rows([self.row("claim " * 200)], self.tokenizer)
        self.assertEqual(len(rows[0]["input_ids"]), 128)
        self.assertEqual(rows[0]["input_ids"][0], self.tokenizer.cls_token_id)
        self.assertEqual(rows[0]["input_ids"][-1], self.tokenizer.sep_token_id)
        self.assertEqual(stats["max"], 202)
        self.assertEqual(stats["tokens_removed"], 74)
        self.assertEqual(stats["truncated_rows"], 1)

    def test_dynamic_padding_masks_and_labels(self):
        rows, _ = encode_rows([self.row("claim"), self.row("not claim")], self.tokenizer)
        self.assertEqual([len(r["input_ids"]) for r in rows], [3, 4])
        batch = next(iter(make_dataloader(rows, self.tokenizer, batch_size=2)))
        self.assertEqual(set(batch), {"input_ids", "attention_mask", "labels"})
        self.assertEqual(tuple(batch["input_ids"].shape), (2, 4))
        self.assertEqual(batch["attention_mask"].tolist(), [[1, 1, 1, 0], [1, 1, 1, 1]])
        self.assertEqual(batch["input_ids"][0, -1].item(), self.tokenizer.pad_token_id)
        self.assertEqual(batch["labels"].tolist(), [0, 0])

    def test_training_shuffle_is_reproducible_and_final_batch_is_kept(self):
        rows, _ = encode_rows([self.row("claim " * n, n) for n in range(1, 12)], self.tokenizer)
        def order(training):
            return [n for batch in make_dataloader(rows, self.tokenizer, training=training, batch_size=4)
                    for n in batch["attention_mask"].sum(dim=1).tolist()]
        self.assertEqual(order(False), list(range(3, 14)))
        self.assertEqual(order(True), order(True))
        self.assertNotEqual(order(True), order(False))
        self.assertEqual(sorted(order(True)), list(range(3, 14)))

    def test_labels_are_validated(self):
        row = self.row("claim")
        row["label"] = 3
        with self.assertRaises(ValueError):
            encode_rows([row], self.tokenizer)

    def test_exact_limit_is_not_truncated(self):
        self.assertEqual(length_summary([127, 128, 129], 128)["truncated_rows"], 1)


if __name__ == "__main__":
    unittest.main()
