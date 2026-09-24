"""Regression checks for LIAR parsing and split leakage prevention."""

import unittest

from training.prepare_dataset import claim_key, deduplicate, parse_split


def tsv_row(statement, label="false", statement_id="1.json"):
    return "\t".join([statement_id, label, statement] + [""] * 11) + "\n"


class DatasetPreparationTests(unittest.TestCase):
    def test_literal_quotes_do_not_merge_rows_or_swallow_metadata(self):
        text = tsv_row('"A quoted claim', statement_id="1.json")
        text += tsv_row('Another claim"', statement_id="2.json")
        rows = parse_split(text.encode(), "train")
        self.assertEqual(len(rows), 2)
        self.assertEqual(rows[0]["statement"], '"A quoted claim')
        self.assertEqual(rows[1]["statement"], 'Another claim"')

    def test_reject_invalid_rows(self):
        for text in ("id\tfalse\tclaim\n", tsv_row(" "), tsv_row("claim", "fake"),
                     tsv_row("a") + tsv_row("b")):
            with self.subTest(text=text), self.assertRaises(ValueError):
                parse_split(text.encode(), "train")

    def test_held_out_duplicate_wins_without_label_based_decisions(self):
        splits = {
            "train": parse_split(tsv_row("Ａ  claim", "true", "1.json").encode(), "train"),
            "validation": parse_split(tsv_row("a claim", "false", "2.json").encode(), "validation"),
            "test": parse_split(tsv_row("A claim", "half-true", "3.json").encode(), "test"),
        }
        kept, groups, excluded = deduplicate(splits)
        self.assertEqual([len(kept[s]) for s in ("train", "validation", "test")], [0, 0, 1])
        self.assertTrue(groups[0]["cross_split"])
        self.assertTrue(groups[0]["conflicting_labels"])
        self.assertEqual(len(excluded), 2)
        self.assertEqual(kept["test"][0]["statement"], "A claim")
        self.assertEqual(kept["test"][0]["label_name"], "half-true")

    def test_within_split_first_row_wins_and_negation_is_preserved(self):
        rows = parse_split((tsv_row("It is true", statement_id="1.json")
                            + tsv_row("it is true", statement_id="2.json")
                            + tsv_row("It is not true", statement_id="3.json")).encode(), "train")
        kept, _, _ = deduplicate({"train": rows, "validation": [], "test": []})
        self.assertEqual([r["id"] for r in kept["train"]], ["1.json", "3.json"])
        self.assertNotEqual(claim_key("It is true"), claim_key("It is not true"))


if __name__ == "__main__":
    unittest.main()
