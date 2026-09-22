"""Tests for extracted article text processing."""

import unittest

from services.text_processing import process_text


class TextProcessingTests(unittest.TestCase):
    def test_normal_text_returns_expected_structure(self):
        result = process_text("The agency opened a new office. It created 400 jobs.")

        self.assertEqual(
            set(result),
            {
                "cleaned_text",
                "sentences",
                "tokens",
                "filtered_tokens",
                "lemmatized_tokens",
                "statistics",
                "word_frequency",
            },
        )
        self.assertEqual(result["statistics"]["sentence_count"], 2)
        self.assertGreater(result["statistics"]["word_count"], 0)

    def test_empty_text_is_safe(self):
        result = process_text("")

        self.assertEqual(result["cleaned_text"], "")
        self.assertEqual(result["sentences"], [])
        self.assertEqual(result["tokens"], [])
        self.assertEqual(result["word_frequency"], {})
        self.assertEqual(result["statistics"]["average_sentence_length"], 0.0)

    def test_whitespace_is_normalized(self):
        result = process_text("  First   sentence.\n\nSecond sentence. ")

        self.assertEqual(result["cleaned_text"], "First sentence. Second sentence.")
        self.assertEqual(len(result["sentences"]), 2)

    def test_sentence_splitting_and_tokenization(self):
        result = process_text("India launched a satellite. It supports weather monitoring!")

        self.assertEqual(len(result["sentences"]), 2)
        self.assertIn("India", result["tokens"])
        self.assertIn("satellite", result["tokens"])

    def test_stopwords_are_removed_from_filtered_tokens(self):
        result = process_text("The satellite is in orbit.")

        self.assertNotIn("the", result["filtered_tokens"])
        self.assertNotIn("is", result["filtered_tokens"])
        self.assertIn("satellite", result["filtered_tokens"])

    def test_lemmatization_and_frequency(self):
        result = process_text("The agencies opened offices. The offices support agencies.")

        self.assertIn("office", result["lemmatized_tokens"])
        self.assertIn("agency", result["lemmatized_tokens"])
        self.assertGreaterEqual(result["word_frequency"].get("office", 0), 1)

    def test_statistics_and_word_frequency(self):
        result = process_text("Red fish. Red fish swim.")
        statistics = result["statistics"]

        self.assertEqual(statistics["character_count"], len(result["cleaned_text"]))
        self.assertEqual(statistics["word_count"], 5)
        self.assertEqual(statistics["unique_word_count"], 3)
        self.assertEqual(statistics["sentence_count"], 2)
        self.assertAlmostEqual(statistics["average_sentence_length"], 2.5)
        self.assertGreaterEqual(result["word_frequency"].get("red", 0), 2)


if __name__ == "__main__":
    unittest.main()
