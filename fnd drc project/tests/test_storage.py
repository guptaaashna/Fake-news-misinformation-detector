"""Tests for SQLite analysis history."""

import tempfile
import unittest
from pathlib import Path

from services.storage import load_history, save_analysis


def result(url: str, title: str) -> dict:
	return {
		"input_type": "url",
		"url": url,
		"title": title,
		"source": "example.com",
		"text": "Article text.",
		"claims": [{"claim": "A claim", "status": "insufficient", "evidence": []}],
		"security": {"https": True},
		"scores": {"information_risk": 10, "cyber_risk": 5, "source_credibility": 20, "evidence_strength": 0},
		"text_processing": {"statistics": {"word_count": 2, "sentence_count": 1}},
	}


class StorageTests(unittest.TestCase):
	def setUp(self):
		self.temp_directory = tempfile.TemporaryDirectory()
		self.database_path = Path(self.temp_directory.name) / "history.db"

	def tearDown(self):
		self.temp_directory.cleanup()

	def test_database_creation_and_empty_history(self):
		self.assertEqual(load_history(self.database_path), [])
		self.assertTrue(self.database_path.exists())

	def test_saves_and_retrieves_analysis(self):
		analysis_id = save_analysis(result("https://example.com/one", "First"), self.database_path)
		history = load_history(self.database_path)

		self.assertEqual(len(history), 1)
		self.assertEqual(history[0]["history_id"], analysis_id)
		self.assertEqual(history[0]["url"], "https://example.com/one")
		self.assertEqual(history[0]["text_statistics"]["word_count"], 2)

	def test_multiple_analyses_are_returned_newest_first(self):
		save_analysis(result("https://example.com/one", "First"), self.database_path)
		save_analysis(result("https://example.com/two", "Second"), self.database_path)

		history = load_history(self.database_path)

		self.assertEqual(len(history), 2)
		self.assertEqual(history[0]["url"], "https://example.com/two")


if __name__ == "__main__":
	unittest.main()