"""Mocked tests for Fact Check and GDELT evidence search."""

import unittest
from unittest.mock import Mock, patch

import requests

from services.evidence import build_evidence_query, search_evidence


class EvidenceSearchTests(unittest.TestCase):
	def test_builds_compact_query_from_claim_terms(self):
		query = build_evidence_query(
			"Share your thoughts in the comments. The ministry opened a centre in Mumbai in 2025."
		)

		self.assertNotIn("comments", query)
		self.assertIn("ministry", query)
		self.assertIn("Mumbai", query)
		self.assertIn("2025", query)

	def test_empty_claim_returns_empty_list(self):
		self.assertEqual(search_evidence(""), [])

	def test_gdelt_results_are_context_only(self):
		gdelt_response = Mock(status_code=200)
		gdelt_response.json.return_value = {
			"articles": [{"title": "News context", "url": "https://news.example/story"}]
		}
		with patch("services.evidence.requests.get", return_value=gdelt_response):
			results = search_evidence("A claim")

		self.assertEqual(results[0]["source_type"], "news")
		self.assertEqual(results[0]["relation"], "neutral")
		self.assertEqual(set(results[0]), {"title", "url", "source_type", "relation"})

	def test_multiple_fact_check_and_news_results(self):
		gdelt_response = Mock(status_code=200)
		gdelt_response.json.return_value = {
			"articles": [
				{"title": "News context", "url": "https://news.example/story"},
				{"title": "More context", "url": "https://news.example/another-story"},
			]
		}
		with patch("services.evidence.requests.get", return_value=gdelt_response):
			results = search_evidence("A claim")

		self.assertEqual(len(results), 2)
		self.assertEqual(results[0]["source_type"], "news")
		self.assertEqual(results[0]["relation"], "neutral")

	def test_google_api_key_is_ignored_and_gdelt_is_used(self):
		gdelt_response = Mock(status_code=200)
		gdelt_response.json.return_value = {"articles": []}
		with patch.dict("os.environ", {"GOOGLE_FACT_CHECK_API_KEY": "test-key"}):
			with patch("services.evidence.requests.get", return_value=gdelt_response) as request:
				self.assertEqual(search_evidence("A claim"), [])
				request.assert_called_once()
				self.assertEqual(request.call_args.args[0], "https://api.gdeltproject.org/api/v2/doc/doc")

	def test_gdelt_timeout_logs_and_returns_empty(self):
		with patch("services.evidence.requests.get", side_effect=requests.Timeout("timeout")):
			with self.assertLogs("services.evidence", level="WARNING"):
				self.assertEqual(search_evidence("A claim"), [])

	def test_duplicate_evidence_urls_are_removed(self):
		gdelt_response = Mock(status_code=200)
		gdelt_response.json.return_value = {
			"articles": [
				{"title": "Review", "url": "https://same.example/source"},
				{"title": "Review", "url": "https://same.example/source"},
			]
		}
		with patch("services.evidence.requests.get", return_value=gdelt_response):
			results = search_evidence("A claim")

		self.assertEqual(len(results), 1)


if __name__ == "__main__":
	unittest.main()