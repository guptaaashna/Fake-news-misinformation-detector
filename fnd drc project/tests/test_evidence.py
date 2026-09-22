"""Mocked tests for Fact Check and GDELT evidence search."""

import os
import unittest
from unittest.mock import Mock, patch

import requests

from services.evidence import search_evidence


class EvidenceSearchTests(unittest.TestCase):
	def test_empty_claim_returns_empty_list(self):
		self.assertEqual(search_evidence(""), [])

	def test_fact_check_result_structure_and_relation(self):
		fact_response = Mock(status_code=200)
		fact_response.json.return_value = {
			"claims": [
				{
					"text": "A claim",
					"claimReview": [
						{
							"title": "Claim review",
							"url": "https://fact.example/review",
							"textualRating": "False",
						}
					],
				}
			]
		}
		gdelt_response = Mock(status_code=200)
		gdelt_response.json.return_value = {"articles": []}
		with patch.dict(os.environ, {"GOOGLE_FACT_CHECK_API_KEY": "test-key"}):
			with patch("services.evidence.requests.get", side_effect=[fact_response, gdelt_response]):
				results = search_evidence("A claim")

		self.assertEqual(results[0]["source_type"], "fact_check")
		self.assertEqual(results[0]["relation"], "contradicts")
		self.assertEqual(set(results[0]), {"title", "url", "source_type", "relation"})

	def test_multiple_fact_check_and_news_results(self):
		fact_response = Mock(status_code=200)
		fact_response.json.return_value = {"claims": []}
		gdelt_response = Mock(status_code=200)
		gdelt_response.json.return_value = {
			"articles": [{"title": "News context", "url": "https://news.example/story"}]
		}
		with patch.dict(os.environ, {"GOOGLE_FACT_CHECK_API_KEY": "test-key"}):
			with patch("services.evidence.requests.get", side_effect=[fact_response, gdelt_response]):
				results = search_evidence("A claim")

		self.assertEqual(len(results), 1)
		self.assertEqual(results[0]["source_type"], "news")
		self.assertEqual(results[0]["relation"], "neutral")

	def test_fact_check_api_unavailable_falls_back_to_gdelt(self):
		gdelt_response = Mock(status_code=200)
		gdelt_response.json.return_value = {
			"articles": [{"title": "Context", "url": "https://news.example/context"}]
		}
		with patch.dict(os.environ, {"GOOGLE_FACT_CHECK_API_KEY": "test-key"}):
			with patch(
				"services.evidence.requests.get",
				side_effect=[requests.Timeout("timeout"), gdelt_response],
			):
				results = search_evidence("A claim")

		self.assertEqual(len(results), 1)
		self.assertEqual(results[0]["source_type"], "news")

	def test_no_fact_check_results_and_gdelt_failure_returns_empty(self):
		fact_response = Mock(status_code=200)
		fact_response.json.return_value = {"claims": []}
		with patch.dict(os.environ, {"GOOGLE_FACT_CHECK_API_KEY": "test-key"}):
			with patch(
				"services.evidence.requests.get",
				side_effect=[fact_response, requests.RequestException("offline")],
			):
				self.assertEqual(search_evidence("A claim"), [])

	def test_missing_api_key_still_tries_gdelt(self):
		gdelt_response = Mock(status_code=200)
		gdelt_response.json.return_value = {"articles": []}
		with patch.dict(os.environ, {}, clear=True):
			with patch("services.evidence.requests.get", return_value=gdelt_response) as request:
				self.assertEqual(search_evidence("A claim"), [])
				request.assert_called_once()

	def test_duplicate_evidence_urls_are_removed(self):
		fact_response = Mock(status_code=200)
		fact_response.json.return_value = {
			"claims": [
				{
					"claimReview": [
						{
							"title": "Review",
							"url": "https://same.example/source",
							"textualRating": "True",
						}
					]
				}
			]
		}
		gdelt_response = Mock(status_code=200)
		gdelt_response.json.return_value = {
			"articles": [{"title": "Review", "url": "https://same.example/source"}]
		}
		with patch.dict(os.environ, {"GOOGLE_FACT_CHECK_API_KEY": "test-key"}):
			with patch("services.evidence.requests.get", side_effect=[fact_response, gdelt_response]):
				results = search_evidence("A claim")

		self.assertEqual(len(results), 1)


if __name__ == "__main__":
	unittest.main()