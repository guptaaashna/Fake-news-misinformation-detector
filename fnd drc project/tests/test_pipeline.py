"""Mocked end-to-end tests for the URL/text verification pipeline."""

import unittest
from unittest.mock import Mock, patch

from services.article import ArticleExtractionError, clean_text, extract_article
from services.claims import extract_claims
from services.pipeline import build_analysis_result


def supporting_evidence(claim: str) -> list[dict]:
	return [
		{
			"title": f"Review of {claim}",
			"url": "https://fact.example/review",
			"source_type": "fact_check",
			"relation": "supports",
		}
	]


class PipelineIntegrationTests(unittest.TestCase):
	def test_pasted_text_runs_through_claims_evidence_and_verification(self):
		text = (
			"India launched a new satellite on Tuesday. "
			"The satellite weighs 2,000 kilograms and supports weather monitoring."
		)
		claims = extract_claims(clean_text(text))
		result = build_analysis_result(
			"text",
			{"title": "Pasted Text", "source": "User Input", "text": clean_text(text)},
			claims,
			supporting_evidence,
		)

		self.assertEqual(result["input_type"], "text")
		self.assertEqual(len(result["claims"]), 2)
		self.assertTrue(all(item["status"] == "supported" for item in result["claims"]))
		self.assertTrue(all(item["evidence"] for item in result["claims"]))

	def test_url_runs_through_article_claims_and_verification(self):
		response = Mock()
		response.text = (
			"<html><head><title>Science News</title></head><body><article>"
			"<p>India launched a new satellite on Tuesday.</p>"
			"<p>The satellite weighs 2,000 kilograms.</p>"
			"</article></body></html>"
		)
		with patch("services.article.requests.get", return_value=response):
			content = extract_article("https://news.example/science")
		claims = extract_claims(content["text"])
		result = build_analysis_result("url", content, claims, supporting_evidence)

		self.assertEqual(result["source"], "news.example")
		self.assertEqual(len(result["claims"]), 2)
		self.assertTrue(all("status" in item for item in result["claims"]))

	def test_url_result_contains_security_indicators(self):
		security = {
			"https": True,
			"hostname": "news.example",
			"dns_resolves": True,
			"ssl_available": True,
			"suspicious_url": False,
			"certificate": {},
			"threat_reputation": "unknown",
			"indicators": ["HTTPS is enabled."],
		}
		result = build_analysis_result(
			"url",
			{"title": "Article", "source": "news.example", "text": "Article text."},
			[],
			lambda claim: [],
			lambda url: security,
			"https://news.example/article",
		)

		self.assertEqual(result["security"], security)

	def test_invalid_url_is_a_graceful_article_error(self):
		with self.assertRaises(ArticleExtractionError):
			extract_article("not-a-valid-url")

	def test_empty_text_produces_no_claims_and_no_result_claims(self):
		claims = extract_claims("")
		result = build_analysis_result("text", {"title": "Pasted Text", "source": "User Input", "text": ""}, claims)

		self.assertEqual(claims, [])
		self.assertEqual(result["claims"], [])

	def test_no_evidence_returns_insufficient_without_crashing(self):
		claims = ["The agency opened a new office in Delhi."]
		result = build_analysis_result(
			"text",
			{"title": "Pasted Text", "source": "User Input", "text": claims[0]},
			claims,
			lambda claim: [],
		)

		self.assertEqual(result["claims"][0]["status"], "insufficient")

	def test_evidence_api_failure_still_returns_insufficient(self):
		def unavailable_search(claim: str) -> list[dict]:
			raise RuntimeError("external API unavailable")

		result = build_analysis_result(
			"text",
			{"title": "Pasted Text", "source": "User Input", "text": "A claim."},
			["The agency opened a new office in Delhi."],
			unavailable_search,
		)

		self.assertEqual(result["claims"][0]["status"], "insufficient")
		self.assertEqual(result["claims"][0]["evidence"], [])

	def test_duplicate_claims_are_searched_once(self):
		searched_claims = []

		def search_once(claim: str) -> list[dict]:
			searched_claims.append(claim)
			return []

		claim = "The agency opened a new office in Delhi."
		build_analysis_result(
			"text",
			{"title": "Pasted Text", "source": "User Input", "text": claim},
			[claim, claim.upper()],
			search_once,
		)

		self.assertEqual(len(searched_claims), 1)

	def test_non_factual_claims_never_reach_evidence_search(self):
		searched_claims = []

		def search(claim: str) -> list[dict]:
			searched_claims.append(claim)
			return []

		result = build_analysis_result(
			"url",
			{"title": "Article", "source": "timesofindia.indiatimes.com", "text": ""},
			[
				"Share your thoughts in the comments Be respectful - TOI community guidelines.",
				"The ministry opened a research centre in Mumbai on Monday.",
			],
			search,
		)

		self.assertEqual(searched_claims, ["The ministry opened a research centre in Mumbai on Monday."])
		self.assertEqual(result["claims"][0]["status"], "insufficient")


if __name__ == "__main__":
	unittest.main()