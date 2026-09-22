"""Tests for HTML, CSV, and PDF report exports."""

import unittest

from services.export import export_csv, export_html, export_pdf


ANALYSIS = {
	"url": "https://example.com/article",
	"title": "Example Article",
	"source": "example.com",
	"text": "The article body.",
	"text_processing": {"statistics": {"word_count": 3, "sentence_count": 1}},
	"claims": [{
		"claim": "The article body exists.",
		"status": "supported",
		"explanation": "Supporting evidence was found.",
		"evidence": [{
			"title": "Fact Check",
			"url": "https://fact.example/review",
			"source_type": "fact_check",
			"relation": "supports",
		}],
	}],
	"security": {"https": True, "hostname": "example.com", "threat_reputation": "unknown"},
	"scores": {"information_risk": 10, "cyber_risk": 5, "source_credibility": 70, "evidence_strength": 80},
	"score_reasons": {"information_risk": ["No contradictions."], "cyber_risk": ["HTTPS is enabled."]},
}


class ExportTests(unittest.TestCase):
	def test_html_contains_report_sections_and_values(self):
		report = export_html(ANALYSIS)

		self.assertIn("Example Article", report)
		self.assertIn("The article body exists.", report)
		self.assertIn("Fact Check", report)
		self.assertIn("Security Indicators", report)
		self.assertIn("information risk", report.lower())

	def test_csv_contains_headers_and_claim_evidence(self):
		report = export_csv(ANALYSIS)

		self.assertIn("url,article_title,source", report)
		self.assertIn("The article body exists.", report)
		self.assertIn("https://fact.example/review", report)
		self.assertIn("70", report)
		self.assertIn("security_indicators", report)
		self.assertIn("score_explanations", report)

	def test_pdf_is_valid_pdf_bytes(self):
		report = export_pdf(ANALYSIS)

		self.assertTrue(report.startswith(b"%PDF"))
		self.assertGreater(len(report), 500)


if __name__ == "__main__":
	unittest.main()