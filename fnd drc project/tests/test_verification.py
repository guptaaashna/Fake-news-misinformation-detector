"""Unit tests for deterministic claim verification."""

import unittest

from services.evidence import verify_claim


def evidence(relation: str, title: str = "Evidence") -> dict:
	return {
		"title": title,
		"url": "https://example.com/evidence",
		"source_type": "fact_check",
		"relation": relation,
	}


class ClaimVerificationTests(unittest.TestCase):
	def assert_result_shape(self, result: dict, claim: str) -> None:
		self.assertEqual(result["claim"], claim)
		self.assertIn(result["status"], {"supported", "contradicted", "insufficient"})
		self.assertTrue(result["explanation"])
		self.assertIsInstance(result["evidence"], list)

	def test_supporting_evidence_only_is_supported(self):
		result = verify_claim("The event happened.", [evidence("supports")])

		self.assertEqual(result["status"], "supported")
		self.assert_result_shape(result, "The event happened.")

	def test_contradicting_evidence_only_is_contradicted(self):
		result = verify_claim("The event happened.", [evidence("contradicts")])

		self.assertEqual(result["status"], "contradicted")

	def test_mixed_evidence_is_insufficient(self):
		result = verify_claim("The event happened.", [evidence("supports"), evidence("contradicts")])

		self.assertEqual(result["status"], "insufficient")

	def test_neutral_evidence_is_insufficient(self):
		result = verify_claim("The event happened.", [evidence("neutral")])

		self.assertEqual(result["status"], "insufficient")

	def test_no_evidence_is_insufficient_with_specific_explanation(self):
		result = verify_claim("The event happened.", [])

		self.assertEqual(result["status"], "insufficient")
		self.assertIn("No relevant external evidence", result["explanation"])

	def test_multiple_supporting_sources_are_supported(self):
		result = verify_claim("The event happened.", [evidence("supports"), evidence("supports")])

		self.assertEqual(result["status"], "supported")

	def test_multiple_contradicting_sources_are_contradicted(self):
		result = verify_claim("The event happened.", [evidence("contradicts"), evidence("contradicts")])

		self.assertEqual(result["status"], "contradicted")


if __name__ == "__main__":
	unittest.main()