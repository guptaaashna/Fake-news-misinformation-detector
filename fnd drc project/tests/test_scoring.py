"""Deterministic tests for heuristic prototype scores."""

import math
import unittest

from services.scoring import calculate_scores

def item(relation: str, source_type: str = "fact_check", url: str = "https://example.com/source") -> dict:
	return {
		"title": "Evidence source",
		"url": url,
		"source_type": source_type,
		"relation": relation,
	}

def security(**overrides) -> dict:
	result = {
		"https": True,
		"suspicious_url": False,
		"hostname": "example.com",
		"dns_resolves": True,
		"ssl_available": True,
		"threat_reputation": "unknown",
	}
	result.update(overrides)
	return result

class ScoringTests(unittest.TestCase):
	def assert_valid_scores(self, result: dict) -> None:
		for key in ("information_risk", "cyber_risk", "source_credibility", "evidence_strength"):
			self.assertIsInstance(result[key], int)
			self.assertGreaterEqual(result[key], 0)
			self.assertLessEqual(result[key], 100)
			self.assertFalse(math.isnan(result[key]))
		self.assertEqual(
			set(result["reasons"]),
			{"information_risk", "cyber_risk", "source_credibility", "evidence_strength"},
		)
		self.assertTrue(all(result["reasons"][key] for key in result["reasons"]))

	def test_no_claims_no_evidence_normal_https(self):
		result = calculate_scores([], [], security())

		self.assert_valid_scores(result)
		self.assertEqual(result["information_risk"], 0)
		self.assertEqual(result["evidence_strength"], 0)

	def test_supported_claims_have_strong_available_evidence(self):
		claims = [{"text": "The agency opened an office.", "status": "supported", "evidence": [item("supports")]}]
		result = calculate_scores(claims, claims[0]["evidence"], security())

		self.assert_valid_scores(result)
		self.assertGreater(result["evidence_strength"], 0)
		self.assertLess(result["information_risk"], 30)

	def test_contradicted_claim_increases_information_risk(self):
		claims = [{"text": "The agency opened an office.", "status": "contradicted", "evidence": [item("contradicts")]}]
		result = calculate_scores(claims, claims[0]["evidence"], security())

		self.assert_valid_scores(result)
		self.assertGreater(result["information_risk"], 30)

	def test_insufficient_claim_is_not_treated_as_false(self):
		claims = [{"text": "The agency opened an office.", "status": "insufficient", "evidence": []}]
		result = calculate_scores(claims, [], security())

		self.assert_valid_scores(result)
		self.assertEqual(result["evidence_strength"], 0)

	def test_conflicting_evidence_reduces_consistency(self):
		claims = [{
			"text": "The agency opened an office.",
			"status": "insufficient",
			"evidence": [item("supports", url="https://example.com/a"), item("contradicts", url="https://example.com/b")],
		}]
		result = calculate_scores(claims, claims[0]["evidence"], security())

		self.assert_valid_scores(result)
		self.assertIn("conflicting", " ".join(result["reasons"]["evidence_strength"]).lower())

	def test_suspicious_url_increases_cyber_risk(self):
		result = calculate_scores([], [], security(suspicious_url=True))

		self.assert_valid_scores(result)
		self.assertGreater(result["cyber_risk"], 0)

	def test_http_url_is_a_security_concern(self):
		result = calculate_scores([], [], security(https=False, ssl_available=False))

		self.assert_valid_scores(result)
		self.assertGreater(result["cyber_risk"], 0)

	def test_unknown_reputation_does_not_create_maliciousness_score(self):
		result = calculate_scores([], [], security(threat_reputation="unknown"))

		self.assert_valid_scores(result)
		self.assertIn("unknown", " ".join(result["reasons"]["cyber_risk"]).lower())

	def test_multiple_supporting_and_contradicting_sources_are_counted(self):
		claims = [{
			"text": "The agency opened an office.",
			"status": "supported",
			"evidence": [
				item("supports", url="https://example.com/a"),
				item("supports", source_type="official", url="https://example.com/b"),
			],
		}]
		result = calculate_scores(claims, claims[0]["evidence"], security())

		self.assert_valid_scores(result)
		self.assertIn("2 evidence sources", " ".join(result["reasons"]["evidence_strength"]))


if __name__ == "__main__":
	unittest.main()
