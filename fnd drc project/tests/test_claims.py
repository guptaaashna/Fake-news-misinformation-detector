"""Tests for rule-based candidate claim extraction."""

import unittest

from services.claims import extract_claims


class ClaimExtractionTests(unittest.TestCase):
	def test_extracts_multiple_factual_sentences(self):
		text = (
			"India launched a new satellite on Tuesday. "
			"The satellite weighs 2,000 kilograms and will support weather monitoring. "
			"Scientists said the mission cost approximately 500 crore rupees. "
			"The launch took place from Sriharikota."
		)
		claims = extract_claims(text)

		self.assertIsInstance(claims, list)
		self.assertGreaterEqual(len(claims), 3)
		self.assertLessEqual(len(claims), 5)
		self.assertTrue(all(isinstance(claim, str) and claim for claim in claims))

	def test_handles_numbers_and_dates(self):
		claims = extract_claims("The project opened on 12 March 2025 and created 400 jobs.")

		self.assertEqual(len(claims), 1)
		self.assertIn("12 March 2025", claims[0])

	def test_excludes_questions(self):
		claims = extract_claims(
			"Did the company launch the product? The company launched the product in June 2025."
		)

		self.assertEqual(len(claims), 1)
		self.assertNotIn("Did the company", claims[0])

	def test_handles_short_and_empty_text(self):
		self.assertEqual(extract_claims(""), [])
		self.assertEqual(extract_claims("Too short."), [])

	def test_removes_duplicate_claims(self):
		text = "The agency opened a new office in Delhi. The agency opened a new office in Delhi."

		self.assertEqual(extract_claims(text), ["The agency opened a new office in Delhi."])

	def test_excludes_text_without_obvious_claim_signals(self):
		claims = extract_claims("Welcome to our website. Read more here. Thanks for visiting.")

		self.assertEqual(claims, [])

	def test_limits_normal_article_to_five_claims(self):
		text = " ".join(
			f"The research team reported finding {number} results in 2025."
			for number in range(1, 10)
		)

		self.assertLessEqual(len(extract_claims(text)), 5)


if __name__ == "__main__":
	unittest.main()
