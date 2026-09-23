"""Tests for URL and pasted-text extraction."""

import unittest
from unittest.mock import Mock, patch

import requests

from services.article import ArticleExtractionError, clean_text, extract_article


class ArticleExtractionTests(unittest.TestCase):
	def test_extracts_title_source_and_paragraphs(self):
		response = Mock()
		response.text = """
			<html><head><title>Example News</title><script>ignore()</script></head>
			<body><nav>Menu</nav><article><h1>Fallback heading</h1>
			<p>First meaningful paragraph.</p><p>Second paragraph.</p>
			</article><footer>Footer</footer></body></html>
		"""
		with patch("services.article.requests.get", return_value=response):
			result = extract_article("https://news.example/story")

		self.assertEqual(result["title"], "Example News")
		self.assertEqual(result["source"], "news.example")
		self.assertIn("First meaningful paragraph.", result["text"])
		self.assertNotIn("Menu", result["text"])

	def test_filters_sidebar_author_related_and_newsletter_content(self):
		response = Mock()
		response.text = """
			<html><head><title>Example News</title></head><body>
			<article class="article-body">
				<p>The main article reports that the agency opened a new office in Delhi.</p>
				<p>The office will employ 400 people during its first year.</p>
			</article>
			<aside class="author-bio"><p>The author has written about technology for ten years.</p></aside>
			<section class="related-articles"><p>Read more related articles about this topic.</p></section>
			<div class="newsletter"><p>Subscribe to our newsletter for daily updates.</p></div>
			</body></html>
		"""
		with patch("services.article.requests.get", return_value=response):
			result = extract_article("https://news.example/story")

		self.assertIn("agency opened a new office", result["text"])
		self.assertNotIn("author has written", result["text"])
		self.assertNotIn("Subscribe", result["text"])
		self.assertNotIn("Read more", result["text"])

	def test_filters_comment_guidelines_and_extracts_author(self):
		response = Mock()
		response.text = """
			<html><head><meta name="author" content="Asha Rao"></head><body>
			<article>
				<p>The ministry opened a research centre in Mumbai on Monday.</p>
				<p>Share your thoughts in the comments Be respectful - TOI community guidelines.</p>
			</article>
			</body></html>
		"""
		with patch("services.article.requests.get", return_value=response):
			result = extract_article("https://timesofindia.indiatimes.com/story")

		self.assertEqual(result["author"], "Asha Rao")
		self.assertIn("ministry opened", result["text"])
		self.assertNotIn("Share your thoughts", result["text"])
		self.assertNotIn("community guidelines", result["text"])

	def test_adds_sentence_boundaries_to_json_ld_body(self):
		response = Mock()
		response.text = """
			<html><head><script type="application/ld+json">
			{"@type":"NewsArticle","articleBody":"The agency opened an office in Delhi.The office employs 400 people."}
			</script></head><body><div>Widget content only.</div></body></html>
		"""
		with patch("services.article.requests.get", return_value=response):
			result = extract_article("https://news.example/story")

		self.assertIn("Delhi. The office", result["text"])

	def test_prefers_article_body_over_page_main_sidebar(self):
		response = Mock()
		response.text = """
			<html><body><main>
				<article><p>The actual article body contains the reported event and its date.</p></article>
				<aside class="sidebar"><p>This sidebar contains unrelated recommendations and metadata.</p></aside>
			</main></body></html>
		"""
		with patch("services.article.requests.get", return_value=response):
			result = extract_article("https://news.example/story")

		self.assertIn("actual article body", result["text"])
		self.assertNotIn("sidebar contains", result["text"])

	def test_rejects_invalid_url(self):
		with self.assertRaises(ArticleExtractionError):
			extract_article("not-a-url")

	def test_handles_unreachable_url(self):
		with patch("services.article.requests.get", side_effect=requests.Timeout("offline")):
			with self.assertRaises(ArticleExtractionError):
				extract_article("https://news.example/story")

	def test_handles_http_failure(self):
		response = Mock()
		response.raise_for_status.side_effect = requests.HTTPError("server error")
		with patch("services.article.requests.get", return_value=response):
			with self.assertRaises(ArticleExtractionError):
				extract_article("https://news.example/story")

	def test_cleans_pasted_text(self):
		self.assertEqual(clean_text("  One   sentence.\n\nTwo.  "), "One sentence. Two.")

	def test_preserves_very_short_text(self):
		self.assertEqual(clean_text("Brief note."), "Brief note.")

	def test_handles_long_article_text(self):
		long_paragraph = "Long article sentence. " * 500
		response = Mock(text=f"<html><body><article><p>{long_paragraph}</p></article></body></html>")
		with patch("services.article.requests.get", return_value=response):
			result = extract_article("https://news.example/long-story")

		self.assertGreater(len(result["text"]), 1000)

	def test_empty_page_fails(self):
		response = Mock(text="<html><body><p> </p></body></html>")
		with patch("services.article.requests.get", return_value=response):
			with self.assertRaises(ArticleExtractionError):
				extract_article("https://news.example/empty")


if __name__ == "__main__":
	unittest.main()