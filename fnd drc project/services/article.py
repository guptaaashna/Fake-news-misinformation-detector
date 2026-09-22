"""Simple webpage and pasted-text extraction helpers."""

import re
from urllib.parse import urlparse

import requests
from bs4 import BeautifulSoup


REQUEST_TIMEOUT = 10
USER_AGENT = "TruthShieldAI/0.1 (educational content extraction)"
BOILERPLATE_MARKERS = (
	"author",
	"advert",
	"breadcrumb",
	"comment",
	"cookie",
	"footer",
	"newsletter",
	"promo",
	"related",
	"read-more",
	"recommend",
	"sidebar",
	"social",
	"subscribe",
	"trending",
	"widget",
)
BOILERPLATE_PHRASES = (
	"read more",
	"related articles",
	"subscribe to",
	"newsletter",
	"advertisement",
	"about the author",
)


class ArticleExtractionError(Exception):
	"""Raised when a webpage cannot be safely converted into article text."""


def clean_text(text: str) -> str:
	"""Collapse excess whitespace while preserving meaningful text."""
	return re.sub(r"\s+", " ", text or "").strip()


def _validate_url(url: str) -> str:
	parsed_url = urlparse((url or "").strip())
	if parsed_url.scheme not in {"http", "https"} or not parsed_url.netloc:
		raise ArticleExtractionError("Please enter a valid HTTP or HTTPS URL.")
	return parsed_url.geturl()


def _get_title(soup: BeautifulSoup) -> str:
	title = clean_text(soup.title.get_text(" ", strip=True)) if soup.title else ""
	if title:
		return title
	h1 = soup.find("h1")
	return clean_text(h1.get_text(" ", strip=True)) if h1 else "Untitled article"


def _has_boilerplate_marker(element) -> bool:
	attributes = " ".join(
		[
			str(element.get("id", "")),
			" ".join(element.get("class", [])),
		]
	).lower()
	return any(marker in attributes for marker in BOILERPLATE_MARKERS)


def _is_useful_paragraph(text: str) -> bool:
	cleaned = clean_text(text)
	if len(cleaned) < 20 or len(cleaned.split()) < 3:
		return False
	lowered = cleaned.lower()
	return not any(phrase in lowered for phrase in BOILERPLATE_PHRASES)


def _select_content_root(soup: BeautifulSoup):
	"""Choose the most likely article container without site-specific selectors."""
	candidates = []
	for selector in (
		'article',
		'[itemprop="articleBody"]',
		'main',
		'[role="main"]',
	):
		candidates.extend(soup.select(selector))
	if not candidates:
		return soup

	def content_score(element) -> tuple[int, int]:
		paragraphs = [
			clean_text(paragraph.get_text(" ", strip=True))
			for paragraph in element.find_all("p")
			if not _has_boilerplate_marker(paragraph)
			and _is_useful_paragraph(paragraph.get_text(" ", strip=True))
		]
		return sum(len(paragraph) for paragraph in paragraphs), len(paragraphs)

	return max(candidates, key=content_score)


def _get_article_text(soup: BeautifulSoup) -> str:
	for element in soup(["script", "style", "noscript", "template", "svg"]):
		element.decompose()
	for element in soup(["nav", "footer", "header", "aside", "form"]):
		element.decompose()
	for element in soup.find_all(_has_boilerplate_marker):
		element.decompose()

	content_root = _select_content_root(soup)
	paragraphs = [
		clean_text(paragraph.get_text(" ", strip=True))
		for paragraph in content_root.find_all("p")
		if _is_useful_paragraph(paragraph.get_text(" ", strip=True))
	]
	return "\n\n".join(paragraphs)


def extract_article(url: str) -> dict[str, str]:
	"""Fetch one webpage and return its title, source domain, and readable text."""
	validated_url = _validate_url(url)
	try:
		response = requests.get(
			validated_url,
			headers={"User-Agent": USER_AGENT},
			timeout=REQUEST_TIMEOUT,
		)
		response.raise_for_status()
	except requests.RequestException as error:
		raise ArticleExtractionError(
			"Unable to extract this article automatically. Please paste the article text instead."
		) from error

	soup = BeautifulSoup(response.text, "html.parser")
	article_text = _get_article_text(soup)
	if not article_text:
		raise ArticleExtractionError(
			"Unable to extract this article automatically. Please paste the article text instead."
		)

	parsed_url = urlparse(validated_url)
	return {
		"title": _get_title(soup),
		"source": parsed_url.hostname or parsed_url.netloc,
		"text": article_text,
	}
