"""Simple webpage and pasted-text extraction helpers."""

import re
import logging
import json
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
	"be respectful",
	"community guidelines",
	"share your thoughts",
	"share your views",
	"leave a comment",
	"join the conversation",
	"read more",
	"related articles",
	"subscribe to",
	"subscribe now",
	"newsletter",
	"advertisement",
	"advertising",
	"about the author",
	"cookie policy",
	"privacy policy",
)
LOGGER = logging.getLogger(__name__)


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
	meta_title = soup.find("meta", attrs={"property": "og:title"}) or soup.find(
		"meta", attrs={"name": "twitter:title"}
	)
	if meta_title and meta_title.get("content"):
		return clean_text(meta_title["content"])
	title = clean_text(soup.title.get_text(" ", strip=True)) if soup.title else ""
	if title:
		return title
	h1 = soup.find("h1")
	return clean_text(h1.get_text(" ", strip=True)) if h1 else "Untitled article"


def _get_author(soup: BeautifulSoup) -> str:
	for selector in (
		'meta[name="author"]',
		'meta[property="article:author"]',
		'[itemprop="author"]',
		'[rel="author"]',
	):
		element = soup.select_one(selector)
		if element:
			value = element.get("content") or element.get_text(" ", strip=True)
			if value:
				return clean_text(value)
	return ""


def _get_structured_article_body(soup: BeautifulSoup) -> str:
	"""Read articleBody from JSON-LD when a publisher omits body paragraphs."""
	for script in soup.find_all("script", type="application/ld+json"):
		try:
			payload = json.loads(script.string or script.get_text())
		except (TypeError, json.JSONDecodeError):
			continue
		items = payload if isinstance(payload, list) else [payload]
		for item in items:
			if not isinstance(item, dict):
				continue
			body = item.get("articleBody")
			if isinstance(body, str) and _is_useful_paragraph(body):
				return clean_text(re.sub(r"([.!?])(?=[A-Z])", r"\1 ", body))
	return ""


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
	structured_body = _get_structured_article_body(soup)
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
	return "\n\n".join(paragraphs) or structured_body


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

	response_content = response.content if isinstance(response.content, (bytes, bytearray)) else response.text
	soup = BeautifulSoup(response_content, "html.parser")
	article_text = _get_article_text(soup)
	if not article_text:
		raise ArticleExtractionError(
			"Unable to extract this article automatically. Please paste the article text instead."
		)

	parsed_url = urlparse(validated_url)
	result = {
		"title": _get_title(soup),
		"source": parsed_url.hostname or parsed_url.netloc,
		"author": _get_author(soup),
		"text": article_text,
	}
	LOGGER.debug(
		"Extracted article title=%r source=%s author=%r body_length=%d",
		result["title"],
		result["source"],
		result["author"],
		len(article_text),
	)
	return result
