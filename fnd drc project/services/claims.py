"""Rule-based candidate claim extraction using spaCy sentence segmentation."""

import re
import logging

try:
	import spacy
except (ImportError, OSError):
	spacy = None


MAX_CLAIMS = 5
MINIMUM_SENTENCE_LENGTH = 35
MODEL_NAME = "en_core_web_sm"
MODEL_INSTALL_MESSAGE = (
	"The full spaCy English model is not installed. "
	"Install it with: python -m spacy download en_core_web_sm. "
	"Using basic sentence segmentation for now."
)

FACTUAL_VERBS = {
	"added",
	"announced",
	"approved",
	"became",
	"began",
	"confirmed",
	"cost",
	"created",
	"declared",
	"developed",
	"died",
	"discovered",
	"found",
	"generated",
	"held",
	"has",
	"have",
	"increased",
	"launched",
	"located",
	"measured",
	"operates",
	"opened",
	"produced",
	"reported",
	"required",
	"resulted",
	"said",
	"showed",
	"signed",
	"took",
	"uses",
	"was",
	"were",
	"weighs",
	"will",
}

PROMOTIONAL_TERMS = {
	"buy",
	"click",
	"discount",
	"exclusive offer",
	"limited time",
	"sale",
	"sign up",
	"sponsored",
	"subscribe now",
}
NAVIGATION_TERMS = {
	"about us",
	"contact us",
	"cookie policy",
	"log in",
	"menu",
	"privacy policy",
	"sign in",
}
LOGGER = logging.getLogger(__name__)
NON_FACTUAL_PHRASES = {
	"be respectful",
	"community guidelines",
	"comments section",
	"share your thoughts",
	"tell us what you think",
	"what do you think",
	"join the conversation",
	"leave a comment",
	"read more",
	"click here",
	"follow us",
	"subscribe",
}


def _load_language_pipeline():
	"""Load the full model, or a safe sentencizer-only fallback."""
	if spacy is None:
		return None, False
	try:
		return spacy.load(MODEL_NAME), True
	except OSError:
		nlp = spacy.blank("en")
		nlp.add_pipe("sentencizer")
		return nlp, False


def is_full_model_available() -> bool:
	"""Return whether the named English model is installed."""
	if spacy is None:
		return False
	try:
		spacy.load(MODEL_NAME)
		return True
	except OSError:
		return False


def _looks_like_navigation_or_promotion(sentence: str) -> bool:
	lowered = sentence.lower()
	return any(term in lowered for term in NAVIGATION_TERMS | PROMOTIONAL_TERMS | NON_FACTUAL_PHRASES)


def _has_factual_signal(text: str, words: set[str], entities: bool = False) -> bool:
	"""Return whether a sentence has a checkable event, state, or measurement."""
	lowered = text.lower()
	raw_words = set(re.findall(r"[a-z]+", lowered))
	return bool(
		(words | raw_words) & FACTUAL_VERBS
		or entities
		or re.search(
			r"\b(?:19|20)\d{2}\b|\b\d+(?:\.\d+)?%?|\b(?:january|february|march|april|may|june|"
			r"july|august|september|october|november|december)\b",
			lowered,
		)
	)


def is_factual_claim(text: str) -> bool:
	"""Guard evidence search from boilerplate and non-verifiable instructions."""
	if not isinstance(text, str):
		return False
	claim = re.sub(r"\s+", " ", text).strip()
	if len(claim) < MINIMUM_SENTENCE_LENGTH or "?" in claim:
		return False
	if _looks_like_navigation_or_promotion(claim):
		return False
	if re.search(r"\b(?:please|should you|would you|can you|click|subscribe|share)\b", claim.lower()):
		return False
	words = set(re.findall(r"[a-z]+", claim.lower()))
	return _has_factual_signal(claim, words)


def _candidate_score(sentence) -> int:
	text = sentence.text.strip()
	lowered = text.lower()
	words = {token.lemma_.lower() for token in sentence if token.is_alpha}
	score = 0
	if sentence.ents:
		score += 2
	if re.search(r"\d|%|\b(?:monday|tuesday|wednesday|thursday|friday|saturday|sunday)\b", lowered):
		score += 2
	if _has_factual_signal(text, words, bool(sentence.ents)):
		score += 2
	if len(text.split()) >= 8:
		score += 1
	return score


def extract_claims(text: str) -> list[str]:
	"""Return up to five factual-looking candidate sentences.

	This is an explainable heuristic, not a truth classifier. It does not
	determine whether any returned claim is true or false.
	"""
	if not isinstance(text, str) or not text.strip():
		return []

	nlp, _ = _load_language_pipeline()
	if nlp is None:
		sentences = [
			sentence.strip()
			for sentence in re.split(r"(?<=[.!?])\s+", text[:200_000])
			if sentence.strip()
		]
		candidates = []
		seen = set()
		for position, claim in enumerate(sentences):
			normalized = claim.casefold()
			if len(claim) < MINIMUM_SENTENCE_LENGTH or normalized in seen:
				continue
			if not is_factual_claim(claim):
				continue
			lowered = claim.lower()
			score = 0
			if re.search(r"\d|%|\b(?:monday|tuesday|wednesday|thursday|friday|saturday|sunday)\b", lowered):
				score += 2
			if any(re.search(rf"\b{re.escape(verb)}\b", lowered) for verb in FACTUAL_VERBS):
				score += 2
			if len(claim.split()) >= 8:
				score += 1
			if score >= 2:
				seen.add(normalized)
				candidates.append((score, -position, claim))
		candidates.sort(reverse=True)
		result = [claim for _, _, claim in candidates[:MAX_CLAIMS]]
		LOGGER.debug("Candidate claims: %s", result)
		return result

	document = nlp(text[:200_000])
	candidates: list[tuple[int, int, str]] = []
	seen: set[str] = set()
	for position, sentence in enumerate(document.sents):
		claim = re.sub(r"\s+", " ", sentence.text).strip()
		normalized = claim.casefold()
		if len(claim) < MINIMUM_SENTENCE_LENGTH or normalized in seen:
			continue
		if not is_factual_claim(claim):
			continue
		score = _candidate_score(sentence)
		if score < 2:
			continue
		seen.add(normalized)
		candidates.append((score, -position, claim))

	# Prefer stronger factual signals while retaining article order for ties.
	candidates.sort(reverse=True)
	result = [claim for _, _, claim in candidates[:MAX_CLAIMS]]
	LOGGER.debug("Candidate claims: %s", result)
	return result
