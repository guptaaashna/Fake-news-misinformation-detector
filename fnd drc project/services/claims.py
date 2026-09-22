"""Rule-based candidate claim extraction using spaCy sentence segmentation."""

import re

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
	"announced",
	"became",
	"began",
	"confirmed",
	"cost",
	"created",
	"declared",
	"found",
	"generated",
	"held",
	"increased",
	"launched",
	"located",
	"measured",
	"opened",
	"produced",
	"reported",
	"required",
	"resulted",
	"said",
	"showed",
	"signed",
	"took",
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
	return any(term in lowered for term in NAVIGATION_TERMS | PROMOTIONAL_TERMS)


def _candidate_score(sentence) -> int:
	text = sentence.text.strip()
	lowered = text.lower()
	words = {token.lemma_.lower() for token in sentence if token.is_alpha}
	score = 0
	if sentence.ents:
		score += 2
	if re.search(r"\d|%|\b(?:monday|tuesday|wednesday|thursday|friday|saturday|sunday)\b", lowered):
		score += 2
	if words & FACTUAL_VERBS:
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
			if claim.endswith("?") or _looks_like_navigation_or_promotion(claim):
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
		return [claim for _, _, claim in candidates[:MAX_CLAIMS]]

	document = nlp(text[:200_000])
	candidates: list[tuple[int, int, str]] = []
	seen: set[str] = set()
	for position, sentence in enumerate(document.sents):
		claim = re.sub(r"\s+", " ", sentence.text).strip()
		normalized = claim.casefold()
		if len(claim) < MINIMUM_SENTENCE_LENGTH or normalized in seen:
			continue
		if claim.endswith("?") or _looks_like_navigation_or_promotion(claim):
			continue
		score = _candidate_score(sentence)
		if score < 2:
			continue
		seen.add(normalized)
		candidates.append((score, -position, claim))

	# Prefer stronger factual signals while retaining article order for ties.
	candidates.sort(reverse=True)
	return [claim for _, _, claim in candidates[:MAX_CLAIMS]]
