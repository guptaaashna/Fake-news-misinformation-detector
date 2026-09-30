"""External evidence search using Fact Check Tools and GDELT."""

import re
import logging

import requests


GDELT_URL = "https://api.gdeltproject.org/api/v2/doc/doc"
REQUEST_TIMEOUT = 8
MAX_RESULTS = 5

_last_search_status = "no_results"
LOGGER = logging.getLogger(__name__)
SEARCH_STOP_WORDS = {
	"a", "an", "and", "are", "as", "at", "by", "for", "from", "has", "in", "is", "it",
	"of", "on", "or", "said", "that", "the", "their", "this", "to", "was", "were", "will",
	"be", "comments", "comment", "guidelines", "respectful", "share", "thoughts", "views", "your",
}


def build_evidence_query(claim: str) -> str:
	"""Build a concise provider query from entities, dates, and factual terms."""
	if not isinstance(claim, str):
		return ""
	words = re.findall(r"[A-Za-z][A-Za-z'-]*|\b\d+(?:\.\d+)?%?\b", claim)
	selected = []
	for word in words:
		if word.lower().strip("'-") in SEARCH_STOP_WORDS:
			continue
		if word not in selected:
			selected.append(word)
	query = " ".join(selected[:12]).strip()
	return query or claim.strip()


def _gdelt_evidence(claim: str) -> list[dict]:
	query = build_evidence_query(claim)
	LOGGER.debug("Evidence query for GDELT: %s", query)
	try:
		response = requests.get(
			GDELT_URL,
			params={
				"query": query,
				"mode": "artlist",
				"format": "json",
				"maxrecords": MAX_RESULTS,
				"sort": "HybridRel",
			},
			timeout=REQUEST_TIMEOUT,
		)
		response.raise_for_status()
		payload = response.json()
	except (requests.RequestException, ValueError):
		LOGGER.warning("GDELT evidence search failed for claim %r", claim, exc_info=True)
		return []
	if not isinstance(payload, dict):
		return []

	evidence = []
	for article in payload.get("articles", []):
		if not isinstance(article, dict):
			continue
		url = article.get("url")
		if not url:
			continue
		evidence.append(
			{
				"title": article.get("title") or "News result",
				"url": url,
				"source_type": "news",
				"relation": "neutral",
			}
		)
		if len(evidence) >= MAX_RESULTS:
			break
	return evidence


def search_evidence(claim: str) -> list[dict]:
	"""Search GDELT for contextual news; results remain neutral evidence."""
	global _last_search_status
	_last_search_status = "no_results"
	if not isinstance(claim, str) or not claim.strip():
		return []
	LOGGER.debug("Searching evidence for claim: %s", claim.strip())

	evidence = _gdelt_evidence(claim)
	evidence = _deduplicate_evidence(evidence)
	LOGGER.debug("Evidence results for claim %r: %s", claim.strip(), evidence)
	if evidence:
		_last_search_status = "evidence_found"
	else:
		_last_search_status = "no_results"
	return evidence[: MAX_RESULTS * 2]


def _deduplicate_evidence(evidence: list[dict]) -> list[dict]:
	"""Keep the first occurrence of each evidence URL or title pair."""
	unique = []
	seen: set[tuple[str, str]] = set()
	for item in evidence:
		if not isinstance(item, dict):
			continue
		key = (str(item.get("url", "")).strip(), str(item.get("title", "")).strip())
		if not key[0] and not key[1]:
			continue
		if key in seen:
			continue
		seen.add(key)
		unique.append(item)
	return unique


def get_last_search_status() -> str:
	"""Return the status of the most recent search for UI messaging."""
	return _last_search_status


def verify_claim(claim: str, evidence: list[dict]) -> dict:
	"""Verify a claim using only explicit evidence relations.

	Neutral evidence cannot establish a conclusion, and mixed supporting and
	contradicting evidence is deliberately reported as insufficient.
	"""
	items = evidence if isinstance(evidence, list) else []
	supporting = [item for item in items if isinstance(item, dict) and item.get("relation") == "supports"]
	contradicting = [item for item in items if isinstance(item, dict) and item.get("relation") == "contradicts"]

	if supporting and not contradicting:
		status = "supported"
		explanation = (
			"Available evidence contains supporting fact-check/context results "
			"and no contradicting evidence was found."
		)
	elif contradicting and not supporting:
		status = "contradicted"
		explanation = (
			"Available evidence contains contradicting fact-check/context results "
			"and no supporting evidence was found."
		)
	elif not items:
		status = "insufficient"
		explanation = "No relevant external evidence was found, so the claim cannot be verified."
	else:
		status = "insufficient"
		explanation = (
			"The available evidence does not provide enough consistent information "
			"to establish whether this claim is supported or contradicted."
		)

	return {
		"claim": claim,
		"status": status,
		"explanation": explanation,
		"evidence": items,
	}
