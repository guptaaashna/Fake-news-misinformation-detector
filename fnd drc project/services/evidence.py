"""External evidence search using Fact Check Tools and GDELT."""

import os

import requests
from dotenv import load_dotenv


FACT_CHECK_URL = "https://factchecktools.googleapis.com/v1alpha1/claims:search"
GDELT_URL = "https://api.gdeltproject.org/api/v2/doc/doc"
REQUEST_TIMEOUT = 8
MAX_RESULTS = 5

_last_search_status = "no_results"


def _relation_from_rating(rating: str) -> str:
	"""Map common fact-check ratings without pretending to validate a claim."""
	lowered = rating.lower()
	if any(word in lowered for word in ("true", "correct", "accurate", "supported")):
		return "supports"
	if any(word in lowered for word in ("false", "fake", "incorrect", "misleading", "unsupported")):
		return "contradicts"
	return "neutral"


def _fact_check_evidence(claim: str, api_key: str) -> list[dict]:
	try:
		response = requests.get(
			FACT_CHECK_URL,
			params={"query": claim, "key": api_key, "pageSize": MAX_RESULTS},
			timeout=REQUEST_TIMEOUT,
		)
		if response.status_code == 429:
			return []
		response.raise_for_status()
		payload = response.json()
	except (requests.RequestException, ValueError):
		return []
	if not isinstance(payload, dict):
		return []

	evidence = []
	for result in payload.get("claims", []):
		if not isinstance(result, dict):
			continue
		for review in result.get("claimReview", []):
			if not isinstance(review, dict):
				continue
			url = review.get("url")
			if not url:
				continue
			rating = review.get("textualRating", "")
			evidence.append(
				{
					"title": review.get("title") or result.get("text") or "Fact-check review",
					"url": url,
					"source_type": "fact_check",
					"relation": _relation_from_rating(rating),
				}
			)
			if len(evidence) >= MAX_RESULTS:
				return evidence
	return evidence


def _gdelt_evidence(claim: str) -> list[dict]:
	try:
		response = requests.get(
			GDELT_URL,
			params={
				"query": claim,
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
	"""Search Fact Check first, then add contextual GDELT news results."""
	global _last_search_status
	_last_search_status = "no_results"
	if not isinstance(claim, str) or not claim.strip():
		return []

	load_dotenv()
	api_key = os.getenv("GOOGLE_FACT_CHECK_API_KEY", "").strip()
	evidence = _fact_check_evidence(claim, api_key) if api_key else []
	evidence.extend(_gdelt_evidence(claim))
	evidence = _deduplicate_evidence(evidence)
	if evidence:
		_last_search_status = "evidence_found"
	else:
		_last_search_status = "unavailable" if not api_key else "no_results"
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
