"""Small orchestration helper for the URL/text verification pipeline."""

from collections.abc import Callable
import logging

from services.evidence import search_evidence, verify_claim
from services.claims import is_factual_claim
from services.security import check_url
from services.scoring import calculate_scores


LOGGER = logging.getLogger(__name__)


def build_analysis_result(
	input_type: str,
	content: dict[str, str],
	claims: list[str],
	evidence_searcher: Callable[[str], list[dict]] = search_evidence,
	security_checker: Callable[[str], dict] = check_url,
	url: str = "",
) -> dict:
	"""Run evidence and verification for extracted URL/text content."""
	claim_results = []
	seen_claims: set[str] = set()
	for claim in claims:
		if (
			not isinstance(claim, str)
			or not claim.strip()
			or not is_factual_claim(claim)
			or claim.casefold() in seen_claims
		):
			continue
		seen_claims.add(claim.casefold())
		try:
			evidence = evidence_searcher(claim)
		except Exception:
			evidence = []
		claim_results.append(verify_claim(claim, evidence))
		LOGGER.debug(
			"Final classification claim=%r status=%s evidence_count=%d",
			claim,
			claim_results[-1]["status"],
			len(evidence),
		)

	try:
		security = security_checker(url) if input_type == "url" and url else {}
	except Exception:
		security = {}
	evidence_items = [
		item
		for claim_result in claim_results
		for item in claim_result.get("evidence", [])
	]
	scores = calculate_scores(claim_results, evidence_items, security)

	return {
		"input_type": input_type,
		"title": content.get("title", ""),
		"source": content.get("source", ""),
		"text": content.get("text", ""),
		"claims": claim_results,
		"security": security,
		"scores": {
			key: scores[key]
			for key in ("information_risk", "cyber_risk", "source_credibility", "evidence_strength")
		},
		"score_reasons": scores["reasons"],
	}