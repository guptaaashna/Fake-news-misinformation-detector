"""Transparent heuristic prototype indicators for URL analyses."""

import re


def clamp_score(value: float) -> int:
	return round(max(0, min(100, value)))


def _evidence_items(claims: list[dict], evidence: list[dict] | None) -> list[dict]:
	if isinstance(evidence, list) and evidence:
		return [item for item in evidence if isinstance(item, dict)]
	items = []
	for claim in claims if isinstance(claims, list) else []:
		if isinstance(claim, dict):
			items.extend(item for item in claim.get("evidence", []) if isinstance(item, dict))
	return items


def _suspicious_language_score(claims: list[dict]) -> tuple[int, list[str]]:
	texts = [claim.get("text", "") for claim in claims if isinstance(claim, dict)]
	joined = " ".join(texts)
	terms = ("shocking", "unbelievable", "secret", "miracle", "urgent", "exposed", "breaking")
	term_hits = sum(joined.lower().count(term) for term in terms)
	punctuation_hits = len(re.findall(r"[!?]{2,}", joined))
	uppercase_hits = sum(
		1 for text in texts if len(re.findall(r"[A-Z]", text)) >= 4 and len(re.findall(r"[a-z]", text)) <= 2
	)
	score = clamp_score(term_hits * 15 + punctuation_hits * 20 + uppercase_hits * 20)
	reasons = []
	if term_hits or punctuation_hits or uppercase_hits:
		reasons.append("Some candidate claims contain modest sensational-language signals.")
	else:
		reasons.append("No strong sensational-language indicators detected.")
	return score, reasons


def _cyber_score(security: dict) -> tuple[int, list[str]]:
	security = security if isinstance(security, dict) else {}
	url_pattern_score = 75 if security.get("suspicious_url") else 0
	if security.get("ssl_available"):
		ssl_score = 0
		ssl_reason = "SSL certificate information was retrieved."
	elif security.get("https"):
		ssl_score = 65
		ssl_reason = "HTTPS is enabled, but SSL certificate information was unavailable."
	else:
		ssl_score = 45
		ssl_reason = "URL does not use HTTPS, which is a security concern but not proof of maliciousness."
	domain_signal_score = 60 if not security.get("dns_resolves", False) else 0
	reputation_score = 0
	reasons = [ssl_reason]
	if security.get("suspicious_url"):
		reasons.append("A potentially suspicious URL pattern was detected.")
	else:
		reasons.append("No suspicious URL pattern was detected.")
	if domain_signal_score:
		reasons.append("Hostname resolution failed, so the domain signal is concerning.")
	else:
		reasons.append("Hostname resolved successfully.")
	reasons.append("Threat reputation is unknown and was not treated as malicious.")
	return clamp_score(
		0.35 * url_pattern_score
		+ 0.25 * ssl_score
		+ 0.20 * domain_signal_score
		+ 0.20 * reputation_score
	), reasons


def calculate_scores(claims, evidence, security) -> dict:
	"""Calculate bounded, explainable prototype indicators from analysis data."""
	claim_list = [claim for claim in claims if isinstance(claim, dict)] if isinstance(claims, list) else []
	evidence_items = _evidence_items(claim_list, evidence)
	total_claims = len(claim_list)
	contradicted = sum(claim.get("status") == "contradicted" for claim in claim_list)
	insufficient = sum(claim.get("status") == "insufficient" for claim in claim_list)
	neutral_only = sum(
		bool(claim.get("evidence"))
		and not any(item.get("relation") in {"supports", "contradicts"} for item in claim.get("evidence", []))
		for claim in claim_list
	)
	contradiction_score = 100 * contradicted / total_claims if total_claims else 0
	weak_claims = insufficient + neutral_only
	weak_evidence_score = 100 * weak_claims / total_claims if total_claims else 0
	language_score, language_reasons = _suspicious_language_score(claim_list)
	information_risk = clamp_score(
		0.45 * contradiction_score + 0.30 * weak_evidence_score + 0.25 * language_score
	)
	information_reasons = [
		f"{contradicted} of {total_claims} claims were marked contradicted." if total_claims else "No claims were available for contradiction analysis.",
		f"{weak_claims} claims had insufficient or neutral-only evidence." if total_claims else "No claims were available for evidence-weakness analysis.",
		*language_reasons,
	]

	cyber_risk, cyber_reasons = _cyber_score(security)
	relations = [item.get("relation") for item in evidence_items]
	supporting = relations.count("supports")
	contradicting = relations.count("contradicts")
	neutral = relations.count("neutral")
	unique_sources = len({item.get("url") or item.get("title") for item in evidence_items})
	conflicting = supporting > 0 and contradicting > 0
	if not evidence_items:
		source_credibility = 20
		source_reasons = ["No evidence sources were available, so source context is limited."]
	else:
		source_credibility = 35
		source_reasons = []
		if any(item.get("source_type") == "fact_check" for item in evidence_items):
			source_credibility += 25
			source_reasons.append("Evidence includes fact-check sources.")
		if any(item.get("source_type") == "official" for item in evidence_items):
			source_credibility += 25
			source_reasons.append("Evidence includes official-source context.")
		if unique_sources > 1:
			source_credibility += 15
			source_reasons.append("Multiple evidence sources were available.")
		if conflicting:
			source_credibility -= 20
			source_reasons.append("Some evidence was conflicting.")
		source_credibility = clamp_score(source_credibility)
	if not source_reasons:
		source_reasons.append("Available source context was limited.")

	if not evidence_items:
		evidence_strength = 0
		evidence_reasons = ["No relevant evidence items were found."]
	else:
		evidence_strength = min(60, unique_sources * 20)
		evidence_reasons = [f"{len(evidence_items)} relevant evidence items were found."]
		if supporting:
			evidence_strength += min(25, supporting * 10)
			evidence_reasons.append(f"{supporting} evidence sources support claims.")
		if contradicting:
			evidence_strength += min(15, contradicting * 5)
			evidence_reasons.append(f"{contradicting} evidence sources provide conflicting information.")
		if neutral and not supporting and not contradicting:
			evidence_strength = min(evidence_strength, 40)
			evidence_reasons.append("Only neutral/contextual sources were available, so strength is limited.")
		if conflicting:
			evidence_strength = max(0, evidence_strength - 20)
			evidence_reasons.append("Conflicting evidence reduced the available evidence strength.")

	return {
		"information_risk": information_risk,
		"cyber_risk": cyber_risk,
		"source_credibility": clamp_score(source_credibility),
		"evidence_strength": clamp_score(evidence_strength),
		"reasons": {
			"information_risk": information_reasons,
			"cyber_risk": cyber_reasons,
			"source_credibility": source_reasons,
			"evidence_strength": evidence_reasons,
		},
	}
