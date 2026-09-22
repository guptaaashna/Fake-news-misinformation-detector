"""Export completed TruthShield analyses to HTML, CSV, and PDF."""

import csv
import html
import io
import json
import textwrap

import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages


SCORE_KEYS = ("information_risk", "cyber_risk", "source_credibility", "evidence_strength")


def _json(value) -> str:
    return json.dumps(value, indent=2, ensure_ascii=True)


def export_html(result: dict) -> str:
    """Return a self-contained HTML verification report."""
    claims_html = []
    for number, claim in enumerate(result.get("claims", []), start=1):
        evidence_html = "".join(
            f"<li><strong>{html.escape(str(item.get('title', 'Evidence')))}</strong> "
            f"<a href=\"{html.escape(str(item.get('url', '')), quote=True)}\">"
            f"{html.escape(str(item.get('url', '')))}</a> "
            f"({html.escape(str(item.get('source_type', 'unknown')))}, "
            f"{html.escape(str(item.get('relation', 'neutral')) )})</li>"
            for item in claim.get("evidence", [])
        ) or "<li>No evidence found.</li>"
        claims_html.append(
            f"<section><h2>Claim {number}</h2><p>{html.escape(str(claim.get('claim', '')))}</p>"
            f"<p><strong>Status:</strong> {html.escape(str(claim.get('status', 'insufficient')))}</p>"
            f"<p><strong>Explanation:</strong> {html.escape(str(claim.get('explanation', '')))}</p>"
            f"<h3>Evidence</h3><ul>{evidence_html}</ul></section>"
        )
    scores = result.get("scores", {})
    score_reasons = result.get("score_reasons", {})
    score_html = "".join(
        f"<li><strong>{html.escape(key.replace('_', ' ').title())}:</strong> "
        f"{html.escape(str(scores.get(key, 'unknown')))} / 100"
        f"<ul>{''.join(f'<li>{html.escape(str(reason))}</li>' for reason in score_reasons.get(key, []))}</ul></li>"
        for key in SCORE_KEYS
    )
    return f"""<!doctype html>
<html><head><meta charset="utf-8"><title>TruthShield AI Report</title>
<style>body{{font-family:Arial,sans-serif;max-width:900px;margin:2rem auto;line-height:1.5}} section{{border-top:1px solid #ccc;padding:1rem 0}} pre{{white-space:pre-wrap;background:#f5f5f5;padding:1rem}}</style>
</head><body><h1>TruthShield AI Verification Report</h1>
<p><strong>URL:</strong> {html.escape(str(result.get('url', '')))}</p>
<p><strong>Article title:</strong> {html.escape(str(result.get('title', '')))}</p>
<p><strong>Source:</strong> {html.escape(str(result.get('source', '')))}</p>
<h2>Processed Text Statistics</h2><pre>{html.escape(_json((result.get('text_processing') or {}).get('statistics', result.get('text_statistics', {}))))}</pre>
<h2>Article Text</h2><pre>{html.escape(str(result.get('text', '')))}</pre>
{''.join(claims_html)}
<h2>Security Indicators</h2><pre>{html.escape(_json(result.get('security', {})))}</pre>
<h2>Scores and Explanations</h2><ul>{score_html}</ul>
</body></html>"""


def export_csv(result: dict) -> str:
    """Return a CSV report with one row per claim/evidence combination."""
    output = io.StringIO(newline="")
    writer = csv.writer(output)
    writer.writerow([
        "url", "article_title", "source", "character_count", "word_count", "sentence_count",
        "unique_word_count", "average_sentence_length", "claim", "status", "explanation",
        "evidence_title", "evidence_url", "source_type", "relation", *SCORE_KEYS,
        "security_indicators", "score_explanations",
    ])
    statistics = (result.get("text_processing") or {}).get("statistics", result.get("text_statistics", {}))
    scores = result.get("scores", {})
    claims = result.get("claims", []) or [{}]
    for claim in claims:
        evidence_items = claim.get("evidence", []) or [{}]
        for evidence in evidence_items:
            writer.writerow([
                result.get("url", ""), result.get("title", ""), result.get("source", ""),
                statistics.get("character_count", ""), statistics.get("word_count", ""),
                statistics.get("sentence_count", ""), statistics.get("unique_word_count", ""),
                statistics.get("average_sentence_length", ""), claim.get("claim", ""),
                claim.get("status", ""), claim.get("explanation", ""), evidence.get("title", ""),
                evidence.get("url", ""), evidence.get("source_type", ""), evidence.get("relation", ""),
                *(scores.get(key, "") for key in SCORE_KEYS),
                _json(result.get("security", {})),
                _json(result.get("score_reasons", {})),
            ])
    return output.getvalue()


def export_pdf(result: dict) -> bytes:
    """Return a readable PDF report using Matplotlib's built-in PDF backend."""
    buffer = io.BytesIO()
    with PdfPages(buffer) as pdf:
        lines = [
            "TruthShield AI Verification Report",
            "",
            f"URL: {result.get('url', '')}",
            f"Article title: {result.get('title', '')}",
            f"Source: {result.get('source', '')}",
            "",
            "Processed Text Statistics:",
            _json((result.get("text_processing") or {}).get("statistics", result.get("text_statistics", {}))),
            "",
            "Claims and Verification:",
        ]
        for number, claim in enumerate(result.get("claims", []), start=1):
            lines.extend([
                f"Claim {number}: {claim.get('claim', '')}",
                f"Status: {claim.get('status', '')}",
                f"Explanation: {claim.get('explanation', '')}",
            ])
            for item in claim.get("evidence", []):
                lines.append(
                    f"Evidence: {item.get('title', '')} | {item.get('url', '')} | "
                    f"{item.get('source_type', '')} | {item.get('relation', '')}"
                )
            lines.append("")
        lines.extend(["Security Indicators:", _json(result.get("security", {})), "", "Scores:"])
        for key in SCORE_KEYS:
            lines.append(f"{key.replace('_', ' ').title()}: {result.get('scores', {}).get(key, 'unknown')} / 100")
            lines.extend(f"- {reason}" for reason in result.get("score_reasons", {}).get(key, []))
        lines.extend(["", "Article Text:", str(result.get("text", ""))])
        figure = plt.figure(figsize=(8.5, 11))
        figure.text(0.06, 0.97, "\n".join(textwrap.wrap("\n".join(lines), width=105)), va="top", fontsize=8)
        pdf.savefig(figure, bbox_inches="tight")
        plt.close(figure)
    return buffer.getvalue()
