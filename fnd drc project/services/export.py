"""Export completed TruthShield analyses to HTML, CSV, and PDF."""

import csv
import html
import io
import json
from pathlib import Path

from matplotlib import font_manager
from reportlab.lib import colors
from reportlab.lib.enums import TA_LEFT
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import inch
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import KeepTogether, Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle


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
    """Return a purpose-built, paginated verification report."""
    unicode_font = Path("C:/Windows/Fonts/ARIALUNI.ttf")
    if unicode_font.exists():
        font_path = str(unicode_font)
        bold_font_path = str(unicode_font)
    else:
        font_path = font_manager.findfont("DejaVu Sans")
        bold_font_path = font_manager.findfont(
            font_manager.FontProperties(family="DejaVu Sans", weight="bold")
        )
    pdfmetrics.registerFont(TTFont("TruthShieldSans", font_path))
    pdfmetrics.registerFont(TTFont("TruthShieldSansBold", bold_font_path))

    styles = getSampleStyleSheet()
    body_style = ParagraphStyle(
        "TruthShieldBody",
        parent=styles["BodyText"],
        fontName="TruthShieldSans",
        fontSize=9,
        leading=12,
        alignment=TA_LEFT,
        spaceAfter=5,
        wordWrap="CJK",
    )
    small_style = ParagraphStyle(
        "TruthShieldSmall",
        parent=body_style,
        fontSize=8,
        leading=10,
        spaceAfter=0,
    )
    label_style = ParagraphStyle(
        "TruthShieldLabel",
        parent=body_style,
        fontName="TruthShieldSansBold",
        spaceAfter=2,
    )
    heading_style = ParagraphStyle(
        "TruthShieldHeading",
        parent=styles["Heading2"],
        fontName="TruthShieldSansBold",
        fontSize=13,
        leading=16,
        textColor=colors.HexColor("#17202a"),
        spaceBefore=10,
        spaceAfter=6,
    )
    title_style = ParagraphStyle(
        "TruthShieldTitle",
        parent=styles["Title"],
        fontName="TruthShieldSansBold",
        fontSize=18,
        leading=22,
        spaceAfter=12,
    )
    subtitle_style = ParagraphStyle(
        "TruthShieldSubtitle",
        parent=body_style,
        fontSize=11,
        leading=14,
        textColor=colors.HexColor("#52606d"),
        spaceAfter=12,
    )
    score_label_style = ParagraphStyle(
        "TruthShieldScoreLabel",
        parent=label_style,
        fontSize=9,
        leading=11,
        textColor=colors.HexColor("#52606d"),
    )
    score_value_style = ParagraphStyle(
        "TruthShieldScoreValue",
        parent=body_style,
        fontName="TruthShieldSansBold",
        fontSize=17,
        leading=20,
        textColor=colors.HexColor("#17202a"),
        spaceAfter=3,
    )

    def paragraph(value, style=body_style):
        text = html.escape(str(value)).replace("\n", "<br/>")
        return Paragraph(text or "-", style)

    def humanize(value) -> str:
        return str(value).replace("_", " ").strip().title()

    def input_type_label(value) -> str:
        return {"url": "URL", "text": "Text", "image": "Image"}.get(
            str(value).lower(), humanize(value)
        )

    def status_label(value) -> str:
        return {
            "supported": "Supported",
            "contradicted": "Contradicted",
            "insufficient": "Insufficient Evidence",
        }.get(str(value).lower(), humanize(value))

    def relation_label(value) -> str:
        return {"supports": "Supports", "contradicts": "Contradicts", "neutral": "Neutral"}.get(
            str(value).lower(), humanize(value)
        )

    def evidence_type_label(value) -> str:
        return {"fact_check": "Fact Check", "news": "News", "official": "Official"}.get(
            str(value).lower(), humanize(value)
        )

    def table(data, widths, header=True, repeat_rows=0):
        style_commands = [
            ("FONTNAME", (0, 0), (-1, -1), "TruthShieldSans"),
            ("FONTSIZE", (0, 0), (-1, -1), 8.5),
            ("LEADING", (0, 0), (-1, -1), 10),
            ("VALIGN", (0, 0), (-1, -1), "TOP"),
            ("LEFTPADDING", (0, 0), (-1, -1), 7),
            ("RIGHTPADDING", (0, 0), (-1, -1), 7),
            ("TOPPADDING", (0, 0), (-1, -1), 6),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
            ("GRID", (0, 0), (-1, -1), 0.35, colors.HexColor("#d9e2ec")),
        ]
        if header:
            style_commands.extend([
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#e9f0f5")),
                ("FONTNAME", (0, 0), (-1, 0), "TruthShieldSansBold"),
                ("TEXTCOLOR", (0, 0), (-1, 0), colors.HexColor("#17202a")),
            ])
        return Table(data, colWidths=widths, repeatRows=repeat_rows, splitByRow=1, hAlign="LEFT"), style_commands

    def draw_footer(canvas, document):
        canvas.saveState()
        canvas.setStrokeColor(colors.HexColor("#d9e2ec"))
        canvas.line(document.leftMargin, 0.48 * inch, letter[0] - document.rightMargin, 0.48 * inch)
        canvas.setFont("TruthShieldSans", 8)
        canvas.setFillColor(colors.HexColor("#7b8794"))
        canvas.drawRightString(letter[0] - document.rightMargin, 0.3 * inch, f"Page {document.page}")
        canvas.restoreState()

    buffer = io.BytesIO()
    document = SimpleDocTemplate(
        buffer,
        pagesize=letter,
        rightMargin=0.65 * inch,
        leftMargin=0.65 * inch,
        topMargin=0.65 * inch,
        bottomMargin=0.65 * inch,
        title="TruthShield AI Verification Report",
        author="TruthShield AI",
    )
    content_width = letter[0] - document.leftMargin - document.rightMargin
    story = [
        Paragraph("TRUTHSHIELD AI", title_style),
        Paragraph("AI-Powered Content Verification Report", subtitle_style),
        Paragraph("1. Analysis Overview", heading_style),
    ]
    overview_data = [
        [paragraph("Input Type", label_style), paragraph(input_type_label(result.get("input_type", "")))],
        [paragraph("Article Title", label_style), paragraph(result.get("title", ""))],
        [paragraph("Source", label_style), paragraph(result.get("source", ""))],
        [paragraph("URL", label_style), paragraph(result.get("url", ""), small_style)],
    ]
    overview_table, overview_style = table(overview_data, [1.35 * inch, content_width - 1.35 * inch], header=False)
    story.extend([overview_table, Spacer(1, 8)])

    story.append(Paragraph("2. Risk Summary", heading_style))
    scores = result.get("scores", {}) or {}
    score_reasons = result.get("score_reasons", {}) or {}
    score_cells = []
    for key in SCORE_KEYS:
        score_cells.append([
            Paragraph(humanize(key), score_label_style),
            Paragraph(f"{scores.get(key, 'unknown')} / 100", score_value_style),
            paragraph(" ".join(str(reason) for reason in score_reasons.get(key, [])) or "No explanation available.", small_style),
        ])
    score_table = Table([score_cells[:2], score_cells[2:]], colWidths=[content_width / 2] * 2, hAlign="LEFT")
    score_table.setStyle(TableStyle([
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("BACKGROUND", (0, 0), (-1, -1), colors.HexColor("#f5f8fa")),
        ("BOX", (0, 0), (-1, -1), 0.5, colors.HexColor("#d9e2ec")),
        ("INNERGRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#d9e2ec")),
        ("LEFTPADDING", (0, 0), (-1, -1), 10),
        ("RIGHTPADDING", (0, 0), (-1, -1), 10),
        ("TOPPADDING", (0, 0), (-1, -1), 9),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 9),
    ]))
    story.extend([score_table, Spacer(1, 8)])

    story.append(Paragraph("3. Claim Verification", heading_style))
    claims = result.get("claims", [])
    if not claims:
        story.append(Paragraph("No claims were extracted.", body_style))
    for number, claim in enumerate(claims, start=1):
        evidence_items = claim.get("evidence", []) or []
        claim_story = [
            Paragraph(f"Claim {number}", heading_style),
            paragraph(f'"{claim.get("claim", "")}"', body_style),
            paragraph(f"Status: {status_label(claim.get('status', 'insufficient'))}", label_style),
            paragraph(f"Explanation: {claim.get('explanation', '') or 'No explanation available.'}"),
            paragraph(f"Evidence: {len(evidence_items)} source(s) found." if evidence_items else "Evidence: No relevant evidence found."),
        ]
        claim_table = Table([[claim_story]], colWidths=[content_width], hAlign="LEFT")
        claim_table.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, -1), colors.HexColor("#f8fafb")),
            ("BOX", (0, 0), (-1, -1), 0.5, colors.HexColor("#d9e2ec")),
            ("LEFTPADDING", (0, 0), (-1, -1), 10),
            ("RIGHTPADDING", (0, 0), (-1, -1), 10),
            ("TOPPADDING", (0, 0), (-1, -1), 7),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
        ]))
        story.extend([claim_table, Spacer(1, 8)])

    story.append(Paragraph("4. Evidence", heading_style))
    evidence_rows = [[paragraph("Source", label_style), paragraph("Title", label_style), paragraph("URL", label_style), paragraph("Type", label_style), paragraph("Relation", label_style)]]
    all_evidence = [(number, item) for number, claim in enumerate(claims, start=1) for item in claim.get("evidence", []) or []]
    for number, item in all_evidence:
        evidence_rows.append([
            paragraph(f"Claim {number}"),
            paragraph(item.get("title", "")),
            paragraph(item.get("url", ""), small_style),
            paragraph(evidence_type_label(item.get("source_type", "unknown"))),
            paragraph(relation_label(item.get("relation", "neutral"))),
        ])
    if len(evidence_rows) == 1:
        story.append(Paragraph("No relevant evidence found.", body_style))
    else:
        evidence_table, evidence_style = table(evidence_rows, [0.7 * inch, 1.45 * inch, 2.6 * inch, 0.85 * inch, 0.85 * inch], repeat_rows=1)
        story.append(evidence_table)

    story.append(Paragraph("5. Cybersecurity Analysis", heading_style))
    security = result.get("security", {}) or {}
    security_rows = [[paragraph("Indicator", label_style), paragraph("Result", label_style)]]
    security_rows.extend([
        [paragraph("HTTPS"), paragraph("Enabled" if security.get("https") else "Not enabled")],
        [paragraph("Suspicious URL"), paragraph("Detected" if security.get("suspicious_url") else "Not detected")],
        [paragraph("Hostname Resolution"), paragraph("Successful" if security.get("dns_resolves") else "Failed")],
        [paragraph("SSL Certificate"), paragraph("Valid" if security.get("ssl_available") else "Unavailable")],
        [paragraph("Threat Reputation"), paragraph(humanize(security.get("threat_reputation", "unknown")))],
    ])
    certificate = security.get("certificate", {}) or {}
    if certificate.get("issuer"):
        security_rows.append([paragraph("Certificate Issuer"), paragraph(certificate["issuer"])])
    if certificate.get("expires"):
        security_rows.append([paragraph("Certificate Expiry"), paragraph(certificate["expires"])])
    for indicator in security.get("indicators", []) or []:
        security_rows.append([paragraph("Indicator"), paragraph(indicator)])
    security_table, security_style = table(security_rows, [2.1 * inch, content_width - 2.1 * inch])
    story.append(security_table)

    document.build(story, onFirstPage=draw_footer, onLaterPages=draw_footer)
    return buffer.getvalue()
