"""Streamlit layout for the TruthShield AI UI skeleton."""

import matplotlib.pyplot as plt
import plotly.graph_objects as go
import streamlit as st

try:
	import seaborn as sns
except (ImportError, OSError):
	sns = None

from services.article import ArticleExtractionError, extract_article
from services.claims import MODEL_INSTALL_MESSAGE, extract_claims, is_full_model_available
from services.export import export_csv, export_html, export_pdf
from services.pipeline import build_analysis_result
from services.storage import load_history, save_analysis
from services.text_processing import process_text
from ui.components import render_score_card


def initialize_session_state() -> None:
	"""Create stable UI state without performing any analysis."""
	defaults = {
		"selected_input_type": "None",
		"selected_input": "",
		"analysis_requested": False,
		"extracted_content": None,
		"extraction_error": "",
		"candidate_claims": [],
		"evidence_results": {},
		"verification_results": {},
		"analysis_result": None,
		"text_processing": None,
		"history": [],
		"selected_history_id": None,
	}
	for key, value in defaults.items():
		st.session_state.setdefault(key, value)


def render_header() -> None:
	st.markdown(
		"""
		<style>
		.block-container {max-width: 1400px; padding-top: 2.2rem; padding-bottom: 3rem;}
		h1 {letter-spacing: -0.02em; margin-bottom: 0.15rem;}
		h2, h3 {margin-top: 1.8rem;}
		.score-label {font-size: 0.78rem; font-weight: 700; color: #52606d; text-transform: uppercase; letter-spacing: 0.06em;}
		.score-value {font-size: 2rem; font-weight: 750; color: #17202a; line-height: 1.15; margin: 0.25rem 0 0.2rem;}
		.score-value span {font-size: 0.95rem; color: #84909c; font-weight: 500;}
		[data-testid="stMetric"] {padding: 0.5rem 0;}
		[data-testid="stExpander"] {border-radius: 8px;}
		</style>
		""",
		unsafe_allow_html=True,
	)
	st.title("TruthShield AI")
	st.subheader("Explainable Information Verification")
	st.write(
		"A transparent workspace for examining an article's claims, evidence, "
		"source context, and URL security indicators."
	)
	st.divider()


def render_input_section() -> None:
	st.header("1. Analyze an article URL")
	url = st.text_input(
		"Article URL",
		placeholder="https://example.com/news-article",
		key="article_url",
	)
	if st.button("Analyze URL", type="primary", key="analyze_url"):
		st.session_state.selected_input_type = "Article URL"
		st.session_state.selected_input = url
		st.session_state.analysis_requested = False
		st.session_state.extracted_content = None
		st.session_state.extraction_error = ""
		st.session_state.candidate_claims = []
		st.session_state.evidence_results = {}
		st.session_state.verification_results = {}
		st.session_state.analysis_result = None
		st.session_state.text_processing = None
		if not url.strip():
			st.warning("Enter an article URL to extract article content.")
		else:
			with st.spinner("Fetching and extracting article text..."):
				try:
					st.session_state.extracted_content = extract_article(url)
					st.session_state.text_processing = process_text(
						st.session_state.extracted_content["text"]
					)
					st.session_state.candidate_claims = extract_claims(
						st.session_state.extracted_content["text"]
					)
					_set_analysis_result("url", st.session_state.extracted_content)
					st.session_state.analysis_requested = True
				except ArticleExtractionError as error:
					st.session_state.extraction_error = str(error)
			if st.session_state.extracted_content:
				st.success("Article extracted successfully.")
			else:
				st.error(st.session_state.extraction_error)


def _set_analysis_result(input_type: str, content: dict[str, str]) -> None:
	"""Build and cache the complete result for one URL/text analysis."""
	with st.spinner("Searching external evidence..."):
		result = build_analysis_result(
			input_type,
			content,
			st.session_state.candidate_claims,
			url=st.session_state.selected_input,
		)
		st.session_state.analysis_result = result
		result["url"] = st.session_state.selected_input
		result["text_processing"] = st.session_state.text_processing
		result["history_id"] = save_analysis(result)
		st.session_state.evidence_results = {
			item["claim"]: item["evidence"] for item in result["claims"]
		}
		st.session_state.verification_results = {
			item["claim"]: item for item in result["claims"]
		}


def render_extracted_content() -> None:
	st.header("2. Extracted Content")
	content = st.session_state.get("extracted_content")
	with st.expander("View extracted content", expanded=content is not None):
		if content is None:
			st.info("Enter a URL or paste text above to display extracted content.")
			return
		with st.container(border=True):
			content_columns = st.columns([1, 2])
			with content_columns[0]:
				st.caption("Article title")
				st.write(content["title"])
				st.caption("Source")
				st.write(content["source"])
			with content_columns[1]:
				st.caption("Extracted text")
				st.text_area(
					"Extracted text preview",
					value=content["text"],
					height=130,
					disabled=True,
					label_visibility="collapsed",
				)


def render_claim_analysis() -> None:
	st.header("4. Candidate Claims")
	st.caption(
		"These are candidate factual claims extracted from the input. "
		"Verification status is based only on the evidence currently available."
	)
	st.warning(
		"Verification is based on the evidence currently available to the system. "
		"Insufficient Evidence does not mean the claim is false."
	)
	claims = st.session_state.get("candidate_claims", [])
	if not claims and st.session_state.get("extracted_content"):
		claims = extract_claims(st.session_state["extracted_content"]["text"])
		st.session_state.candidate_claims = claims
	if not claims:
		st.info("No suitable factual-looking candidate claims were found.")
		return
	for number, claim in enumerate(claims, start=1):
		with st.expander(f"Claim {number}: {claim}", expanded=number == 1):
			verification = st.session_state.get("verification_results", {}).get(claim)
			if verification is None:
				verification = {
					"claim": claim,
					"status": "insufficient",
					"explanation": "No analysis result is available for this claim yet.",
					"evidence": [],
				}
			evidence = verification["evidence"]
			status = verification["status"]
			status_label = {
				"supported": "SUPPORTED",
				"contradicted": "CONTRADICTED",
				"insufficient": "INSUFFICIENT EVIDENCE",
			}[status]
			if status == "supported":
				st.success(f"Verification status: {status_label}")
			elif status == "contradicted":
				st.error(f"Verification status: {status_label}")
			else:
				st.info(f"Verification status: {status_label}")
			if evidence:
				fact_checks = [item for item in evidence if item["source_type"] == "fact_check"]
				news_results = [item for item in evidence if item["source_type"] == "news"]
				if fact_checks:
					st.markdown("**Fact Check Sources**")
					_render_evidence_items(fact_checks)
				if news_results:
					st.markdown("**News/Context Sources**")
					_render_evidence_items(news_results)
			else:
				st.warning(
					"No external evidence is currently available. "
					"No relevant evidence was found for this claim."
				)
			st.markdown(f"**Explanation**  \n{verification['explanation']}")
	if not is_full_model_available():
		st.warning(MODEL_INSTALL_MESSAGE)


def render_text_processing() -> None:
	st.header("3. Text Processing")
	processed = st.session_state.get("text_processing")
	if not processed:
		st.info("Analyze a URL to process the extracted article text.")
		return
	statistics = processed["statistics"]
	stat_columns = st.columns(5)
	for column, (label, value) in zip(
		stat_columns,
		[
			("Characters", statistics["character_count"]),
			("Words", statistics["word_count"]),
			("Sentences", statistics["sentence_count"]),
			("Unique Words", statistics["unique_word_count"]),
			("Avg. Sentence Length", f"{statistics['average_sentence_length']:.1f}"),
		],
	):
		with column:
			st.metric(label, value)
	with st.expander("Cleaned text and token previews"):
		st.text_area("Cleaned text", processed["cleaned_text"], height=140, disabled=True)
		preview_columns = st.columns(2)
		with preview_columns[0]:
			st.caption("Token preview")
			st.code(" ".join(processed["tokens"][:40]) or "No tokens")
		with preview_columns[1]:
			st.caption("Lemmatized token preview")
			st.code(" ".join(processed["lemmatized_tokens"][:40]) or "No tokens")
	chart_columns = st.columns(2)
	with chart_columns[0]:
		frequency = processed["word_frequency"]
		if frequency:
			frequency_items = list(frequency.items())[:15]
			figure, axis = plt.subplots(figsize=(7, 4))
			if sns is not None:
				sns.barplot(
					x=[item[1] for item in frequency_items],
					y=[item[0] for item in frequency_items],
					hue=[item[0] for item in frequency_items],
					legend=False,
					palette="crest",
					ax=axis,
				)
			else:
				axis.barh(
					[item[0] for item in frequency_items],
					[item[1] for item in frequency_items],
					color="#168f80",
				)
			axis.set_title("Top Word Frequency")
			figure.tight_layout()
			st.pyplot(figure)
			plt.close(figure)
		else:
			st.info("No words available for frequency analysis.")
	with chart_columns[1]:
		sentence_lengths = [len(sentence.split()) for sentence in processed["sentences"]]
		if sentence_lengths:
			figure, axis = plt.subplots(figsize=(7, 4))
			if sns is not None:
				sns.histplot(
					x=sentence_lengths,
					bins=min(10, len(sentence_lengths)),
					kde=False,
					color="#168f80",
					ax=axis,
				)
			else:
				axis.hist(sentence_lengths, bins=min(10, len(sentence_lengths)), color="#168f80")
			axis.set_title("Sentence-Length Distribution")
			figure.tight_layout()
			st.pyplot(figure)
			plt.close(figure)
		else:
			st.info("No sentences available for length analysis.")


def _render_evidence_items(items: list[dict]) -> None:
	for item in items:
		st.markdown(
			f"- **{item['title']}**  "
			f"\n  URL: [{item['url']}]({item['url']})  "
			f"\n  Source type: `{item['source_type']}` | Relation: `{item['relation']}`"
		)


def render_scores() -> None:
	st.header("5. Prototype Scores")
	analysis_result = st.session_state.get("analysis_result") or {}
	scores = analysis_result.get("scores")
	reasons = analysis_result.get("score_reasons", {})
	if not scores:
		st.info("Analyze a URL to calculate prototype score indicators.")
		return
	st.caption(
		"These scores are heuristic prototype indicators based on the evidence and URL "
		"signals available to the system. They are not scientifically validated probabilities "
		"and should not be interpreted as definitive proof that an article is true or false."
	)
	scores = [
		("Information Risk", scores["information_risk"], "Higher means more verification concern.", "information_risk"),
		("Cyber Risk", scores["cyber_risk"], "Higher means more website/security concern.", "cyber_risk"),
		("Source Credibility", scores["source_credibility"], "Higher means stronger source context.", "source_credibility"),
		("Evidence Strength", scores["evidence_strength"], "Higher means stronger available evidence.", "evidence_strength"),
	]
	columns = st.columns(4)
	for column, (label, value, description, reason_key) in zip(columns, scores):
		with column:
			with st.container(border=True):
				render_score_card(label, value, description)
			with st.expander("Why this score?"):
				for reason in reasons.get(reason_key, []):
					st.write(f"- {reason}")


def render_visualizations() -> None:
	st.header("6. Visualizations")
	st.caption("Charts summarize the current URL analysis.")
	chart_columns = st.columns(2)

	claim_results = (st.session_state.get("analysis_result") or {}).get("claims", [])
	status_counts = {"supported": 0, "contradicted": 0, "insufficient": 0}
	for claim in claim_results:
		status = claim.get("status", "insufficient")
		if status in status_counts:
			status_counts[status] += 1
	claim_data = {
		"Status": ["Supported", "Contradicted", "Insufficient Evidence"],
		"Claims": [
			status_counts["supported"],
			status_counts["contradicted"],
			status_counts["insufficient"],
		],
	}
	with chart_columns[0]:
		if claim_results:
			status_chart = go.Figure(
				data=[
					go.Bar(
						x=claim_data["Status"],
						y=claim_data["Claims"],
						marker_color=["#168f80", "#d86645", "#d99b35"],
						showlegend=False,
					)
				]
			)
			status_chart.update_layout(title="Claim Status Distribution", height=340, showlegend=False)
			st.plotly_chart(status_chart, use_container_width=True)
		else:
			st.info("Analyze a URL to display claim visualizations.")

	with chart_columns[1]:
		computed_scores = (st.session_state.get("analysis_result") or {}).get("scores")
		if computed_scores:
			score_data = {
				"Signal": ["Information Risk", "Cyber Risk", "Source Credibility", "Evidence Strength"],
				"Score": [
					computed_scores["information_risk"],
					computed_scores["cyber_risk"],
					computed_scores["source_credibility"],
					computed_scores["evidence_strength"],
				],
			}
			comparison_chart = go.Figure(
				data=[
					go.Bar(
						x=score_data["Score"],
						y=score_data["Signal"],
						orientation="h",
						marker=dict(
							color=score_data["Score"],
							colorscale="Tealgrn",
							showscale=False,
						),
						showlegend=False,
					)
				]
			)
			comparison_chart.update_layout(title="Computed Score Comparison", height=340, showlegend=False)
			st.plotly_chart(comparison_chart, use_container_width=True)
		else:
			st.info("Analyze a URL to display computed score comparisons.")

	with st.expander("Static claim-status view"):
		if claim_results:
			figure, axis = plt.subplots(figsize=(8, 3.2))
			statuses = [claim.get("status", "insufficient") for claim in claim_results]
			if sns is not None:
				sns.countplot(x=statuses, hue=statuses, palette="Set2", legend=False, ax=axis)
			else:
				labels = list(dict.fromkeys(statuses))
				axis.bar(labels, [statuses.count(label) for label in labels], color="#168f80")
			axis.set_title("Current Claim Status Frequency")
			axis.set_xlabel("")
			axis.set_ylabel("Claims")
			figure.tight_layout()
			st.pyplot(figure)
			plt.close(figure)
		else:
			st.info("No claim data is available yet.")


def render_security_panel() -> None:
	st.header("7. Cybersecurity Indicators")
	security = (st.session_state.get("analysis_result") or {}).get("security", {})
	if not security:
		st.info("Analyze a URL to collect security indicators.")
		return
	indicators = [
		("HTTPS", "Enabled" if security.get("https") else "Not Enabled"),
		("Hostname", security.get("hostname", "unknown")),
		("DNS", "Resolves" if security.get("dns_resolves") else "Does Not Resolve"),
		("SSL", "Available" if security.get("ssl_available") else "Unavailable"),
		(
			"Suspicious URL Pattern",
			"Detected" if security.get("suspicious_url") else "Not Detected",
		),
		("Threat Reputation", security.get("threat_reputation", "unknown").title()),
	]
	columns = st.columns(len(indicators))
	for column, (label, value) in zip(columns, indicators):
		with column:
			with st.container(border=True):
				st.caption(label)
				st.write(value)
	st.markdown("**Security Indicators**")
	for indicator in security.get("indicators", []):
		st.write(f"- {indicator}")
	st.caption(
		"These are basic URL and connection security indicators. They are not proof "
		"that a website is malicious and do not determine whether the article's claims "
		"are true or false."
	)


def render_history_and_export() -> None:
	history_column, export_column = st.columns(2)
	with history_column:
		st.header("8. Analysis History")
		history = load_history()
		st.session_state.history = history
		with st.container(border=True):
			if not history:
				st.info("No previous analyses.")
			else:
				options = {
					f"{item['timestamp']} | {item['url']}": item["history_id"]
					for item in history
				}
				selected_label = st.selectbox("Select an analysis", list(options))
				selected_id = options[selected_label]
				selected = next(item for item in history if item["history_id"] == selected_id)
				st.caption(f"Title: {selected['title']}")
				st.write(f"Claims: {len(selected.get('claims', []))}")
				st.write(f"Information Risk: {selected.get('scores', {}).get('information_risk', 'unknown')}/100")
				with st.expander("View saved analysis"):
					st.json(selected)
	with export_column:
		st.header("9. Export Report")
		st.caption("Download the current URL analysis in a portable report format.")
		export_buttons = st.columns(3)
		analysis_result = st.session_state.get("analysis_result")
		if analysis_result:
			exports = [
				(export_html(analysis_result), "truthshield-report.html", "text/html", "Export HTML"),
				(export_csv(analysis_result), "truthshield-report.csv", "text/csv", "Export CSV"),
				(export_pdf(analysis_result), "truthshield-report.pdf", "application/pdf", "Export PDF"),
			]
			for column, (data, filename, mime, label) in zip(export_buttons, exports):
				with column:
					st.download_button(label, data, filename, mime=mime, use_container_width=True)
		else:
			for column, label in zip(export_buttons, ["Export HTML", "Export CSV", "Export PDF"]):
				with column:
					st.download_button(label, "", disabled=True, use_container_width=True)


def render_dashboard() -> None:
    """Render the complete Step 2 interface."""
    initialize_session_state()
    render_header()
    render_input_section()
    render_extracted_content()
    render_text_processing()
    render_claim_analysis()
    render_scores()
    render_visualizations()
    render_security_panel()
    render_history_and_export()


render_dashboard()
