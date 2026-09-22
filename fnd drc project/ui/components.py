"""Reusable presentation helpers for the TruthShield dashboard."""

import streamlit as st


def render_score_card(label: str, value: int, description: str) -> None:
	"""Render one compact score card with an interpretation hint."""
	st.markdown(f"<div class='score-label'>{label}</div>", unsafe_allow_html=True)
	st.markdown(f"<div class='score-value'>{value}<span>/100</span></div>", unsafe_allow_html=True)
	st.caption(description)


def render_status_badge(status: str) -> None:
	"""Render a readable status indicator."""
	styles = {
		"Supported": ("✅", "success"),
		"Contradicted": ("⚠️", "error"),
		"Insufficient Evidence": ("ℹ️", "info"),
	}
	icon, message_type = styles.get(status, ("•", "info"))
	message = f"{icon} {status}"
	getattr(st, message_type)(message)


def render_placeholder_panel(title: str, message: str) -> None:
	"""Render a compact informational panel."""
	with st.container(border=True):
		st.subheader(title)
		st.caption(message)
