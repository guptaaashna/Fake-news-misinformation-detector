import streamlit as st

from ui.dashboard import render_dashboard

st.set_page_config(page_title="TruthShield AI", page_icon="TS", layout="wide")

render_dashboard()
