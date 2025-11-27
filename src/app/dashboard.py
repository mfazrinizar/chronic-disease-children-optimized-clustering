import os
import sys

# Ensure project root is on sys.path
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if ROOT not in sys.path:
	sys.path.insert(0, ROOT)

import streamlit as st

from src.app.pages import (
	show_overview,
	show_search_history,
	show_normalized_data,
	show_model_explorer,
	show_comparison,
)


st.set_page_config(
	page_title="Clustering of Provinces in Indonesia Based on Chronic Children Disease",
	page_icon="📊",
	layout="wide",
	initial_sidebar_state="expanded",
)


def main():
	st.title("Clustering of Provinces in Indonesia Based on Chronic Children Disease")
	st.markdown("---")

	with st.sidebar:
		st.header("Navigation")
		page = st.radio(
			"Select Page",
			["Overview", "Search History", "Normalized Data", "Model Explorer", "Comparison"],
		)

	if page == "Overview":
		show_overview()
	elif page == "Search History":
		show_search_history()
	elif page == "Normalized Data":
		show_normalized_data()
	elif page == "Model Explorer":
		show_model_explorer()
	elif page == "Comparison":
		show_comparison()


if __name__ == "__main__":
	main()
