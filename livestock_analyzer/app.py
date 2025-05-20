import streamlit as st
import os
import pandas as pd
import numpy as np
from PIL import Image

# Import components
from components.sidebar import Sidebar
from components.home import HomePage
from components.data_explorer import DataExplorerPage
from components.analysis import AnalysisPage
from components.forecasting import ForecastingPage
from components.about import AboutPage

# Set page configuration
st.set_page_config(
    page_title="Livestock Analyzer",
    page_icon="🐄",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Display FarmWise logo and name
try:
    logo = Image.open("logo.png")

    col1, col2 = st.columns([1, 6])
    with col1:
        st.image(logo, width=60)
    with col2:
        st.markdown(
            "<h1 style='color: #2e7d32; margin-top: 50px;'>FarmWise Leading the Future of Farming </h1>",
            unsafe_allow_html=True
        )

    st.markdown("<hr style='border: 1px solid #c8e6c9;'>", unsafe_allow_html=True)
except Exception as e:
    st.warning("⚠️ Could not load logo image. Make sure 'logo.png' exists in the root folder.")

# Initialize session state if it doesn't exist
if 'page' not in st.session_state:
    st.session_state.page = "Home"

if 'selected_dataset' not in st.session_state:
    st.session_state.selected_dataset = None

# Create data directory if it doesn't exist
if not os.path.exists("data"):
    os.makedirs("data")

# Copy CSV files from parent directory if data directory is empty
if len(os.listdir("data")) == 0:
    import shutil
    import glob

    parent_csv_files = glob.glob("../*.csv")
    for file in parent_csv_files:
        if os.path.isfile(file):
            shutil.copy2(file, "data/")
    st.info("✅ Copied dataset files from parent directory to data directory.")

# Create sidebar
sidebar_result = Sidebar.create_sidebar()

# Update session state based on sidebar selection
st.session_state.page = sidebar_result.get("page", "Home")
st.session_state.selected_dataset = sidebar_result.get("selected_dataset", None)

# Display the appropriate page based on selection
if st.session_state.page == "Home":
    HomePage.render()

elif st.session_state.page == "Data Explorer":
    DataExplorerPage.render(st.session_state.selected_dataset)

elif st.session_state.page == "Analysis":
    AnalysisPage.render(st.session_state.selected_dataset)

elif st.session_state.page == "Forecasting":
    ForecastingPage.render(st.session_state.selected_dataset)

elif st.session_state.page == "About":
    AboutPage.render()

