import streamlit as st
import pandas as pd
import os

class HomePage:
    """
    Class to handle the home page of the Streamlit application
    """
    
    @staticmethod
    def render():
        """Render the home page"""
        st.title("Livestock Time Series Analysis and Forecasting")
        
        st.markdown("""
        ## Welcome to the Livestock Analyzer!
        
        This application provides comprehensive tools for analyzing and forecasting livestock time series data.
        
        ### Features:
        
        - **Data Exploration**: Visualize and understand your livestock data
        - **Comprehensive Analysis**: Perform detailed statistical analysis on time series data
        - **Forecasting**: Predict future values using ARIMA and hybrid ARIMA-ANN models
        - **Comparative Analysis**: Compare different livestock metrics
        
        ### Available Datasets:
        """)
        
        # Display available datasets
        datasets = HomePage._get_available_datasets()
        
        if datasets:
            for dataset in datasets:
                st.markdown(f"- {dataset}")
        else:
            st.warning("No datasets found. Please upload a dataset using the sidebar.")
        
        # Show sample of datasets
        if datasets:
            st.markdown("### Sample Data:")
            selected_sample = st.selectbox(
                "Select a dataset to preview:",
                options=datasets
            )
            
            if selected_sample:
                try:
                    df = pd.read_csv(os.path.join("data", selected_sample))
                    st.dataframe(df.head())
                except Exception as e:
                    st.error(f"Error loading dataset: {e}")
        
        # Getting started section
        st.markdown("""
        ### Getting Started:
        
        1. Use the sidebar to navigate between different sections
        2. Upload your own CSV data or use the provided datasets
        3. Explore data, perform analysis, and generate forecasts
        
        ### Data Format:
        
        The application supports two CSV formats:
        
        1. **Simple format**: Year and Value columns
        2. **FAO format**: Multiple columns including Domain, Area, Element, Year, Value, etc.
        
        For best results, ensure your data is clean and contains consistent yearly values.
        """)
    
    @staticmethod
    def _get_available_datasets():
        """Get a list of available datasets in the data directory"""
        # Check if data directory exists
        if not os.path.exists("data"):
            os.makedirs("data")
        
        # Get all CSV files in the data directory
        csv_files = []
        for file in os.listdir("data"):
            if file.endswith(".csv"):
                csv_files.append(file)
        
        return csv_files
