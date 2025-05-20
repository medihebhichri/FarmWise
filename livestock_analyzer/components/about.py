import streamlit as st

class AboutPage:
    """
    Class to handle the about page of the Streamlit application
    """
    
    @staticmethod
    def render():
        """Render the about page"""
        st.title("About Livestock Analyzer")
        
        st.markdown("""
        ## Project Overview
        
        The Livestock Analyzer is a comprehensive tool for analyzing and forecasting livestock time series data. It provides users with the ability to:
        
        - **Visualize** livestock time series data
        - **Analyze** trends, growth rates, and statistical properties
        - **Forecast** future values using state-of-the-art models
        - **Compare** different forecasting methodologies
        
        ## Models and Methodologies
        
        ### Time Series Analysis
        
        The application utilizes various statistical methods to analyze time series data:
        
        - **Basic Statistics**: Calculate descriptive statistics on the time series
        - **Trend Analysis**: Identify and quantify long-term trends
        - **Growth Rate Analysis**: Calculate year-over-year growth and CAGR
        - **Stationarity Test**: Check if the time series is stationary using Augmented Dickey-Fuller test
        - **Volatility Analysis**: Measure the variability of the time series
        - **Significant Changes**: Identify years with notable changes in values
        
        ### Forecasting Models
        
        The application implements two main forecasting models:
        
        1. **ARIMA (Autoregressive Integrated Moving Average)**
           - A classical statistical method for time series forecasting
           - Combines autoregression (AR), differencing (I), and moving average (MA) components
           - Well-suited for capturing linear trends and seasonality
        
        2. **ARIMA-ANN Hybrid**
           - A hybrid model that combines ARIMA with Artificial Neural Networks
           - ARIMA captures the linear component of the time series
           - ANN learns the residual patterns that ARIMA cannot capture
           - Often outperforms pure ARIMA models, especially for complex patterns
        
        ## Data Format
        
        The application supports two CSV formats:
        
        1. **Simple format**: Contains 'Year' and 'Value' columns
        2. **FAO format**: Contains multiple columns including 'Domain', 'Area', 'Element', 'Item', 'Year', 'Unit', and 'Value'
        
        ## Contact Information
        
        For more information or support, please contact the development team.
        
        ## Version Information
        
        Current Version: 1.0.0  
        Last Updated: April 2025
        """)