import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

class DataExplorerPage:
    """
    Class to handle the data explorer page of the Streamlit application
    """
    
    @staticmethod
    def render(selected_dataset):
        """
        Render the data explorer page
        
        Parameters:
        -----------
        selected_dataset : str
            Name of the selected dataset
        """
        st.title("Data Explorer")
        
        if not selected_dataset:
            st.warning("Please select a dataset from the sidebar.")
            return
        
        # Load the dataset
        try:
            file_path = os.path.join("data", selected_dataset)
            df = pd.read_csv(file_path)
            
            # Display basic information
            st.subheader(f"Dataset: {selected_dataset}")
            
            # Check if it's a standard format or multi-column format
            if 'Year' in df.columns and 'Value' in df.columns:
                is_standard_format = True
                
                # For multi-column FAO format, display metadata if available
                if 'Domain' in df.columns or 'Area' in df.columns or 'Element' in df.columns:
                    is_standard_format = False
                    st.subheader("Dataset Metadata")
                    metadata_cols = [col for col in df.columns if col not in ['Year', 'Value']]
                    metadata = {col: df[col].unique()[0] if len(df[col].unique()) == 1 else df[col].unique() for col in metadata_cols}
                    
                    for col, value in metadata.items():
                        if isinstance(value, np.ndarray) and len(value) == 1:
                            st.text(f"{col}: {value[0]}")
                        else:
                            st.text(f"{col}: {value}")
            else:
                is_standard_format = False
                st.warning("Dataset doesn't have expected 'Year' and 'Value' columns. Analysis may be limited.")
            
            # Display the first few rows
            st.subheader("Data Preview")
            st.dataframe(df.head())
            
            # Display data information
            st.subheader("Data Information")
            
            # Create two columns for info
            col1, col2 = st.columns(2)
            
            with col1:
                st.text(f"Rows: {df.shape[0]}")
                st.text(f"Columns: {df.shape[1]}")
                
                if 'Year' in df.columns:
                    years = df['Year'].unique()
                    st.text(f"Year range: {min(years)} - {max(years)}")
                    st.text(f"Number of years: {len(years)}")
            
            with col2:
                if 'Value' in df.columns:
                    st.text(f"Min value: {df['Value'].min():.2f}")
                    st.text(f"Max value: {df['Value'].max():.2f}")
                    st.text(f"Mean value: {df['Value'].mean():.2f}")
                    st.text(f"Median value: {df['Value'].median():.2f}")
            
            # Interactive data exploration
            st.subheader("Interactive Data Exploration")
            
            # Display full data
            if st.checkbox("Show full dataset"):
                st.dataframe(df)
            
            # Column selection for further analysis
            if is_standard_format:
                # Time series visualization
                st.subheader("Time Series Visualization")
                
                # Prepare data for plotting
                if 'Year' in df.columns and 'Value' in df.columns:
                    # Sort by year
                    plot_df = df.sort_values(by='Year')
                    
                    fig, ax = plt.subplots(figsize=(12, 6))
                    sns.lineplot(x='Year', y='Value', data=plot_df, ax=ax)
                    ax.set_title(f'Time Series: {selected_dataset}', fontsize=16)
                    ax.set_xlabel('Year', fontsize=12)
                    ax.set_ylabel('Value', fontsize=12)
                    st.pyplot(fig)
                    
                    # Plot options
                    st.subheader("Visualization Options")
                    
                    # Rolling average
                    if st.checkbox("Show rolling average"):
                        window_size = st.slider("Window size for rolling average", min_value=2, max_value=10, value=3)
                        
                        # Create a time series with year as index
                        ts_df = plot_df.set_index('Year')['Value']
                        rolling_avg = ts_df.rolling(window=window_size).mean()
                        
                        fig, ax = plt.subplots(figsize=(12, 6))
                        ax.plot(ts_df.index, ts_df.values, 'b-', label='Original')
                        ax.plot(rolling_avg.index, rolling_avg.values, 'r-', label=f'{window_size}-Year Rolling Avg')
                        ax.set_title(f'Time Series with Rolling Average: {selected_dataset}', fontsize=16)
                        ax.set_xlabel('Year', fontsize=12)
                        ax.set_ylabel('Value', fontsize=12)
                        ax.legend()
                        st.pyplot(fig)
                    
                    # Year-over-year growth rate
                    if st.checkbox("Show year-over-year growth rate"):
                        # Create a time series with year as index
                        ts_df = plot_df.set_index('Year')['Value']
                        growth_rate = ts_df.pct_change() * 100
                        
                        fig, ax = plt.subplots(figsize=(12, 6))
                        ax.bar(growth_rate.index, growth_rate.values, color='steelblue')
                        ax.axhline(y=0, color='r', linestyle='-', alpha=0.3)
                        ax.set_title(f'Year-over-Year Growth Rate: {selected_dataset}', fontsize=16)
                        ax.set_xlabel('Year', fontsize=12)
                        ax.set_ylabel('Growth Rate (%)', fontsize=12)
                        st.pyplot(fig)
                    
                    # Distribution of values
                    if st.checkbox("Show distribution of values"):
                        fig, ax = plt.subplots(figsize=(12, 6))
                        sns.histplot(plot_df['Value'], kde=True, ax=ax)
                        ax.set_title(f'Value Distribution: {selected_dataset}', fontsize=16)
                        ax.set_xlabel('Value', fontsize=12)
                        ax.set_ylabel('Frequency', fontsize=12)
                        st.pyplot(fig)
                        
                        # Show basic stats
                        st.text(f"Standard deviation: {plot_df['Value'].std():.2f}")
                        st.text(f"Skewness: {plot_df['Value'].skew():.2f}")
                        st.text(f"Kurtosis: {plot_df['Value'].kurtosis():.2f}")
                
            else:
                st.info("For non-standard data formats, analysis options are limited. Please ensure your data has 'Year' and 'Value' columns for full functionality.")
        
        except Exception as e:
            st.error(f"Error loading or analyzing dataset: {e}")
