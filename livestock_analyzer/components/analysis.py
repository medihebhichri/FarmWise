import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from scipy import stats
from statsmodels.tsa.stattools import adfuller

from utils.time_series_utils import TimeSeriesUtils

class AnalysisPage:
    """
    Class to handle the analysis page of the Streamlit application
    """
    
    @staticmethod
    def render(selected_dataset):
        """
        Render the analysis page
        
        Parameters:
        -----------
        selected_dataset : str
            Name of the selected dataset
        """
        st.title("Time Series Analysis")
        
        if not selected_dataset:
            st.warning("Please select a dataset from the sidebar.")
            return
        
        # Load the dataset
        try:
            file_path = os.path.join("data", selected_dataset)
            df = pd.read_csv(file_path)
            
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
                return
            
            # Preprocess data
            df['Year'] = pd.to_numeric(df['Year'], errors='coerce')
            df['Value'] = pd.to_numeric(df['Value'], errors='coerce')
            
            # Sort by year
            df = df.sort_values(by='Year')
            
            # Set index to year for time series analysis
            ts_df = df[['Year', 'Value']].set_index('Year')
            
            # Add analysis options in sidebar
            st.sidebar.header("Analysis Options")
            
            analysis_options = st.sidebar.multiselect(
                "Select analyses to perform:",
                [
                    "Basic Statistics", 
                    "Trend Analysis", 
                    "Growth Rates Analysis", 
                    "Stationarity Test", 
                    "Volatility Analysis", 
                    "Significant Changes"
                ],
                default=["Basic Statistics", "Trend Analysis"]
            )
            
            # Display dataset info
            st.subheader(f"Dataset: {selected_dataset}")
            st.write(f"Time range: {df['Year'].min()} - {df['Year'].max()}")
            st.write(f"Number of records: {len(df)}")
            
            # Display dataset plot
            st.subheader("Time Series Plot")
            fig, ax = plt.subplots(figsize=(12, 6))
            ax.plot(df['Year'], df['Value'])
            ax.set_title(f'Time Series: {selected_dataset}', fontsize=16)
            ax.set_xlabel('Year', fontsize=12)
            ax.set_ylabel('Value', fontsize=12)
            ax.grid(True)
            st.pyplot(fig)
            
            # Perform selected analyses
            if "Basic Statistics" in analysis_options:
                st.subheader("Basic Statistics")
                basic_stats = TimeSeriesUtils.calculate_basic_stats(ts_df, 'Value')
                
                # Create a cleaner display with columns
                col1, col2 = st.columns(2)
                
                with col1:
                    st.write("**Time Range Statistics:**")
                    st.write(f"- Start year: {basic_stats['start_year']}")
                    st.write(f"- End year: {basic_stats['end_year']}")
                    st.write(f"- Time span: {basic_stats['time_span']} years")
                    
                    st.write("**Central Tendency:**")
                    st.write(f"- Mean: {basic_stats['mean']:.2f}")
                    st.write(f"- Median: {basic_stats['median']:.2f}")
                
                with col2:
                    st.write("**Extremes:**")
                    st.write(f"- Minimum: {basic_stats['min']:.2f} (in {basic_stats['min_year']})")
                    st.write(f"- Maximum: {basic_stats['max']:.2f} (in {basic_stats['max_year']})")
                    st.write(f"- Range: {basic_stats['range']:.2f}")
                    
                    st.write("**Dispersion:**")
                    st.write(f"- Standard deviation: {basic_stats['std_dev']:.2f}")
                    st.write(f"- Variance: {basic_stats['variance']:.2f}")
                    st.write(f"- IQR: {basic_stats['iqr']:.2f}")
                
                # Additional statistics in expandable section
                with st.expander("Show additional statistics"):
                    st.write(f"**Distribution Shape:**")
                    st.write(f"- Skewness: {basic_stats['skewness']:.4f}")
                    st.write(f"- Kurtosis: {basic_stats['kurtosis']:.4f}")
                    
                    st.write("**Percentiles:**")
                    st.write(f"- 25th percentile: {basic_stats['percentile_25']:.2f}")
                    st.write(f"- 75th percentile: {basic_stats['percentile_75']:.2f}")
                    
                    st.write("**First and Last Values:**")
                    st.write(f"- First value: {basic_stats['first_value']:.2f}")
                    st.write(f"- Last value: {basic_stats['last_value']:.2f}")
            
            if "Trend Analysis" in analysis_options:
                st.subheader("Trend Analysis")
                trend_results = TimeSeriesUtils.analyze_trend(ts_df, 'Value')
                
                # Display trend stats
                st.write(f"**Linear Trend: {trend_results['trend_direction']}**")
                st.write(f"- Slope: {trend_results['slope']:.4f} per year")
                st.write(f"- Intercept: {trend_results['intercept']:.4f}")
                st.write(f"- R-squared: {trend_results['r_squared']:.4f}")
                st.write(f"- p-value: {trend_results['p_value']:.4f}")
                
                # Plot with trend line
                df_with_trend = trend_results['df_with_trend']
                
                fig, ax = plt.subplots(figsize=(12, 6))
                ax.plot(df_with_trend.index, df_with_trend['Value'], 'b-', label='Original Data')
                ax.plot(df_with_trend.index, df_with_trend['trend_line'], 'r--', label='Linear Trend')
                ax.set_title(f'Time Series with Linear Trend: {selected_dataset}', fontsize=16)
                ax.set_xlabel('Year', fontsize=12)
                ax.set_ylabel('Value', fontsize=12)
                ax.legend()
                ax.grid(True)
                st.pyplot(fig)
            
            if "Growth Rates Analysis" in analysis_options:
                st.subheader("Growth Rates Analysis")
                growth_rates = TimeSeriesUtils.calculate_growth_rates(ts_df, 'Value')
                
                # Display summary growth statistics
                st.write(f"**Growth Summary:**")
                st.write(f"- Average annual growth rate: {growth_rates['avg_annual_growth_rate']:.2f}%")
                st.write(f"- Compound annual growth rate (CAGR): {growth_rates['cagr']:.2f}%")
                st.write(f"- Total growth over period: {growth_rates['total_growth']:.2f}%")
                
                # Plot yearly growth rates
                yearly_growth = pd.DataFrame(list(growth_rates['yearly_growth_rates'].items()), 
                                            columns=['Year', 'Growth Rate'])
                
                fig, ax = plt.subplots(figsize=(12, 6))
                ax.bar(yearly_growth['Year'], yearly_growth['Growth Rate'], color='steelblue')
                ax.axhline(y=0, color='r', linestyle='-', alpha=0.3)
                ax.set_title(f'Year-over-Year Growth Rate: {selected_dataset}', fontsize=16)
                ax.set_xlabel('Year', fontsize=12)
                ax.set_ylabel('Growth Rate (%)', fontsize=12)
                ax.grid(True)
                st.pyplot(fig)
            
            if "Stationarity Test" in analysis_options:
                st.subheader("Stationarity Test")
                stationarity_results = TimeSeriesUtils.check_stationarity(ts_df['Value'])
                
                # Display stationarity test results
                st.write(f"**Augmented Dickey-Fuller Test:**")
                st.write(f"- Test statistic: {stationarity_results['test_statistic']:.4f}")
                st.write(f"- p-value: {stationarity_results['p_value']:.4f}")
                st.write(f"- Is stationary: {'Yes' if stationarity_results['is_stationary'] else 'No'}")
                
                # Display critical values
                st.write("Critical Values:")
                for key, value in stationarity_results['critical_values'].items():
                    st.write(f"- {key}: {value:.4f}")
                
                # Interpretation
                if stationarity_results['is_stationary']:
                    st.success("The time series is stationary, meaning it doesn't have time-dependent structure. This is good for forecasting.")
                else:
                    st.warning("The time series is non-stationary. Differencing or transformation may be needed for accurate forecasting.")
                    
                    # Calculate and plot differenced series
                    st.subheader("First Difference of Time Series")
                    diff_series = ts_df['Value'].diff().dropna()
                    
                    # Test stationarity of differenced series
                    diff_stationarity = TimeSeriesUtils.check_stationarity(diff_series)
                    st.write(f"**First Difference Stationarity:**")
                    st.write(f"- Test statistic: {diff_stationarity['test_statistic']:.4f}")
                    st.write(f"- p-value: {diff_stationarity['p_value']:.4f}")
                    st.write(f"- Is stationary: {'Yes' if diff_stationarity['is_stationary'] else 'No'}")
                    
                    # Plot differenced series
                    fig, ax = plt.subplots(figsize=(12, 6))
                    ax.plot(diff_series.index, diff_series.values)
                    ax.set_title(f'Differenced Series: {selected_dataset}', fontsize=16)
                    ax.set_xlabel('Year', fontsize=12)
                    ax.set_ylabel('Differenced Value', fontsize=12)
                    ax.axhline(y=0, color='r', linestyle='-', alpha=0.3)
                    ax.grid(True)
                    st.pyplot(fig)
            
            if "Volatility Analysis" in analysis_options:
                st.subheader("Volatility Analysis")
                
                # Calculate volatility metrics
                mean_val = ts_df['Value'].mean()
                std_dev = ts_df['Value'].std()
                cv = std_dev / mean_val if mean_val != 0 else np.nan
                
                # Display volatility metrics
                st.write(f"**Volatility Metrics:**")
                st.write(f"- Coefficient of Variation: {cv:.4f}")
                st.write(f"- Standard Deviation: {std_dev:.2f}")
                
                # Calculate and plot rolling standard deviation
                if len(ts_df) >= 5:
                    window_size = min(5, len(ts_df) // 2)
                    rolling_std = ts_df['Value'].rolling(window=window_size).std()
                    
                    fig, ax = plt.subplots(figsize=(12, 6))
                    ax.plot(rolling_std.index, rolling_std.values)
                    ax.set_title(f'Rolling {window_size}-Year Standard Deviation: {selected_dataset}', fontsize=16)
                    ax.set_xlabel('Year', fontsize=12)
                    ax.set_ylabel('Standard Deviation', fontsize=12)
                    ax.grid(True)
                    st.pyplot(fig)
                else:
                    st.warning(f"Not enough data points for rolling standard deviation calculation. Need at least 5 data points.")
            
            if "Significant Changes" in analysis_options:
                st.subheader("Significant Changes Analysis")
                
                # Calculate year-over-year changes
                df_changes = ts_df.copy()
                df_changes['change'] = ts_df['Value'].diff()
                df_changes['pct_change'] = ts_df['Value'].pct_change() * 100
                
                # Find years with largest increases and decreases
                if not df_changes['change'].dropna().empty:
                    max_increase_idx = df_changes['change'].idxmax()
                    max_increase_val = df_changes.loc[max_increase_idx, 'change']
                    max_increase_pct = df_changes.loc[max_increase_idx, 'pct_change']
                    
                    min_increase_idx = df_changes['change'].idxmin()
                    min_increase_val = df_changes.loc[min_increase_idx, 'change']
                    min_increase_pct = df_changes.loc[min_increase_idx, 'pct_change']
                    
                    # Display significant changes
                    st.write(f"**Significant Changes:**")
                    st.write(f"- Largest increase: {max_increase_val:.2f} in {max_increase_idx} ({max_increase_pct:.2f}%)")
                    st.write(f"- Largest decrease: {min_increase_val:.2f} in {min_increase_idx} ({min_increase_pct:.2f}%)")
                    
                    # Plot changes
                    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
                    
                    # Absolute changes
                    ax1.bar(df_changes.index[1:], df_changes['change'].iloc[1:], color='steelblue')
                    ax1.axhline(y=0, color='r', linestyle='-', alpha=0.3)
                    ax1.set_title(f'Absolute Year-over-Year Changes: {selected_dataset}', fontsize=14)
                    ax1.set_ylabel('Change in Value', fontsize=12)
                    ax1.grid(True)
                    
                    # Percentage changes
                    ax2.bar(df_changes.index[1:], df_changes['pct_change'].iloc[1:], color='darkorange')
                    ax2.axhline(y=0, color='r', linestyle='-', alpha=0.3)
                    ax2.set_title(f'Percentage Year-over-Year Changes: {selected_dataset}', fontsize=14)
                    ax2.set_xlabel('Year', fontsize=12)
                    ax2.set_ylabel('Percentage Change (%)', fontsize=12)
                    ax2.grid(True)
                    
                    plt.tight_layout()
                    st.pyplot(fig)
                    
                    # Identify outlier years (changes greater than 1.5 standard deviations)
                    std_threshold = 1.5 * df_changes['change'].std()
                    significant_changes = df_changes[abs(df_changes['change']) > std_threshold]
                    
                    if not significant_changes.empty:
                        st.subheader("Outlier Years")
                        st.write("Years with changes greater than 1.5 standard deviations:")
                        
                        # Create a DataFrame for display
                        outliers_df = pd.DataFrame({
                            'Year': significant_changes.index,
                            'Value': significant_changes['Value'],
                            'Absolute Change': significant_changes['change'],
                            'Percentage Change (%)': significant_changes['pct_change']
                        })
                        
                        st.dataframe(outliers_df)
                
        except Exception as e:
            st.error(f"Error analyzing dataset: {e}")
            st.write("Please make sure your dataset has a proper structure with 'Year' and 'Value' columns.")