import pandas as pd
import numpy as np
from scipy import stats
from statsmodels.tsa.stattools import adfuller

class TimeSeriesUtils:
    """Utility class with common time series analysis methods"""
    
    @staticmethod
    def check_stationarity(series):
        """Check for stationarity using Augmented Dickey-Fuller test"""
        results = {}
        
        # ADF test
        adf_result = adfuller(series.dropna())
        
        results['test_statistic'] = adf_result[0]
        results['p_value'] = adf_result[1]
        results['critical_values'] = adf_result[4]
        results['is_stationary'] = results['p_value'] < 0.05
        
        return results
    
    @staticmethod
    def calculate_basic_stats(df, value_col='Value'):
        """Calculate basic descriptive statistics"""
        stats_dict = {}
        stats_dict['count'] = len(df)
        stats_dict['start_year'] = df.index.min()
        stats_dict['end_year'] = df.index.max()
        stats_dict['time_span'] = stats_dict['end_year'] - stats_dict['start_year'] + 1
        stats_dict['min'] = df[value_col].min()
        stats_dict['min_year'] = df[value_col].idxmin()
        stats_dict['max'] = df[value_col].max()
        stats_dict['max_year'] = df[value_col].idxmax()
        stats_dict['mean'] = df[value_col].mean()
        stats_dict['median'] = df[value_col].median()
        stats_dict['std_dev'] = df[value_col].std()
        stats_dict['variance'] = df[value_col].var()
        stats_dict['skewness'] = df[value_col].skew()
        stats_dict['kurtosis'] = df[value_col].kurtosis()
        stats_dict['first_value'] = df[value_col].iloc[0]
        stats_dict['last_value'] = df[value_col].iloc[-1]
        stats_dict['range'] = stats_dict['max'] - stats_dict['min']
        
        # Calculate percentiles
        stats_dict['percentile_25'] = df[value_col].quantile(0.25)
        stats_dict['percentile_75'] = df[value_col].quantile(0.75)
        stats_dict['iqr'] = stats_dict['percentile_75'] - stats_dict['percentile_25']
        
        return stats_dict
    
    @staticmethod
    def analyze_trend(df, value_col='Value'):
        """Analyze the trend in the time series"""
        results = {}
        
        # Simple linear regression for trend
        X = np.array(range(len(df))).reshape(-1, 1)
        y = df[value_col].values
        
        slope, intercept, r_value, p_value, std_err = stats.linregress(range(len(df)), y)
        
        results['slope'] = slope
        results['intercept'] = intercept
        results['r_squared'] = r_value ** 2
        results['p_value'] = p_value
        results['std_err'] = std_err
        
        # Determine trend direction
        if p_value < 0.05:
            if slope > 0:
                results['trend_direction'] = "Significant upward trend"
            else:
                results['trend_direction'] = "Significant downward trend"
        else:
            results['trend_direction'] = "No significant trend"
            
        # Calculate the linear prediction for each year
        df_with_trend = df.copy()
        df_with_trend['trend_line'] = intercept + slope * np.array(range(len(df)))
        results['df_with_trend'] = df_with_trend
        
        return results
    
    @staticmethod
    def calculate_growth_rates(df, value_col='Value'):
        """Calculate various growth rates for the time series"""
        results = {}
        
        # Calculate year-over-year percent changes
        df_growth = df.copy()
        df_growth['pct_change'] = df[value_col].pct_change() * 100
        results['yearly_growth_rates'] = df_growth['pct_change'].dropna().to_dict()
        
        # Average annual growth rate
        results['avg_annual_growth_rate'] = df_growth['pct_change'].mean()
        
        # Compound annual growth rate (CAGR)
        first_val = df[value_col].iloc[0]
        last_val = df[value_col].iloc[-1]
        n_years = len(df) - 1
        
        if first_val > 0 and n_years > 0:
            cagr = (((last_val / first_val) ** (1 / n_years)) - 1) * 100
            results['cagr'] = cagr
        else:
            results['cagr'] = np.nan
        
        # Total growth over the period
        if first_val > 0:
            results['total_growth'] = ((last_val - first_val) / first_val) * 100
        else:
            results['total_growth'] = np.nan
        
        return results
    
    @staticmethod
    def create_sequences(data, seq_length):
        """Create sequences for machine learning models"""
        X, y = [], []
        for i in range(len(data) - seq_length):
            X.append(data[i:i+seq_length])
            y.append(data[i+seq_length])
        return np.array(X), np.array(y)
