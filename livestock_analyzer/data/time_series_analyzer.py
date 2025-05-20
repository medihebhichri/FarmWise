import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from scipy import stats
import json

from ..utils.time_series_utils import TimeSeriesUtils

class TimeSeriesAnalyzer:
    """
    Class to analyze time series data for livestock metrics
    """
    
    def __init__(self, output_dir="analysis_results"):
        """
        Initialize the TimeSeriesAnalyzer
        
        Parameters:
        -----------
        output_dir : str, default="analysis_results"
            Directory to save analysis outputs
        """
        self.output_dir = output_dir
        self.results = {}
        
        # Create output directory if it doesn't exist
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
    
    def load_data(self, file_path, separator=',', header=0, year_col='Year', value_col='Value',
                start_year=None, end_year=None):
        """
        Load time series data from a CSV file
        
        Parameters:
        -----------
        file_path : str
            Path to the CSV file containing time series data
        separator : str, default=','
            Delimiter used in the CSV file
        header : int, default=0
            Row to use as column names
        year_col : str, default='Year'
            Name of the column containing year values
        value_col : str, default='Value'
            Name of the column containing numeric values to analyze
        start_year : int, optional
            Starting year for analysis (if None, uses earliest year in data)
        end_year : int, optional
            Ending year for analysis (if None, uses latest year in data)
            
        Returns:
        --------
        pandas.DataFrame
            Time series data with year as index
        """
        # Extract filename for reporting
        self.file_name = os.path.basename(file_path)
        self.series_name = os.path.splitext(self.file_name)[0]
        self.dataset_dir = os.path.join(self.output_dir, self.series_name)
        
        # Create dataset-specific directory
        if not os.path.exists(self.dataset_dir):
            os.makedirs(self.dataset_dir)
        
        print(f"\nLoading data from {file_path}...")
        
        # Handle different file formats
        if 'goats_number.csv' in file_path or 'meat_of' in file_path or 'raw_milk' in file_path or 'sheep_numbers.csv' in file_path:
            # Multi-column format with Domain, Area, etc.
            df = pd.read_csv(file_path, sep=separator)
            if 'Unit' in df.columns:
                self.unit = df['Unit'].iloc[0] if not df['Unit'].empty else 'Unknown'
            else:
                self.unit = 'Unknown'
                
            # Extract year and value columns
            df[year_col] = pd.to_numeric(df['Year'], errors='coerce')
            df[value_col] = pd.to_numeric(df['Value'], errors='coerce')
            
        else:
            # Standard format with Year, Value columns
            df = pd.read_csv(file_path, sep=separator, header=header)
            self.unit = 'Count' if 'numbers' in file_path.lower() else 'Unknown'
        
        # Filter by year range if specified
        if start_year is not None:
            df = df[df[year_col] >= start_year]
        if end_year is not None:
            df = df[df[year_col] <= end_year]
        
        # Ensure data is sorted by year
        df = df.sort_values(by=year_col)
        
        # Set index to year for time series analysis
        self.df = df
        self.ts_df = df[[year_col, value_col]].set_index(year_col)
        self.year_col = year_col
        self.value_col = value_col
        
        return self.ts_df
    
    def analyze(self):
        """
        Perform comprehensive time series analysis on the loaded dataset
        
        Returns:
        --------
        dict
            Dictionary containing analysis results
        """
        print(f"\n{'='*80}")
        print(f"ANALYZING: {self.series_name}")
        print(f"{'='*80}\n")
        
        # Basic statistics
        self.basic_stats = TimeSeriesUtils.calculate_basic_stats(self.ts_df, self.value_col)
        print("\nBASIC STATISTICS:")
        for stat, value in self.basic_stats.items():
            print(f"{stat}: {value}")
        
        # Trend analysis
        self.trend_results = TimeSeriesUtils.analyze_trend(self.ts_df, self.value_col)
        print("\nTREND ANALYSIS:")
        print(f"Linear trend slope: {self.trend_results['slope']:.4f} per year")
        print(f"Linear trend p-value: {self.trend_results['p_value']:.4f}")
        print(f"Trend direction: {self.trend_results['trend_direction']}")
        
        # Growth rates analysis
        self.growth_rates = TimeSeriesUtils.calculate_growth_rates(self.ts_df, self.value_col)
        print("\nGROWTH RATES ANALYSIS:")
        print(f"Average annual growth rate: {self.growth_rates['avg_annual_growth_rate']:.2f}%")
        print(f"Compound annual growth rate (CAGR): {self.growth_rates['cagr']:.2f}%")
        print(f"Total growth over period: {self.growth_rates['total_growth']:.2f}%")
        
        # Stationarity test
        self.stationarity = TimeSeriesUtils.check_stationarity(self.ts_df[self.value_col])
        print("\nSTATIONARITY TEST (Augmented Dickey-Fuller):")
        print(f"Test statistic: {self.stationarity['test_statistic']:.4f}")
        print(f"p-value: {self.stationarity['p_value']:.4f}")
        print(f"Is stationary: {self.stationarity['is_stationary']}")
        
        # Volatility analysis
        self.volatility = self._analyze_volatility()
        print("\nVOLATILITY ANALYSIS:")
        print(f"Coefficient of variation: {self.volatility['cv']:.4f}")
        print(f"Rolling 5-year std dev (end of period): {self.volatility['rolling_std_end']:.4f}")
        
        # Identify significant changes
        self.changes = self._identify_significant_changes()
        print("\nSIGNIFICANT CHANGES:")
        if self.changes['largest_increase']['year'] is not None:
            print(f"Largest increase: {self.changes['largest_increase']['value']:.2f} in {self.changes['largest_increase']['year']} ({self.changes['largest_increase']['percent']:.2f}%)")
        if self.changes['largest_decrease']['year'] is not None:
            print(f"Largest decrease: {self.changes['largest_decrease']['value']:.2f} in {self.changes['largest_decrease']['year']} ({self.changes['largest_decrease']['percent']:.2f}%)")
        
        # Periods analysis
        self.periods = self._analyze_periods()
        print("\nPERIODS ANALYSIS:")
        print(f"Period with highest values: {self.periods['highest_period']['start_year']}-{self.periods['highest_period']['end_year']}")
        print(f"Period with lowest values: {self.periods['lowest_period']['start_year']}-{self.periods['lowest_period']['end_year']}")
        
        # Create visualizations
        print("\nGenerating visualizations...")
        self._generate_visualizations()
        
        # Save results to JSON
        self._save_results()
        
        print(f"\nAnalysis complete for {self.series_name}!")
        
        return self.results
    
    def _analyze_volatility(self):
        """Analyze the volatility of the time series"""
        results = {}
        
        # Coefficient of variation
        results['cv'] = self.ts_df[self.value_col].std() / self.ts_df[self.value_col].mean() if self.ts_df[self.value_col].mean() != 0 else np.nan
        
        # Rolling standard deviation (5-year window)
        if len(self.ts_df) >= 5:
            rolling_std = self.ts_df[self.value_col].rolling(window=5).std()
            results['rolling_std'] = rolling_std.dropna().to_dict()
            results['rolling_std_end'] = rolling_std.iloc[-1] if not rolling_std.empty else np.nan
        else:
            results['rolling_std'] = {}
            results['rolling_std_end'] = np.nan
        
        return results
    
    def _identify_significant_changes(self):
        """Identify years with significant changes in the time series"""
        results = {
            'largest_increase': {'year': None, 'value': 0, 'percent': 0},
            'largest_decrease': {'year': None, 'value': 0, 'percent': 0},
            'significant_years': []
        }
        
        # Calculate year-over-year changes
        df_changes = self.ts_df.copy()
        df_changes['change'] = self.ts_df[self.value_col].diff()
        df_changes['pct_change'] = self.ts_df[self.value_col].pct_change() * 100
        
        # Find largest increase and decrease
        if not df_changes['change'].dropna().empty:
            max_increase_idx = df_changes['change'].idxmax()
            max_increase_val = df_changes.loc[max_increase_idx, 'change']
            max_increase_pct = df_changes.loc[max_increase_idx, 'pct_change']
            
            min_increase_idx = df_changes['change'].idxmin()
            min_increase_val = df_changes.loc[min_increase_idx, 'change']
            min_increase_pct = df_changes.loc[min_increase_idx, 'pct_change']
            
            results['largest_increase'] = {
                'year': max_increase_idx,
                'value': max_increase_val,
                'percent': max_increase_pct
            }
            
            results['largest_decrease'] = {
                'year': min_increase_idx,
                'value': min_increase_val,
                'percent': min_increase_pct
            }
        
        # Identify years with changes greater than 1.5 standard deviations
        std_threshold = 1.5 * df_changes['change'].std()
        significant_changes = df_changes[abs(df_changes['change']) > std_threshold]
        
        for year, row in significant_changes.iterrows():
            results['significant_years'].append({
                'year': year,
                'value': row[self.value_col],
                'change': row['change'],
                'percent_change': row['pct_change']
            })
        
        return results
    
    def _analyze_periods(self):
        """Analyze different periods within the time series"""
        results = {
            'highest_period': {'start_year': None, 'end_year': None, 'avg_value': 0},
            'lowest_period': {'start_year': None, 'end_year': None, 'avg_value': float('inf')},
            'decade_averages': {}
        }
        
        # Calculate 5-year rolling averages if we have enough data
        if len(self.ts_df) >= 5:
            rolling_avg = self.ts_df[self.value_col].rolling(window=5).mean()
            
            # Find highest 5-year period
            if not rolling_avg.empty:
                max_period_end_idx = rolling_avg.idxmax()
                max_period_start_idx = max_period_end_idx - 4 if max_period_end_idx - 4 >= self.ts_df.index.min() else self.ts_df.index.min()
                
                results['highest_period'] = {
                    'start_year': max_period_start_idx,
                    'end_year': max_period_end_idx,
                    'avg_value': rolling_avg.loc[max_period_end_idx]
                }
                
                # Find lowest 5-year period
                min_period_end_idx = rolling_avg.idxmin()
                min_period_start_idx = min_period_end_idx - 4 if min_period_end_idx - 4 >= self.ts_df.index.min() else self.ts_df.index.min()
                
                results['lowest_period'] = {
                    'start_year': min_period_start_idx,
                    'end_year': min_period_end_idx,
                    'avg_value': rolling_avg.loc[min_period_end_idx]
                }
        
        return results
    
    def _generate_visualizations(self):
        """Generate visualizations for the time series analysis"""
        # Set style
        sns.set(style="whitegrid")
        
        # 1. Basic time series plot with trend line
        plt.figure(figsize=(12, 6))
        plt.plot(self.ts_df.index, self.ts_df[self.value_col], 'b-', label='Actual')
        
        if 'df_with_trend' in self.trend_results:
            plt.plot(self.ts_df.index, self.trend_results['df_with_trend']['trend_line'], 'r--', label='Trend')
        
        plt.title(f'{self.series_name} Time Series', fontsize=16)
        plt.xlabel('Year', fontsize=12)
        plt.ylabel(f'{self.value_col} ({self.unit})', fontsize=12)
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(os.path.join(self.dataset_dir, '1_time_series.png'), dpi=300)
        plt.close()
        
        # 2. Growth rates visualization
        if 'yearly_growth_rates' in self.growth_rates:
            years = list(self.growth_rates['yearly_growth_rates'].keys())
            growth_rates = list(self.growth_rates['yearly_growth_rates'].values())
            
            plt.figure(figsize=(12, 6))
            plt.bar(years, growth_rates, color='steelblue')
            plt.axhline(y=0, color='r', linestyle='-', alpha=0.3)
            plt.title(f'{self.series_name} Annual Growth Rates', fontsize=16)
            plt.xlabel('Year', fontsize=12)
            plt.ylabel('Percentage Change (%)', fontsize=12)
            plt.grid(True)
            plt.tight_layout()
            plt.savefig(os.path.join(self.dataset_dir, '2_growth_rates.png'), dpi=300)
            plt.close()
        
        # 3. ACF and PACF plots
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
        plot_acf(self.ts_df[self.value_col].dropna(), lags=20, ax=ax1)
        ax1.set_title(f'Autocorrelation: {self.series_name}', fontsize=14)
        
        plot_pacf(self.ts_df[self.value_col].dropna(), lags=20, ax=ax2)
        ax2.set_title(f'Partial Autocorrelation: {self.series_name}', fontsize=14)
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.dataset_dir, '3_acf_pacf.png'), dpi=300)
        plt.close()
        
        # 4. Rolling statistics
        plt.figure(figsize=(12, 6))
        
        # Plot rolling mean and standard deviation (5-year window)
        rolling_mean = self.ts_df[self.value_col].rolling(window=5).mean()
        rolling_std = self.ts_df[self.value_col].rolling(window=5).std()
        
        plt.plot(self.ts_df.index, self.ts_df[self.value_col], 'b-', label='Original')
        plt.plot(rolling_mean.index, rolling_mean, 'g-', label='Rolling Mean (5-year)')
        plt.plot(rolling_std.index, rolling_std, 'r-', label='Rolling Std (5-year)')
        
        plt.title(f'{self.series_name} Rolling Statistics', fontsize=16)
        plt.xlabel('Year', fontsize=12)
        plt.ylabel(f'{self.value_col} ({self.unit})', fontsize=12)
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(os.path.join(self.dataset_dir, '4_rolling_stats.png'), dpi=300)
        plt.close()
        
        # 5. Seasonal decomposition if we have enough data
        if len(self.ts_df) >= 6:  # Need at least twice the period length
            try:
                decomposition = seasonal_decompose(self.ts_df[self.value_col], model='additive', period=3)
                
                plt.figure(figsize=(12, 10))
                plt.subplot(411)
                plt.plot(decomposition.observed)
                plt.title('Observed', fontsize=14)
                plt.subplot(412)
                plt.plot(decomposition.trend)
                plt.title('Trend', fontsize=14)
                plt.subplot(413)
                plt.plot(decomposition.seasonal)
                plt.title('Seasonality', fontsize=14)
                plt.subplot(414)
                plt.plot(decomposition.resid)
                plt.title('Residuals', fontsize=14)
                plt.tight_layout()
                plt.savefig(os.path.join(self.dataset_dir, '5_seasonal_decomposition.png'), dpi=300)
                plt.close()
            except:
                print("Could not perform seasonal decomposition - insufficient data or other issue.")
    
    def _save_results(self):
        """Save analysis results to JSON file"""
        # Combine all results
        self.results = {
            'series_name': self.series_name,
            'basic_stats': self.basic_stats,
            'trend_analysis': {k: v for k, v in self.trend_results.items() if k != 'df_with_trend'},
            'growth_rates': self.growth_rates,
            'stationarity': self.stationarity,
            'volatility': self.volatility,
            'significant_changes': self.changes,
            'periods_analysis': self.periods
        }
        
        # Convert any NumPy values to Python native types
        self.results = self._convert_to_serializable(self.results)
        
        # Save to JSON
        with open(os.path.join(self.dataset_dir, 'analysis_results.json'), 'w') as f:
            json.dump(self.results, f, indent=4)
    
    def _convert_to_serializable(self, obj):
        """Convert NumPy values to Python native types for JSON serialization"""
        if isinstance(obj, dict):
            return {k: self._convert_to_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [self._convert_to_serializable(v) for v in obj]
        elif isinstance(obj, (np.int64, np.int32, np.int16, np.int8)):
            return int(obj)
        elif isinstance(obj, (np.float64, np.float32, np.float16)):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return self._convert_to_serializable(obj.tolist())
        elif isinstance(obj, np.bool_):
            return bool(obj)
        else:
            return obj
