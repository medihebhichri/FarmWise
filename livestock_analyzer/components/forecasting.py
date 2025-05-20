import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import pickle
import json
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.stattools import adfuller
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, mean_absolute_percentage_error
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.optimizers import Adam

from utils.time_series_utils import TimeSeriesUtils

class ForecastingPage:
    """
    Class to handle the forecasting page of the Streamlit application
    """
    
    @staticmethod
    def render(selected_dataset):
        """
        Render the forecasting page
        
        Parameters:
        -----------
        selected_dataset : str
            Name of the selected dataset
        """
        st.title("Time Series Forecasting")
        
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
                st.warning("Dataset doesn't have expected 'Year' and 'Value' columns. Forecasting is not possible.")
                return
            
            # Preprocess data
            df['Year'] = pd.to_numeric(df['Year'], errors='coerce')
            df['Value'] = pd.to_numeric(df['Value'], errors='coerce')
            
            # Sort by year
            df = df.sort_values(by='Year')
            
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
            
            # Forecasting Options
            st.sidebar.header("Forecasting Options")
            
            # Model selection
            model_type = st.sidebar.selectbox(
                "Select forecasting model:",
                ["ARIMA", "ARIMA-ANN Hybrid"]
            )
            
            # Test size selection
            test_size = st.sidebar.slider(
                "Test set size (% of data):",
                min_value=10,
                max_value=30,
                value=20,
                step=5
            ) / 100
            
            # Forecast horizon
            forecast_horizon = st.sidebar.slider(
                "Forecast horizon (years):",
                min_value=1,
                max_value=10,
                value=5
            )
            
            # ARIMA parameters
            if model_type == "ARIMA":
                st.sidebar.subheader("ARIMA Parameters")
                
                p = st.sidebar.slider("p (Auto-Regressive)", 0, 2, 1)
                d = st.sidebar.slider("d (Differencing)", 0, 2, 1)
                q = st.sidebar.slider("q (Moving Average)", 0, 2, 1)
                
                arima_order = (p, d, q)
                
                st.sidebar.write(f"ARIMA Order: {arima_order}")
            
            # Hybrid model parameters
            if model_type == "ARIMA-ANN Hybrid":
                st.sidebar.subheader("ARIMA-ANN Parameters")
                
                p = st.sidebar.slider("ARIMA p", 0, 2, 0)
                d = st.sidebar.slider("ARIMA d", 0, 2, 1)
                q = st.sidebar.slider("ARIMA q", 0, 2, 1)
                
                arima_order = (p, d, q)
                
                st.sidebar.write(f"ARIMA Order: {arima_order}")
                
                # Neural network parameters
                st.sidebar.subheader("Neural Network Parameters")
                layers_str = st.sidebar.text_input("Hidden layers (comma-separated)", "16, 8")
                epochs = st.sidebar.slider("Training epochs", 50, 200, 100)
                
                try:
                    ann_layers = [int(layer.strip()) for layer in layers_str.split(",")]
                except ValueError:
                    st.sidebar.error("Invalid layer format. Using default [16, 8].")
                    ann_layers = [16, 8]
            
            # Generate forecast button
            if st.button("Generate Forecast"):
                st.info("Starting forecasting process... This may take a moment.")
                
                # Create directory for model outputs
                forecasting_dir = os.path.join("forecasting_data")
                if not os.path.exists(forecasting_dir):
                    os.makedirs(forecasting_dir)
                
                # Prepare data for forecasting
                dataset_name = os.path.splitext(selected_dataset)[0]
                
                # Check for stationarity
                ts_df = df[['Year', 'Value']].set_index('Year')
                stationarity_results = TimeSeriesUtils.check_stationarity(ts_df['Value'])
                
                st.write("**Stationarity Test:**")
                st.write(f"- ADF Test p-value: {stationarity_results['p_value']:.4f}")
                st.write(f"- Series is {'stationary' if stationarity_results['is_stationary'] else 'non-stationary'}")
                
                if model_type == "ARIMA":
                    # If series is non-stationary and differencing is 0, warn the user
                    if not stationarity_results['is_stationary'] and d == 0:
                        st.warning("The series is non-stationary. Consider using differencing (d > 0) for better results.")
                
                # Split data into training and testing sets
                n = len(df)
                train_size = int(n * (1 - test_size))
                
                train_data = df.iloc[:train_size]
                test_data = df.iloc[train_size:]
                
                st.write(f"**Data Split:**")
                st.write(f"- Training set: {len(train_data)} samples ({train_data['Year'].min()} - {train_data['Year'].max()})")
                st.write(f"- Testing set: {len(test_data)} samples ({test_data['Year'].min()} - {test_data['Year'].max()})")
                
                # Set index to year for time series modeling
                train_ts = train_data.set_index('Year')['Value']
                test_ts = test_data.set_index('Year')['Value']
                
                # Generate forecast based on selected model
                if model_type == "ARIMA":
                    with st.spinner("Training ARIMA model..."):
                        results = ForecastingPage._forecast_with_arima(
                            train_ts, 
                            test_ts, 
                            arima_order, 
                            forecast_horizon
                        )
                    
                    # Display results
                    ForecastingPage._display_arima_results(results, train_data, test_data, forecast_horizon)
                    
                elif model_type == "ARIMA-ANN Hybrid":
                    with st.spinner("Training ARIMA-ANN Hybrid model..."):
                        results = ForecastingPage._forecast_with_hybrid(
                            train_ts, 
                            test_ts, 
                            arima_order, 
                            ann_layers, 
                            epochs, 
                            forecast_horizon
                        )
                    
                    # Display results
                    ForecastingPage._display_hybrid_results(results, train_data, test_data, forecast_horizon)
        
        except Exception as e:
            st.error(f"Error during forecasting: {e}")
            st.write("Please make sure your dataset has a proper structure with 'Year' and 'Value' columns.")
    
    @staticmethod
    def _forecast_with_arima(train_data, test_data, order, forecast_horizon):
        """
        Forecast using ARIMA model
        
        Parameters:
        -----------
        train_data : Series
            Training time series data
        test_data : Series
            Testing time series data
        order : tuple
            ARIMA order (p, d, q)
        forecast_horizon : int
            Number of periods to forecast into the future
        
        Returns:
        --------
        dict
            Dictionary containing forecasting results
        """
        results = {}
        
        # Train ARIMA model
        model = ARIMA(train_data, order=order)
        fitted_model = model.fit()
        
        # In-sample predictions
        in_sample_predictions = fitted_model.predict(start=0, end=len(train_data)-1)
        
        # Test set forecasts
        test_forecasts = fitted_model.forecast(steps=len(test_data))
        
        # Future forecasts
        future_forecasts = fitted_model.forecast(steps=len(test_data) + forecast_horizon)[-forecast_horizon:]
        
        # Calculate error metrics
        # In-sample errors
        in_sample_rmse = np.sqrt(mean_squared_error(train_data, in_sample_predictions))
        in_sample_mae = mean_absolute_error(train_data, in_sample_predictions)
        in_sample_mape = mean_absolute_percentage_error(train_data, in_sample_predictions) * 100
        
        # Out-of-sample errors
        out_sample_rmse = np.sqrt(mean_squared_error(test_data, test_forecasts))
        out_sample_mae = mean_absolute_error(test_data, test_forecasts)
        out_sample_mape = mean_absolute_percentage_error(test_data, test_forecasts) * 100
        
        # Store results
        results['model'] = fitted_model
        results['in_sample_predictions'] = in_sample_predictions
        results['test_forecasts'] = test_forecasts
        results['future_forecasts'] = future_forecasts
        results['in_sample_rmse'] = in_sample_rmse
        results['in_sample_mae'] = in_sample_mae
        results['in_sample_mape'] = in_sample_mape
        results['out_sample_rmse'] = out_sample_rmse
        results['out_sample_mae'] = out_sample_mae
        results['out_sample_mape'] = out_sample_mape
        
        return results
    
    @staticmethod
    def _forecast_with_hybrid(train_data, test_data, arima_order, ann_layers, epochs, forecast_horizon):
        """
        Forecast using ARIMA-ANN hybrid model
        
        Parameters:
        -----------
        train_data : Series
            Training time series data
        test_data : Series
            Testing time series data
        arima_order : tuple
            ARIMA order (p, d, q)
        ann_layers : list
            List containing the number of neurons in each hidden layer
        epochs : int
            Number of training epochs
        forecast_horizon : int
            Number of periods to forecast into the future
        
        Returns:
        --------
        dict
            Dictionary containing forecasting results
        """
        results = {}
        
        # Step 1: Train ARIMA model
        model_arima = ARIMA(train_data, order=arima_order)
        fitted_arima = model_arima.fit()
        
        # Get ARIMA predictions for training and test sets
        arima_train_pred = fitted_arima.predict(start=0, end=len(train_data)-1)
        arima_test_pred = fitted_arima.forecast(steps=len(test_data))
        
        # Calculate residuals for training data
        train_residuals = train_data.values - arima_train_pred.values
        
        # Step 2: Train ANN on residuals
        # Create lagged features for residuals (using last 3 residuals to predict next)
        X_train, y_train = ForecastingPage._create_lagged_features(train_residuals, 3)
        
        # Normalize the data
        scaler_X = MinMaxScaler(feature_range=(-1, 1))
        scaler_y = MinMaxScaler(feature_range=(-1, 1))
        
        X_train_scaled = scaler_X.fit_transform(X_train)
        y_train_scaled = scaler_y.fit_transform(y_train.reshape(-1, 1)).flatten()
        
        # Build ANN model
        model_ann = Sequential()
        model_ann.add(Dense(ann_layers[0], activation='relu', input_dim=X_train.shape[1]))
        model_ann.add(Dropout(0.2))
        
        for units in ann_layers[1:]:
            model_ann.add(Dense(units, activation='relu'))
            model_ann.add(Dropout(0.2))
        
        model_ann.add(Dense(1))
        
        # Compile and train
        model_ann.compile(optimizer=Adam(learning_rate=0.001), loss='mse')
        
        early_stopping = EarlyStopping(
            monitor='val_loss',
            patience=10,
            restore_best_weights=True
        )
        
        history = model_ann.fit(
            X_train_scaled, y_train_scaled,
            epochs=epochs,
            batch_size=8,
            validation_split=0.2,
            callbacks=[early_stopping],
            verbose=0
        )
        
        # Step 3: Generate hybrid predictions for test set
        # First, we need the initial residuals from the training set to predict the first test residuals
        initial_residuals = train_residuals[-3:]
        
        test_residuals_pred = []
        
        # Iteratively predict each test residual
        for i in range(len(test_data)):
            if i == 0:
                # For the first prediction, use the last 3 residuals from training
                X_pred = np.array(initial_residuals).reshape(1, -1)
            else:
                # For subsequent predictions, use the last 3 predicted residuals
                start_idx = max(0, i-3)
                previous_residuals = test_residuals_pred[start_idx:i]
                
                # If we don't have enough previous residuals, pad with training residuals
                if len(previous_residuals) < 3:
                    n_needed = 3 - len(previous_residuals)
                    X_pred = np.array(list(initial_residuals[-n_needed:]) + previous_residuals).reshape(1, -1)
                else:
                    X_pred = np.array(previous_residuals).reshape(1, -1)
            
            # Scale the input
            X_pred_scaled = scaler_X.transform(X_pred)
            
            # Predict
            residual_pred_scaled = model_ann.predict(X_pred_scaled, verbose=0)
            
            # Inverse scale
            residual_pred = scaler_y.inverse_transform(residual_pred_scaled)[0][0]
            
            # Store the prediction
            test_residuals_pred.append(residual_pred)
        
        # Combine ARIMA predictions with predicted residuals for test set
        hybrid_test_pred = arima_test_pred.values + np.array(test_residuals_pred)
        
        # Step 4: Generate future forecasts
        # Get ARIMA forecasts for future periods
        arima_future = fitted_arima.forecast(steps=len(test_data) + forecast_horizon)[-forecast_horizon:]
        
        # Predict residuals for future periods
        future_residuals_pred = []
        
        # Use the last 3 predicted residuals from test set to start
        recent_residuals = test_residuals_pred[-3:]
        
        for i in range(forecast_horizon):
            X_pred = np.array(recent_residuals).reshape(1, -1)
            X_pred_scaled = scaler_X.transform(X_pred)
            
            residual_pred_scaled = model_ann.predict(X_pred_scaled, verbose=0)
            residual_pred = scaler_y.inverse_transform(residual_pred_scaled)[0][0]
            
            future_residuals_pred.append(residual_pred)
            
            # Update recent residuals for next iteration
            recent_residuals = recent_residuals[1:] + [residual_pred]
        
        # Combine ARIMA forecasts with predicted residuals for future periods
        hybrid_future = arima_future.values + np.array(future_residuals_pred)
        
        # Calculate error metrics
        # ARIMA test errors
        arima_rmse = np.sqrt(mean_squared_error(test_data, arima_test_pred))
        arima_mae = mean_absolute_error(test_data, arima_test_pred)
        arima_mape = mean_absolute_percentage_error(test_data, arima_test_pred) * 100
        
        # Hybrid test errors
        hybrid_rmse = np.sqrt(mean_squared_error(test_data, hybrid_test_pred))
        hybrid_mae = mean_absolute_error(test_data, hybrid_test_pred)
        hybrid_mape = mean_absolute_percentage_error(test_data, hybrid_test_pred) * 100
        
        # Store results
        results['arima_model'] = fitted_arima
        results['ann_model'] = model_ann
        results['arima_train_pred'] = arima_train_pred
        results['arima_test_pred'] = arima_test_pred
        results['hybrid_test_pred'] = hybrid_test_pred
        results['arima_future'] = arima_future
        results['hybrid_future'] = hybrid_future
        results['arima_rmse'] = arima_rmse
        results['arima_mae'] = arima_mae
        results['arima_mape'] = arima_mape
        results['hybrid_rmse'] = hybrid_rmse
        results['hybrid_mae'] = hybrid_mae
        results['hybrid_mape'] = hybrid_mape
        results['improvement'] = ((arima_rmse - hybrid_rmse) / arima_rmse) * 100
        results['training_history'] = history.history
        
        return results
    
    @staticmethod
    def _create_lagged_features(series, lag=3):
        """Create lagged features for time series data"""
        X, y = [], []
        for i in range(lag, len(series)):
            X.append(series[i-lag:i])
            y.append(series[i])
        return np.array(X), np.array(y)
    
    @staticmethod
    def _display_arima_results(results, train_data, test_data, forecast_horizon):
        """
        Display ARIMA forecasting results
        
        Parameters:
        -----------
        results : dict
            Dictionary containing forecasting results
        train_data : DataFrame
            Training data with Year and Value columns
        test_data : DataFrame
            Testing data with Year and Value columns
        forecast_horizon : int
            Number of periods forecast into the future
        """
        st.subheader("ARIMA Model Results")
        
        # Display model summary
        with st.expander("ARIMA Model Summary"):
            st.text(results['model'].summary())
        
        # Display performance metrics
        st.write("**Performance Metrics:**")
        
        metrics_df = pd.DataFrame({
            'Metric': ['RMSE', 'MAE', 'MAPE (%)'],
            'Training Set': [results['in_sample_rmse'], results['in_sample_mae'], results['in_sample_mape']],
            'Test Set': [results['out_sample_rmse'], results['out_sample_mae'], results['out_sample_mape']]
        })
        
        st.dataframe(metrics_df.style.format({
            'Training Set': '{:.2f}',
            'Test Set': '{:.2f}'
        }))
        
        # Plot results
        st.subheader("Forecast Visualization")
        
        # Create future years
        last_year = test_data['Year'].max()
        future_years = np.arange(last_year + 1, last_year + forecast_horizon + 1)
        
        # Confidence intervals for test predictions
        test_ci_lower = results['test_forecasts'] - 1.96 * results['out_sample_rmse']
        test_ci_upper = results['test_forecasts'] + 1.96 * results['out_sample_rmse']
        
        # Confidence intervals for future predictions
        future_ci_lower = results['future_forecasts'] - 1.96 * results['out_sample_rmse']
        future_ci_upper = results['future_forecasts'] + 1.96 * results['out_sample_rmse']
        
        # Plot
        fig, ax = plt.subplots(figsize=(12, 6))
        
        # Plot training data
        ax.plot(train_data['Year'], train_data['Value'], 'b-', label='Training Data')
        
        # Plot in-sample predictions
        ax.plot(train_data['Year'], results['in_sample_predictions'], 'g--', alpha=0.7, label='In-Sample Predictions')
        
        # Plot test data and forecasts
        ax.plot(test_data['Year'], test_data['Value'], 'r-', label='Test Data')
        ax.plot(test_data['Year'], results['test_forecasts'], 'm--', label='Test Forecasts')
        
        # Plot future forecasts
        ax.plot(future_years, results['future_forecasts'], 'c--', label='Future Forecasts')
        
        # Add confidence intervals
        ax.fill_between(test_data['Year'], test_ci_lower, test_ci_upper, color='m', alpha=0.2)
        ax.fill_between(future_years, future_ci_lower, future_ci_upper, color='c', alpha=0.2, label='95% CI')
        
        ax.set_title('ARIMA Forecasting Results', fontsize=16)
        ax.set_xlabel('Year', fontsize=12)
        ax.set_ylabel('Value', fontsize=12)
        ax.legend()
        ax.grid(True)
        
        st.pyplot(fig)
        
        # Display forecast data
        st.subheader("Forecast Data")
        
        # Create forecast DataFrame
        forecast_df = pd.DataFrame({
            'Year': np.concatenate([test_data['Year'].values, future_years]),
            'Actual': np.concatenate([test_data['Value'].values, [None] * forecast_horizon]),
            'Forecast': np.concatenate([results['test_forecasts'], results['future_forecasts']]),
            'Lower_CI': np.concatenate([test_ci_lower, future_ci_lower]),
            'Upper_CI': np.concatenate([test_ci_upper, future_ci_upper])
        })
        
        # Fix the formatting to handle None values properly
        st.dataframe(forecast_df.style.format({
            'Actual': lambda x: '{:.2f}'.format(x) if pd.notnull(x) else '',
            'Forecast': lambda x: '{:.2f}'.format(x) if pd.notnull(x) else '',
            'Lower_CI': lambda x: '{:.2f}'.format(x) if pd.notnull(x) else '',
            'Upper_CI': lambda x: '{:.2f}'.format(x) if pd.notnull(x) else ''
        }))
        
        # Download link for forecast data
        csv = forecast_df.to_csv(index=False)
        st.download_button(
            label="Download Forecast Data",
            data=csv,
            file_name="arima_forecast.csv",
            mime="text/csv"
        )
    
    @staticmethod
    def _display_hybrid_results(results, train_data, test_data, forecast_horizon):
        """
        Display ARIMA-ANN hybrid forecasting results
        
        Parameters:
        -----------
        results : dict
            Dictionary containing forecasting results
        train_data : DataFrame
            Training data with Year and Value columns
        test_data : DataFrame
            Testing data with Year and Value columns
        forecast_horizon : int
            Number of periods forecast into the future
        """
        st.subheader("ARIMA-ANN Hybrid Model Results")
        
        # Display model summary
        with st.expander("ARIMA Component Summary"):
            st.text(results['arima_model'].summary())
        
        # Display neural network training history
        st.subheader("Neural Network Training")
        
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.plot(results['training_history']['loss'], label='Training Loss')
        ax.plot(results['training_history']['val_loss'], label='Validation Loss')
        ax.set_title('ANN Model Training History', fontsize=16)
        ax.set_xlabel('Epoch', fontsize=12)
        ax.set_ylabel('Loss (MSE)', fontsize=12)
        ax.legend()
        ax.grid(True)
        
        st.pyplot(fig)
        
        # Display performance metrics
        st.subheader("Performance Metrics")
        
        # Comparison of ARIMA vs Hybrid
        metrics_df = pd.DataFrame({
            'Metric': ['RMSE', 'MAE', 'MAPE (%)'],
            'ARIMA Model': [results['arima_rmse'], results['arima_mae'], results['arima_mape']],
            'Hybrid Model': [results['hybrid_rmse'], results['hybrid_mae'], results['hybrid_mape']]
        })
        
        st.dataframe(metrics_df.style.format({
            'ARIMA Model': '{:.2f}',
            'Hybrid Model': '{:.2f}'
        }))
        
        # Display improvement
        if results['improvement'] > 0:
            st.success(f"The hybrid model improved forecast accuracy by {results['improvement']:.2f}% compared to ARIMA alone.")
        else:
            st.warning(f"The hybrid model did not improve forecast accuracy (change: {results['improvement']:.2f}%).")
        
        # Plot results
        st.subheader("Forecast Visualization")
        
        # Create future years
        last_year = test_data['Year'].max()
        future_years = np.arange(last_year + 1, last_year + forecast_horizon + 1)
        
        # Confidence intervals for hybrid predictions
        hybrid_ci_lower = results['hybrid_test_pred'] - 1.96 * results['hybrid_rmse']
        hybrid_ci_upper = results['hybrid_test_pred'] + 1.96 * results['hybrid_rmse']
        
        # Confidence intervals for future predictions
        future_ci_lower = results['hybrid_future'] - 1.96 * results['hybrid_rmse']
        future_ci_upper = results['hybrid_future'] + 1.96 * results['hybrid_rmse']
        
        # Plot
        fig, ax = plt.subplots(figsize=(12, 6))
        
        # Plot training data
        ax.plot(train_data['Year'], train_data['Value'], 'b-', label='Training Data')
        
        # Plot test data
        ax.plot(test_data['Year'], test_data['Value'], 'r-', label='Test Data')
        
        # Plot ARIMA and hybrid test forecasts
        ax.plot(test_data['Year'], results['arima_test_pred'], 'g--', alpha=0.7, label='ARIMA Forecasts')
        ax.plot(test_data['Year'], results['hybrid_test_pred'], 'm--', label='Hybrid Forecasts')
        
        # Plot future forecasts
        ax.plot(future_years, results['hybrid_future'], 'c--', label='Future Forecasts')
        
        # Add confidence intervals
        ax.fill_between(future_years, future_ci_lower, future_ci_upper, color='c', alpha=0.2, label='95% CI')
        
        ax.set_title('ARIMA-ANN Hybrid Forecasting Results', fontsize=16)
        ax.set_xlabel('Year', fontsize=12)
        ax.set_ylabel('Value', fontsize=12)
        ax.legend()
        ax.grid(True)
        
        st.pyplot(fig)
        
        # Display forecast data
        st.subheader("Forecast Data")
        
        # Create forecast DataFrame
        forecast_df = pd.DataFrame({
            'Year': np.concatenate([test_data['Year'].values, future_years]),
            'Actual': np.concatenate([test_data['Value'].values, [None] * forecast_horizon]),
            'ARIMA_Forecast': np.concatenate([results['arima_test_pred'], results['arima_future']]),
            'Hybrid_Forecast': np.concatenate([results['hybrid_test_pred'], results['hybrid_future']]),
            'Lower_CI': np.concatenate([hybrid_ci_lower, future_ci_lower]),
            'Upper_CI': np.concatenate([hybrid_ci_upper, future_ci_upper])
        })
        
        # Fix the formatting to handle None values properly
        st.dataframe(forecast_df.style.format({
            'Actual': lambda x: '{:.2f}'.format(x) if pd.notnull(x) else '',
            'ARIMA_Forecast': lambda x: '{:.2f}'.format(x) if pd.notnull(x) else '',
            'Hybrid_Forecast': lambda x: '{:.2f}'.format(x) if pd.notnull(x) else '',
            'Lower_CI': lambda x: '{:.2f}'.format(x) if pd.notnull(x) else '',
            'Upper_CI': lambda x: '{:.2f}'.format(x) if pd.notnull(x) else ''
        }))
        
        # Download link for forecast data
        csv = forecast_df.to_csv(index=False)
        st.download_button(
            label="Download Forecast Data",
            data=csv,
            file_name="hybrid_forecast.csv",
            mime="text/csv"
        )