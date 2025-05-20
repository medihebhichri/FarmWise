import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import json
import pickle
from statsmodels.tsa.arima.model import ARIMA
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, mean_absolute_percentage_error
from tensorflow.keras.models import Sequential, load_model, save_model
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.optimizers import Adam

from ..utils.time_series_utils import TimeSeriesUtils

class TimeSeriesForecaster:
    """
    Class to perform time series forecasting using ARIMA and hybrid ARIMA-ANN models
    """
    
    def __init__(self, output_dir="forecasting_results"):
        """
        Initialize the TimeSeriesForecaster
        
        Parameters:
        -----------
        output_dir : str, default="forecasting_results"
            Directory to save forecasting outputs
        """
        self.output_dir = output_dir
        
        # Create output directory if it doesn't exist
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
    
    def prepare_data(self, file_path, test_size=0.2, separator=','):
        """
        Prepare time series data for forecasting
        
        Parameters:
        -----------
        file_path : str
            Path to the CSV file containing time series data
        test_size : float, default=0.2
            Proportion of data to use for testing
        separator : str, default=','
            Delimiter used in the CSV file
            
        Returns:
        --------
        dict
            Dictionary containing prepared datasets
        """
        # Extract filename for reporting
        self.file_name = os.path.basename(file_path)
        self.series_name = os.path.splitext(self.file_name)[0]
        self.dataset_dir = os.path.join(self.output_dir, self.series_name)
        
        # Create dataset-specific directory
        if not os.path.exists(self.dataset_dir):
            os.makedirs(self.dataset_dir)
        
        print(f"\n{'='*80}")
        print(f"PREPARING DATA FOR FORECASTING: {self.series_name}")
        print(f"{'='*80}\n")
        
        # Load the data
        print(f"Loading data from {file_path}...")
        
        # Handle different file formats
        if 'goats_number.csv' in file_path or 'meat_of' in file_path or 'raw_milk' in file_path or 'sheep_numbers.csv' in file_path:
            # Multi-column format with Domain, Area, etc.
            df = pd.read_csv(file_path, sep=separator)
            
            # Extract year and value columns
            df['Year'] = pd.to_numeric(df['Year'], errors='coerce')
            df['Value'] = pd.to_numeric(df['Value'], errors='coerce')
            
        else:
            # Standard format with Year, Value columns
            df = pd.read_csv(file_path, sep=separator)
        
        # Ensure data is sorted by year
        df = df.sort_values(by='Year')
        
        # Create time series data frame
        self.df = df
        self.ts_df = df[['Year', 'Value']].set_index('Year')
        
        # Check for stationarity
        stationarity_results = TimeSeriesUtils.check_stationarity(self.ts_df['Value'])
        print("\nSTATIONARITY TEST (Augmented Dickey-Fuller):")
        print(f"Test statistic: {stationarity_results['test_statistic']:.4f}")
        print(f"p-value: {stationarity_results['p_value']:.4f}")
        print(f"Is stationary: {stationarity_results['is_stationary']}")
        
        # Split data into training and testing sets
        train_size = int(len(self.ts_df) * (1 - test_size))
        self.train_data = self.ts_df.iloc[:train_size]
        self.test_data = self.ts_df.iloc[train_size:]
        
        print(f"\nData split: {len(self.train_data)} training samples, {len(self.test_data)} testing samples")
        
        # Save the full, training, and testing datasets
        self.ts_df.to_csv(os.path.join(self.dataset_dir, f"{self.series_name}_full.csv"))
        self.train_data.to_csv(os.path.join(self.dataset_dir, f"{self.series_name}_train.csv"))
        self.test_data.to_csv(os.path.join(self.dataset_dir, f"{self.series_name}_test.csv"))
        
        # Create differenced series for ARIMA
        self.diff_series = self.ts_df['Value'].diff().dropna()
        diff_df = pd.DataFrame(self.diff_series)
        diff_df.to_csv(os.path.join(self.dataset_dir, f"{self.series_name}_differenced.csv"))
        
        # Check stationarity of differenced series
        diff_stationarity = TimeSeriesUtils.check_stationarity(self.diff_series)
        print("\nSTATIONARITY OF DIFFERENCED SERIES:")
        print(f"Test statistic: {diff_stationarity['test_statistic']:.4f}")
        print(f"p-value: {diff_stationarity['p_value']:.4f}")
        print(f"Is stationary: {diff_stationarity['is_stationary']}")
        
        # Create normalized data for ANN and LSTM
        self.scaler = MinMaxScaler(feature_range=(0, 1))
        normalized_values = self.scaler.fit_transform(self.ts_df[['Value']])
        self.normalized_df = pd.DataFrame(normalized_values, index=self.ts_df.index, columns=['Value'])
        self.normalized_df.to_csv(os.path.join(self.dataset_dir, f"{self.series_name}_normalized.csv"))
        
        # Create sequence data for LSTM
        X, y = TimeSeriesUtils.create_sequences(normalized_values, 5)  # Using 5 time steps for LSTM input
        
        # Split sequence data
        self.X_train, self.X_test = X[:train_size-5], X[train_size-5:]
        self.y_train, self.y_test = y[:train_size-5], y[train_size-5:]
        
        # Save sequence data as NumPy files
        np.save(os.path.join(self.dataset_dir, f"{self.series_name}_X_train.npy"), self.X_train)
        np.save(os.path.join(self.dataset_dir, f"{self.series_name}_y_train.npy"), self.y_train)
        np.save(os.path.join(self.dataset_dir, f"{self.series_name}_X_test.npy"), self.X_test)
        np.save(os.path.join(self.dataset_dir, f"{self.series_name}_y_test.npy"), self.y_test)
        
        # Save scaler for later use
        with open(os.path.join(self.dataset_dir, f"{self.series_name}_scaler.pkl"), 'wb') as f:
            pickle.dump(self.scaler, f)
        
        # Generate visualizations
        self._create_visualizations()
        
        # Prepare metadata for the dataset
        self.metadata = {
            'series_name': self.series_name,
            'total_samples': int(len(self.ts_df)),
            'train_samples': int(len(self.train_data)),
            'test_samples': int(len(self.test_data)),
            'start_year': int(self.ts_df.index.min()),
            'end_year': int(self.ts_df.index.max()),
            'is_stationary': bool(stationarity_results['is_stationary']),
            'diff1_is_stationary': bool(diff_stationarity['is_stationary']),
            'scaled_min': float(self.scaler.data_min_[0]),
            'scaled_max': float(self.scaler.data_max_[0])
        }
        
        # Save metadata as JSON
        with open(os.path.join(self.dataset_dir, f"{self.series_name}_metadata.json"), 'w') as f:
            json.dump(self.metadata, f, indent=4)
        
        print(f"\nData preparation complete for {self.series_name}!")
        print(f"Files saved to {self.dataset_dir}/")
        
        return {
            'full_data': self.ts_df,
            'train_data': self.train_data,
            'test_data': self.test_data,
            'differenced_data': diff_df,
            'normalized_data': self.normalized_df,
            'X_train': self.X_train,
            'y_train': self.y_train,
            'X_test': self.X_test,
            'y_test': self.y_test,
            'scaler': self.scaler,
            'metadata': self.metadata
        }
    
    def forecast_arima(self, forecast_horizon=5, order=(0,1,1), seasonal_order=None):
        """
        Forecast time series data using ARIMA model
        
        Parameters:
        -----------
        forecast_horizon : int, default=5
            Number of periods to forecast into the future
        order : tuple, default=(0,1,1)
            ARIMA order parameters (p,d,q)
        seasonal_order : tuple or None, default=None
            Seasonal ARIMA parameters (P,D,Q,s) or None if not using seasonal model
            
        Returns:
        --------
        dict
            Dictionary containing forecasting results
        """
        print(f"\n{'='*80}")
        print(f"ARIMA FORECASTING: {self.series_name}")
        print(f"{'='*80}\n")
        
        # Check if we need differencing based on stationarity test
        if not self.metadata['is_stationary'] and not self.metadata['diff1_is_stationary']:
            print("Warning: Data is not stationary even after first differencing.")
            print("Consider using a different model or additional transformations.")
        
        # Create output directory for ARIMA results
        arima_dir = os.path.join(self.dataset_dir, "arima_results")
        if not os.path.exists(arima_dir):
            os.makedirs(arima_dir)
        
        # Train ARIMA model
        print(f"Training ARIMA{order} model...")
        
        if seasonal_order:
            model = ARIMA(self.train_data['Value'], order=order, seasonal_order=seasonal_order)
            model_name = f"SARIMA{order}x{seasonal_order}"
        else:
            model = ARIMA(self.train_data['Value'], order=order)
            model_name = f"ARIMA{order}"
        
        try:
            fitted_model = model.fit()
            print("Model Summary:")
            print(fitted_model.summary())
            
            # Save model summary
            with open(os.path.join(arima_dir, f"{model_name}_summary.txt"), 'w') as f:
                f.write(str(fitted_model.summary()))
            
            # In-sample predictions
            in_sample_predictions = fitted_model.predict(start=0, end=len(self.train_data)-1)
            
            # Out-of-sample forecasts for test period
            forecasts = fitted_model.forecast(steps=len(self.test_data))
            
            # Evaluate in-sample performance
            in_sample_rmse = np.sqrt(mean_squared_error(self.train_data['Value'], in_sample_predictions))
            in_sample_mae = mean_absolute_error(self.train_data['Value'], in_sample_predictions)
            in_sample_mape = mean_absolute_percentage_error(self.train_data['Value'], in_sample_predictions) * 100
            
            # Evaluate out-of-sample performance
            out_of_sample_rmse = np.sqrt(mean_squared_error(self.test_data['Value'], forecasts))
            out_of_sample_mae = mean_absolute_error(self.test_data['Value'], forecasts)
            out_of_sample_mape = mean_absolute_percentage_error(self.test_data['Value'], forecasts) * 100
            
            print("\nModel Performance:")
            print(f"In-Sample RMSE: {in_sample_rmse:.4f}")
            print(f"In-Sample MAE: {in_sample_mae:.4f}")
            print(f"In-Sample MAPE: {in_sample_mape:.2f}%")
            print(f"Out-of-Sample RMSE: {out_of_sample_rmse:.4f}")
            print(f"Out-of-Sample MAE: {out_of_sample_mae:.4f}")
            print(f"Out-of-Sample MAPE: {out_of_sample_mape:.2f}%")
            
            # Future forecast beyond test data
            future_forecasts = fitted_model.forecast(steps=forecast_horizon)
            
            # Create future index for forecasts
            last_year = int(self.test_data.index[-1])
            future_years = range(last_year + 1, last_year + forecast_horizon + 1)
            
            # Create visualization of results
            plt.figure(figsize=(12, 6))
            
            # Plot training data
            plt.plot(self.train_data.index, self.train_data['Value'], 'b-', label='Training Data')
            
            # Plot in-sample predictions
            plt.plot(self.train_data.index, in_sample_predictions, 'g--', alpha=0.7, label='In-Sample Predictions')
            
            # Plot test data and forecasts
            plt.plot(self.test_data.index, self.test_data['Value'], 'r-', label='Test Data')
            plt.plot(self.test_data.index, forecasts, 'm--', label='Forecasts')
            
            # Plot future forecasts
            plt.plot(future_years, future_forecasts, 'c--', label='Future Forecasts')
            
            # Add confidence intervals for future forecasts
            plt.fill_between(future_years, 
                            future_forecasts - 1.96 * out_of_sample_rmse,
                            future_forecasts + 1.96 * out_of_sample_rmse,
                            color='c', alpha=0.2, label='95% CI')
            
            plt.title(f'{model_name} Forecasting: {self.series_name}', fontsize=16)
            plt.xlabel('Year', fontsize=12)
            plt.ylabel('Value', fontsize=12)
            plt.legend()
            plt.grid(True)
            plt.tight_layout()
            plt.savefig(os.path.join(arima_dir, f"{model_name}_forecasts.png"), dpi=300)
            plt.close()
            
            # Create forecast DataFrame
            forecast_data = pd.DataFrame({
                'Year': list(self.test_data.index) + list(future_years),
                'Actual': list(self.test_data['Value']) + [None] * forecast_horizon,
                'Forecast': list(forecasts) + list(future_forecasts),
                'Lower_CI': list(forecasts - 1.96 * out_of_sample_rmse) + list(future_forecasts - 1.96 * out_of_sample_rmse),
                'Upper_CI': list(forecasts + 1.96 * out_of_sample_rmse) + list(future_forecasts + 1.96 * out_of_sample_rmse)
            })
            
            # Save forecast data
            forecast_data.to_csv(os.path.join(arima_dir, f"{model_name}_forecasts.csv"), index=False)
            
            # Save model performance metrics
            performance = {
                'model': model_name,
                'dataset': self.series_name,
                'in_sample_rmse': float(in_sample_rmse),
                'in_sample_mae': float(in_sample_mae),
                'in_sample_mape': float(in_sample_mape),
                'out_of_sample_rmse': float(out_of_sample_rmse),
                'out_of_sample_mae': float(out_of_sample_mae),
                'out_of_sample_mape': float(out_of_sample_mape)
            }
            
            with open(os.path.join(arima_dir, f"{model_name}_performance.json"), 'w') as f:
                json.dump(performance, f, indent=4)
            
            print(f"\nARIMA forecasting complete for {self.series_name}!")
            
            # Store ARIMA results
            self.arima_results = {
                'model': fitted_model,
                'in_sample_predictions': in_sample_predictions,
                'forecasts': forecasts,
                'future_forecasts': future_forecasts,
                'performance': performance,
                'forecast_data': forecast_data
            }
            
            return self.arima_results
        
        except Exception as e:
            print(f"Error training ARIMA model: {e}")
            return None
    
    def find_optimal_arima_order(self, p_range=(0,2), d_range=(0,2), q_range=(0,2)):
        """
        Find the optimal ARIMA order by grid searching through parameter combinations
        
        Parameters:
        -----------
        p_range : tuple, default=(0,2)
            Range of AR parameters to try (min, max)
        d_range : tuple, default=(0,2)
            Range of differencing parameters to try (min, max)
        q_range : tuple, default=(0,2)
            Range of MA parameters to try (min, max)
            
        Returns:
        --------
        tuple
            Best ARIMA order (p,d,q)
        """
        print(f"\n{'='*80}")
        print(f"FINDING OPTIMAL ARIMA ORDER FOR: {self.series_name}")
        print(f"{'='*80}\n")
        
        # Create output directory for ARIMA results
        arima_dir = os.path.join(self.dataset_dir, "arima_results")
        if not os.path.exists(arima_dir):
            os.makedirs(arima_dir)
        
        # Grid search
        results = []
        
        for p in range(p_range[0], p_range[1] + 1):
            for d in range(d_range[0], d_range[1] + 1):
                for q in range(q_range[0], q_range[1] + 1):
                    order = (p, d, q)
                    try:
                        print(f"Trying ARIMA{order}...")
                        model = ARIMA(self.train_data['Value'], order=order)
                        fitted_model = model.fit()
                        
                        # Forecast test period
                        forecasts = fitted_model.forecast(steps=len(self.test_data))
                        
                        # Calculate error metrics
                        mse = mean_squared_error(self.test_data['Value'], forecasts)
                        rmse = np.sqrt(mse)
                        mae = mean_absolute_error(self.test_data['Value'], forecasts)
                        mape = mean_absolute_percentage_error(self.test_data['Value'], forecasts) * 100
                        
                        # Calculate AIC and BIC
                        aic = fitted_model.aic
                        bic = fitted_model.bic
                        
                        results.append({
                            'order': order,
                            'rmse': rmse,
                            'mae': mae,
                            'mape': mape,
                            'aic': aic,
                            'bic': bic
                        })
                        
                        print(f"ARIMA{order} - RMSE: {rmse:.4f}, MAPE: {mape:.2f}%, AIC: {aic:.2f}")
                        
                    except Exception as e:
                        print(f"Error with ARIMA{order}: {e}")
        
        # Sort results by RMSE
        results.sort(key=lambda x: x['rmse'])
        
        # Save results
        results_df = pd.DataFrame(results)
        results_df.to_csv(os.path.join(arima_dir, 'arima_grid_search_results.csv'), index=False)
        
        # Create visualization of results
        plt.figure(figsize=(12, 8))
        
        # Create order labels
        order_labels = [f"({r['order'][0]},{r['order'][1]},{r['order'][2]})" for r in results]
        
        # Plot RMSE
        plt.subplot(2, 2, 1)
        plt.bar(order_labels, [r['rmse'] for r in results])
        plt.title('RMSE by ARIMA Order')
        plt.xlabel('ARIMA(p,d,q)')
        plt.ylabel('RMSE')
        plt.xticks(rotation=90)
        
        # Plot MAPE
        plt.subplot(2, 2, 2)
        plt.bar(order_labels, [r['mape'] for r in results])
        plt.title('MAPE by ARIMA Order')
        plt.xlabel('ARIMA(p,d,q)')
        plt.ylabel('MAPE (%)')
        plt.xticks(rotation=90)
        
        # Plot AIC
        plt.subplot(2, 2, 3)
        plt.bar(order_labels, [r['aic'] for r in results])
        plt.title('AIC by ARIMA Order')
        plt.xlabel('ARIMA(p,d,q)')
        plt.ylabel('AIC')
        plt.xticks(rotation=90)
        
        # Plot BIC
        plt.subplot(2, 2, 4)
        plt.bar(order_labels, [r['bic'] for r in results])
        plt.title('BIC by ARIMA Order')
        plt.xlabel('ARIMA(p,d,q)')
        plt.ylabel('BIC')
        plt.xticks(rotation=90)
        
        plt.tight_layout()
        plt.savefig(os.path.join(arima_dir, 'arima_grid_search_results.png'), dpi=300)
        plt.close()
        
        # Return best order
        best_order = results[0]['order']
        print(f"\nBest ARIMA order: {best_order} with RMSE: {results[0]['rmse']:.4f}, MAPE: {results[0]['mape']:.2f}%")
        
        return best_order
    
    def forecast_hybrid(self, forecast_horizon=5, arima_order=(0,1,1), ann_layers=[16, 8], epochs=100):
        """
        Forecast time series data using a hybrid ARIMA-ANN model
        
        Parameters:
        -----------
        forecast_horizon : int, default=5
            Number of periods to forecast into the future
        arima_order : tuple, default=(0,1,1)
            ARIMA order parameters (p,d,q)
        ann_layers : list, default=[16, 8]
            List of neurons for each hidden layer in the ANN
        epochs : int, default=100
            Number of training epochs for the ANN
            
        Returns:
        --------
        dict
            Dictionary containing forecasting results
        """
        print(f"\n{'='*80}")
        print(f"ARIMA-ANN HYBRID FORECASTING: {self.series_name}")
        print(f"{'='*80}\n")
        
        # Create output directory for hybrid results
        hybrid_dir = os.path.join(self.dataset_dir, "hybrid_results")
        if not os.path.exists(hybrid_dir):
            os.makedirs(hybrid_dir)
        
        # Step 1: Train ARIMA model for linear component
        print(f"Training ARIMA{arima_order} model for linear component...")
        
        model_arima = ARIMA(self.train_data['Value'], order=arima_order)
        
        try:
            fitted_arima = model_arima.fit()
            
            # Get residuals from training data
            arima_preds_train = fitted_arima.predict(start=0, end=len(self.train_data)-1)
            train_residuals = self.train_data['Value'].values - arima_preds_train.values
            
            # Get ARIMA predictions for test data
            arima_preds_test = fitted_arima.forecast(steps=len(self.test_data))
            
            # Step 2: Train ANN on residuals
            print(f"Training ANN model on ARIMA residuals...")
            
            # Create lagged features for residuals (using last 3 residuals to predict next)
            X_train, y_train = self._create_lagged_features(train_residuals, 3)
            
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
                verbose=1
            )
            
            # Save the ANN model
            model_ann.save(os.path.join(hybrid_dir, f"{self.series_name}_residual_ann.h5"))
            
            # Save the scalers
            with open(os.path.join(hybrid_dir, f"{self.series_name}_scaler_X.pkl"), 'wb') as f:
                pickle.dump(scaler_X, f)
            
            with open(os.path.join(hybrid_dir, f"{self.series_name}_scaler_y.pkl"), 'wb') as f:
                pickle.dump(scaler_y, f)
            
            # Plot training history
            plt.figure(figsize=(10, 6))
            plt.plot(history.history['loss'], label='Training Loss')
            plt.plot(history.history['val_loss'], label='Validation Loss')
            plt.title('ANN Model Training History', fontsize=16)
            plt.xlabel('Epoch', fontsize=12)
            plt.ylabel('Loss (MSE)', fontsize=12)
            plt.legend()
            plt.grid(True)
            plt.tight_layout()
            plt.savefig(os.path.join(hybrid_dir, f"{self.series_name}_ann_training_history.png"), dpi=300)
            plt.close()
            
            # Step 3: Generate hybrid predictions for test set
            # First, we need the initial residuals from the training set to predict the first test residuals
            initial_residuals = train_residuals[-3:]
            
            test_residuals_pred = []
            
            # Iteratively predict each test residual
            for i in range(len(self.test_data)):
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
            hybrid_preds_test = arima_preds_test.values + np.array(test_residuals_pred)
            
            # Step 4: Generate future forecasts
            # Get ARIMA forecasts for future periods
            arima_future_forecasts = fitted_arima.forecast(steps=len(self.test_data) + forecast_horizon)[-forecast_horizon:]
            
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
            hybrid_future_forecasts = arima_future_forecasts.values + np.array(future_residuals_pred)
            
            # Step 5: Evaluate performance
            # In-sample performance (ARIMA only for now)
            in_sample_rmse = np.sqrt(mean_squared_error(self.train_data['Value'], arima_preds_train))
            in_sample_mae = mean_absolute_error(self.train_data['Value'], arima_preds_train)
            in_sample_mape = mean_absolute_percentage_error(self.train_data['Value'], arima_preds_train) * 100
            
            # Out-of-sample performance
            # ARIMA only
            arima_rmse = np.sqrt(mean_squared_error(self.test_data['Value'], arima_preds_test))
            arima_mae = mean_absolute_error(self.test_data['Value'], arima_preds_test)
            arima_mape = mean_absolute_percentage_error(self.test_data['Value'], arima_preds_test) * 100
            
            # Hybrid model
            hybrid_rmse = np.sqrt(mean_squared_error(self.test_data['Value'], hybrid_preds_test))
            hybrid_mae = mean_absolute_error(self.test_data['Value'], hybrid_preds_test)
            hybrid_mape = mean_absolute_percentage_error(self.test_data['Value'], hybrid_preds_test) * 100
            
            print("\nModel Performance:")
            print(f"In-Sample RMSE (ARIMA): {in_sample_rmse:.4f}")
            print(f"In-Sample MAPE (ARIMA): {in_sample_mape:.2f}%")
            print(f"Out-of-Sample RMSE (ARIMA): {arima_rmse:.4f}")
            print(f"Out-of-Sample MAPE (ARIMA): {arima_mape:.2f}%")
            print(f"Out-of-Sample RMSE (Hybrid): {hybrid_rmse:.4f}")
            print(f"Out-of-Sample MAPE (Hybrid): {hybrid_mape:.2f}%")
            print(f"Improvement over ARIMA: {((arima_rmse - hybrid_rmse) / arima_rmse * 100):.2f}%")
            
            # Create future index for forecasts
            last_year = int(self.test_data.index[-1])
            future_years = range(last_year + 1, last_year + forecast_horizon + 1)
            
            # Create visualization of results
            plt.figure(figsize=(12, 6))
            
            # Plot training data
            plt.plot(self.train_data.index, self.train_data['Value'], 'b-', label='Training Data')
            
            # Plot test data
            plt.plot(self.test_data.index, self.test_data['Value'], 'r-', label='Test Data')
            
            # Plot ARIMA predictions for test set
            plt.plot(self.test_data.index, arima_preds_test, 'g--', alpha=0.7, label='ARIMA Forecasts')
            
            # Plot hybrid predictions for test set
            plt.plot(self.test_data.index, hybrid_preds_test, 'm--', label='Hybrid Forecasts')
            
            # Plot future forecasts
            plt.plot(future_years, hybrid_future_forecasts, 'c--', label='Future Forecasts')
            
            # Add confidence intervals for future forecasts
            plt.fill_between(future_years, 
                            hybrid_future_forecasts - 1.96 * hybrid_rmse,
                            hybrid_future_forecasts + 1.96 * hybrid_rmse,
                            color='c', alpha=0.2, label='95% CI')
            
            plt.title(f'ARIMA-ANN Hybrid Forecasting: {self.series_name}', fontsize=16)
            plt.xlabel('Year', fontsize=12)
            plt.ylabel('Value', fontsize=12)
            plt.legend()
            plt.grid(True)
            plt.tight_layout()
            plt.savefig(os.path.join(hybrid_dir, f"{self.series_name}_hybrid_forecasts.png"), dpi=300)
            plt.close()
            
            # Create forecast DataFrame
            forecast_data = pd.DataFrame({
                'Year': list(self.test_data.index) + list(future_years),
                'Actual': list(self.test_data['Value']) + [None] * forecast_horizon,
                'ARIMA_Forecast': list(arima_preds_test) + list(arima_future_forecasts),
                'Hybrid_Forecast': list(hybrid_preds_test) + list(hybrid_future_forecasts),
                'Lower_CI': list(hybrid_preds_test - 1.96 * hybrid_rmse) + list(hybrid_future_forecasts - 1.96 * hybrid_rmse),
                'Upper_CI': list(hybrid_preds_test + 1.96 * hybrid_rmse) + list(hybrid_future_forecasts + 1.96 * hybrid_rmse)
            })
            
            # Save forecast data
            forecast_data.to_csv(os.path.join(hybrid_dir, f"{self.series_name}_hybrid_forecasts.csv"), index=False)
            
            # Save model performance metrics
            performance = {
                'dataset': self.series_name,
                'arima_order': str(arima_order),
                'ann_layers': str(ann_layers),
                'in_sample_rmse': float(in_sample_rmse),
                'in_sample_mae': float(in_sample_mae),
                'in_sample_mape': float(in_sample_mape),
                'arima_rmse': float(arima_rmse),
                'arima_mae': float(arima_mae),
                'arima_mape': float(arima_mape),
                'hybrid_rmse': float(hybrid_rmse),
                'hybrid_mae': float(hybrid_mae),
                'hybrid_mape': float(hybrid_mape),
                'improvement_percentage': float((arima_rmse - hybrid_rmse) / arima_rmse * 100)
            }
            
            with open(os.path.join(hybrid_dir, f"{self.series_name}_hybrid_performance.json"), 'w') as f:
                json.dump(performance, f, indent=4)
            
            print(f"\nARIMA-ANN hybrid forecasting complete for {self.series_name}!")
            
            # Store hybrid results
            self.hybrid_results = {
                'arima_model': fitted_arima,
                'ann_model': model_ann,
                'arima_preds_test': arima_preds_test,
                'hybrid_preds_test': hybrid_preds_test,
                'arima_future_forecasts': arima_future_forecasts,
                'hybrid_future_forecasts': hybrid_future_forecasts,
                'performance': performance,
                'forecast_data': forecast_data
            }
            
            return self.hybrid_results
        
        except Exception as e:
            print(f"Error in ARIMA-ANN hybrid model: {e}")
            return None
    
    def _create_lagged_features(self, series, lag=3):
        """Create lagged features for time series data"""
        X, y = [], []
        for i in range(lag, len(series)):
            X.append(series[i-lag:i])
            y.append(series[i])
        return np.array(X), np.array(y)
    
    def _create_visualizations(self):
        """Create visualizations for data analysis"""
        # Set up the visualization directory
        vis_dir = os.path.join(self.dataset_dir, "visualizations")
        if not os.path.exists(vis_dir):
            os.makedirs(vis_dir)
        
        # Set style
        plt.style.use('seaborn-v0_8-darkgrid')
        
        # 1. Original time series with train/test split
        plt.figure(figsize=(12, 6))
        plt.plot(self.ts_df.index, self.ts_df['Value'], 'b-', label='Full Dataset')
        plt.plot(self.train_data.index, self.train_data['Value'], 'g-', label='Training Data')
        plt.plot(self.test_data.index, self.test_data['Value'], 'r-', label='Testing Data')
        plt.title(f'Time Series Split: {self.series_name}', fontsize=16)
        plt.xlabel('Year', fontsize=12)
        plt.ylabel('Value', fontsize=12)
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(os.path.join(vis_dir, '1_data_split.png'), dpi=300)
        plt.close()
        
        # 2. ACF and PACF plots for original series
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
        plot_acf(self.ts_df['Value'].dropna(), lags=20, ax=ax1)
        ax1.set_title(f'ACF: {self.series_name}', fontsize=14)
        
        plot_pacf(self.ts_df['Value'].dropna(), lags=20, ax=ax2)
        ax2.set_title(f'PACF: {self.series_name}', fontsize=14)
        
        plt.tight_layout()
        plt.savefig(os.path.join(vis_dir, '2_acf_pacf_original.png'), dpi=300)
        plt.close()
        
        # 3. Differenced series
        plt.figure(figsize=(12, 6))
        plt.plot(self.diff_series.index, self.diff_series.values)
        plt.title(f'Differenced Series: {self.series_name}', fontsize=16)
        plt.xlabel('Year', fontsize=12)
        plt.ylabel('Differenced Value', fontsize=12)
        plt.axhline(y=0, color='r', linestyle='-', alpha=0.3)
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(os.path.join(vis_dir, '3_differenced_series.png'), dpi=300)
        plt.close()
        
        # 4. ACF and PACF plots for differenced series
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
        plot_acf(self.diff_series.dropna(), lags=20, ax=ax1)
        ax1.set_title(f'ACF of Differenced Series: {self.series_name}', fontsize=14)
        
        plot_pacf(self.diff_series.dropna(), lags=20, ax=ax2)
        ax2.set_title(f'PACF of Differenced Series: {self.series_name}', fontsize=14)
        
        plt.tight_layout()
        plt.savefig(os.path.join(vis_dir, '4_acf_pacf_differenced.png'), dpi=300)
        plt.close()
        
        # 5. Seasonal Decomposition (if enough data points)
        if len(self.ts_df) >= 12:  # Need at least 2 periods for seasonal decomposition
            try:
                decomposition = seasonal_decompose(self.ts_df['Value'], model='additive', period=5)
                
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
                plt.savefig(os.path.join(vis_dir, '5_seasonal_decomposition.png'), dpi=300)
                plt.close()
            except:
                print("Could not perform seasonal decomposition - insufficient data or other issue.")
