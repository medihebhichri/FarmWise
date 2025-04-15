import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import LSTM, Dense, Dropout, Input, Concatenate, BatchNormalization
import xgboost as xgb
from prophet import Prophet
import statsmodels.api as sm
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

class EnhancedRiskAnalysis:
    def __init__(self):
        self.scalers = {
            'features': StandardScaler(),
            'target': StandardScaler(),
            'land': StandardScaler()
        }
        self.label_encoder = LabelEncoder()
        self.models = {
            'lstm': None,
            'xgb': None,
            'rf': None,
            'gb': None,
            'prophet': {}
        }
        
    def preprocess_land_data(self, land_df):
        """Process the complex land data structure"""
        # Extract Tunisia-specific features
        tunisia_cols = [col for col in land_df.columns if col.startswith('Tunisia')]
        tunisia_land = land_df[tunisia_cols].copy()
        
        # Handle missing values first
        tunisia_land = tunisia_land.fillna(method='ffill').fillna(method='bfill')
        
        # Create derived metrics with safe division
        tunisia_land['land_utilization'] = (
            tunisia_land['Tunisia_Agricultural_land_Share_in_Land_area'].fillna(0) / 100
        )
        tunisia_land['irrigation_efficiency'] = (
            tunisia_land['Tunisia_Agriculture_area_actually_irrigated_Share_in_Agricultural_land'].fillna(0) / 100
        )
        
        # Safe division for crop diversity
        temp_area = tunisia_land['Tunisia_Temporary_crops_Area'].replace(0, 1)
        tunisia_land['crop_diversity'] = (
            tunisia_land['Tunisia_Permanent_crops_Share_in_Agricultural_land'].fillna(0) / temp_area
        )
        
        return tunisia_land
    
    def calculate_climate_risk_metrics(self, temp_df):
        """Calculate advanced climate risk metrics"""
        temp_df = temp_df[temp_df['Area'] == 'Tunisia'].copy()
        
        # Handle missing values in temperature data
        temp_df['TemperatureChange'] = temp_df['TemperatureChange'].fillna(method='ffill')
        temp_df['StandardDeviation'] = temp_df['StandardDeviation'].fillna(method='ffill')
        
        # Calculate rolling statistics with min_periods to handle edge cases
        temp_df['temp_volatility'] = temp_df['TemperatureChange'].rolling(12, min_periods=1).std()
        temp_df['temp_trend'] = temp_df['TemperatureChange'].rolling(12, min_periods=1).mean()
        temp_df['extreme_temp_events'] = (
            (temp_df['TemperatureChange'].abs() > 2 * temp_df['StandardDeviation']).rolling(12, min_periods=1).sum()
        )
        
        # Handle missing values in seasonal decomposition
        temp_yearly = temp_df.set_index(pd.to_datetime(temp_df['Date']))['TemperatureChange']
        decomposition = sm.tsa.seasonal_decompose(temp_yearly.fillna(method='ffill'), period=12)
        temp_df['seasonal_factor'] = decomposition.seasonal
        temp_df['temp_trend_decomp'] = decomposition.trend
        
        # Fill any remaining NaN values
        temp_df = temp_df.fillna(method='ffill').fillna(method='bfill')
        
        return temp_df
    
    def create_feature_matrix(self, agriculture_df, temp_df, land_df):
        """Create an enhanced feature matrix combining all data sources"""
        
        # Process temperature data
        climate_features = self.calculate_climate_risk_metrics(temp_df)
        climate_features['Year'] = pd.to_datetime(climate_features['Date']).dt.year
        climate_yearly = climate_features.groupby('Year').agg({
            'temp_volatility': 'mean',
            'temp_trend': 'last',
            'extreme_temp_events': 'sum',
            'seasonal_factor': 'std',
            'temp_trend_decomp': 'mean'
        }).reset_index()
        
        # Process land data
        land_features = self.preprocess_land_data(land_df)
        
        # Process agriculture data
        tunisia_ag = agriculture_df[agriculture_df['Area'] == 'Tunisia'].copy()
        tunisia_ag['Year'] = pd.to_datetime(tunisia_ag['Year']).dt.year
        
        # Handle missing values in agriculture data
        tunisia_ag['Value'] = tunisia_ag['Value'].fillna(method='ffill')
        
        # Merge features with proper handling of missing values
        features = pd.merge(
            tunisia_ag,
            climate_yearly,
            on='Year',
            how='left'
        ).fillna(method='ffill')
        
        # Add land features
        for col in land_features.columns:
            features[col] = land_features[col].values[0]
        
        # Calculate risk metrics with safe operations
        features['production_volatility'] = features.groupby('Item')['Value'].transform(
            lambda x: x.rolling(3, min_periods=1).std().fillna(0)
        )
        features['production_trend'] = features.groupby('Item')['Value'].transform(
            lambda x: x.rolling(3, min_periods=1).mean().fillna(method='ffill')
        )
        
        # Safe calculations for risk metrics
        features['climate_sensitivity'] = features['production_volatility'] * features['temp_volatility'].fillna(0)
        features['land_climate_risk'] = features['temp_volatility'] * (1 - features['land_utilization'])
        features['production_risk_score'] = (
            features['production_volatility'] * 
            features['climate_sensitivity'] * 
            (1 - features['irrigation_efficiency'])
        )
        
        # Final cleanup of any remaining NaN values
        features = features.fillna(method='ffill').fillna(method='bfill').fillna(0)
        
        return features
    
    def build_deep_learning_model(self, input_shape, aux_shape):
        """Build an enhanced deep learning model with attention mechanism"""
        # Time series branch
        ts_input = Input(shape=input_shape)
        x = LSTM(128, return_sequences=True)(ts_input)
        x = BatchNormalization()(x)
        x = Dropout(0.3)(x)
        x = LSTM(64)(x)
        x = BatchNormalization()(x)
        x = Dropout(0.3)(x)
        
        # Auxiliary features branch
        aux_input = Input(shape=(aux_shape,))
        aux = Dense(64, activation='relu')(aux_input)
        aux = BatchNormalization()(aux)
        aux = Dropout(0.3)(aux)
        
        # Merge branches
        merged = Concatenate()([x, aux])
        merged = Dense(128, activation='relu')(merged)
        merged = BatchNormalization()(merged)
        merged = Dropout(0.3)(merged)
        merged = Dense(64, activation='relu')(merged)
        merged = BatchNormalization()(merged)
        output = Dense(1)(merged)
        
        model = Model(inputs=[ts_input, aux_input], outputs=output)
        model.compile(optimizer='adam', loss='huber')
        return model
    
    def fit(self, agriculture_df, temp_df, land_df):
        """Train the ensemble of models"""
        # Prepare feature matrix
        features = self.create_feature_matrix(agriculture_df, temp_df, land_df)
        
        # Prepare data for deep learning
        sequence_features = features[[
            'Value', 'temp_volatility', 'temp_trend', 'production_volatility',
            'climate_sensitivity'
        ]].values
        sequence_features = self.scalers['features'].fit_transform(sequence_features)
        
        aux_features = features[[
            'land_utilization', 'irrigation_efficiency', 'crop_diversity',
            'production_risk_score', 'land_climate_risk'
        ]].values
        aux_features = self.scalers['land'].fit_transform(aux_features)
        
        # Create sequences for LSTM
        X_seq, y_seq = [], []
        lookback = 12
        for i in range(len(sequence_features) - lookback):
            X_seq.append(sequence_features[i:(i + lookback)])
            y_seq.append(sequence_features[i + lookback, 0])
        X_seq = np.array(X_seq)
        y_seq = np.array(y_seq)
        
        # Train deep learning model
        self.models['lstm'] = self.build_deep_learning_model(
            X_seq.shape[1:], aux_features.shape[1]
        )
        self.models['lstm'].fit(
            [X_seq, aux_features[lookback:]], 
            y_seq,
            epochs=100,
            batch_size=32,
            validation_split=0.2,
            verbose=0
        )
        
        # Train XGBoost model
        self.models['xgb'] = xgb.XGBRegressor(
            n_estimators=200,
            learning_rate=0.1,
            max_depth=5,
            objective='reg:squarederror'
        )
        self.models['xgb'].fit(
            np.hstack([sequence_features, aux_features]),
            features['Value'].values
        )
        
        # Train Random Forest model
        self.models['rf'] = RandomForestRegressor(
            n_estimators=200,
            max_depth=10,
            random_state=42
        )
        self.models['rf'].fit(
            np.hstack([sequence_features, aux_features]),
            features['Value'].values
        )
        
        # Train Gradient Boosting model
        self.models['gb'] = GradientBoostingRegressor(
            n_estimators=200,
            learning_rate=0.1,
            max_depth=5,
            random_state=42
        )
        self.models['gb'].fit(
            np.hstack([sequence_features, aux_features]),
            features['Value'].values
        )
        
        # Train Prophet models for each item
        for item in features['Item'].unique():
            item_data = features[features['Item'] == item].copy()
            
            # Prepare Prophet dataframe with required regressors
            prophet_df = pd.DataFrame({
                'ds': pd.to_datetime(item_data['Year'], format='%Y'),
                'y': item_data['Value'],
                'temp_volatility': item_data['temp_volatility'],
                'production_risk_score': item_data['production_risk_score']
            })
            
            model = Prophet(
                yearly_seasonality=True,
                weekly_seasonality=False,
                daily_seasonality=False,
                seasonality_mode='multiplicative'
            )
            # Add the regressors
            model.add_regressor('temp_volatility')
            model.add_regressor('production_risk_score')
            
            # Fit the model
            model.fit(prophet_df)
            self.models['prophet'][item] = model
    
    def calculate_var_metrics(self, predictions, confidence_level=0.95):
        """Calculate Value at Risk metrics"""
        var = np.percentile(predictions, (1 - confidence_level) * 100)
        cvar = np.mean(predictions[predictions <= var])
        return var, cvar
    
    def calculate_risk_metrics(self, features, predictions):
        """Calculate comprehensive risk metrics"""
        risk_metrics = {
            'volatility': np.std(predictions),
            'var_95': self.calculate_var_metrics(predictions)[0],
            'cvar_95': self.calculate_var_metrics(predictions)[1],
            'climate_risk_score': np.mean(features['climate_sensitivity']),
            'land_risk_score': np.mean(features['land_climate_risk']),
            'production_risk_score': np.mean(features['production_risk_score']),
            'trend': np.polyfit(np.arange(len(predictions)), predictions, 1)[0],
            'extreme_event_probability': np.mean(
                predictions <= np.percentile(predictions, 5)
            )
        }
        return risk_metrics
    
    def predict_risk(self, features, forecast_periods=12):
        """Generate ensemble predictions and comprehensive risk analysis"""
        # Prepare data
        sequence_features = features[[
            'Value', 'temp_volatility', 'temp_trend', 'production_volatility',
            'climate_sensitivity'
        ]].values
        sequence_features = self.scalers['features'].transform(sequence_features)
        
        aux_features = features[[
            'land_utilization', 'irrigation_efficiency', 'crop_diversity',
            'production_risk_score', 'land_climate_risk'
        ]].values
        aux_features = self.scalers['land'].transform(aux_features)
        
        X_seq, _ = [], []
        lookback = 12
        for i in range(len(sequence_features) - lookback):
            X_seq.append(sequence_features[i:(i + lookback)])
        X_seq = np.array(X_seq)
        
        # Generate predictions from each model
        lstm_pred = self.models['lstm'].predict([X_seq, aux_features[lookback:]])
        xgb_pred = self.models['xgb'].predict(
            np.hstack([sequence_features, aux_features])
        )
        rf_pred = self.models['rf'].predict(
            np.hstack([sequence_features, aux_features])
        )
        gb_pred = self.models['gb'].predict(
            np.hstack([sequence_features, aux_features])
        )
        
        # Get Prophet predictions
        prophet_predictions = {}
        for item in features['Item'].unique():
            model = self.models['prophet'].get(item)
            if model:
                future = pd.DataFrame({
                    'ds': pd.date_range(
                        start=features['Year'].max(),
                        periods=forecast_periods,
                        freq='Y'
                    )
                })
                future['temp_volatility'] = features['temp_volatility'].mean()
                future['production_risk_score'] = features['production_risk_score'].mean()
                forecast = model.predict(future)
                prophet_predictions[item] = forecast['yhat'].values
        
        # Calculate the number of predictions to use (use last n_preds predictions from each model)
        n_preds = min(forecast_periods, len(lstm_pred), len(xgb_pred))
        
        # Align predictions to the same length
        lstm_aligned = lstm_pred[-n_preds:].flatten()
        xgb_aligned = xgb_pred[-n_preds:]
        rf_aligned = rf_pred[-n_preds:]
        gb_aligned = gb_pred[-n_preds:]
        
        # Calculate mean of Prophet predictions for the same period
        prophet_aligned = np.mean([
            predictions[-n_preds:] for predictions in prophet_predictions.values()
        ], axis=0)
        
        # Ensure all arrays have the same shape
        assert all(len(x) == n_preds for x in [
            lstm_aligned, xgb_aligned, rf_aligned, gb_aligned, prophet_aligned
        ]), "Prediction arrays must have the same length"
        
        # Ensemble predictions with weighted averaging
        ensemble_pred = (
            0.3 * lstm_aligned +
            0.2 * xgb_aligned +
            0.2 * rf_aligned +
            0.2 * gb_aligned +
            0.1 * prophet_aligned
        )
        
        # Calculate comprehensive risk metrics
        risk_metrics = self.calculate_risk_metrics(features, ensemble_pred)
        
        # Add forecast uncertainty
        prediction_std = np.std([
            lstm_aligned,
            xgb_aligned,
            rf_aligned,
            gb_aligned,
            prophet_aligned
        ], axis=0)
        
        risk_metrics['forecast_uncertainty'] = np.mean(prediction_std)
        risk_metrics['model_disagreement'] = stats.variation(prediction_std)
        
        return ensemble_pred, risk_metrics, prophet_predictions

def generate_risk_report(predictions, risk_metrics, features):
    """Generate a comprehensive risk analysis report"""
    report = {
        'summary': {
            'overall_risk_score': np.mean([
                risk_metrics['climate_risk_score'],
                risk_metrics['land_risk_score'],
                risk_metrics['production_risk_score']
            ]),
            'trend_direction': 'Upward' if risk_metrics['trend'] > 0 else 'Downward',
            'volatility': risk_metrics['volatility'],
            'var_95': risk_metrics['var_95'],
            'cvar_95': risk_metrics['cvar_95']
        },
        'risk_factors': {
            'climate_risk': {
                'score': risk_metrics['climate_risk_score'],
                'temperature_volatility': features['temp_volatility'].mean(),
                'extreme_events_probability': risk_metrics['extreme_event_probability']
            },
            'land_risk': {
                'score': risk_metrics['land_risk_score'],
                'land_utilization': features['land_utilization'].mean(),
                'irrigation_efficiency': features['irrigation_efficiency'].mean()
            },
            'production_risk': {
                'score': risk_metrics['production_risk_score'],
                'production_volatility': features['production_volatility'].mean(),
                'forecast_uncertainty': risk_metrics['forecast_uncertainty']
            }
        },
        'recommendations': []
    }
    
    # Generate recommendations based on risk factors
    if report['risk_factors']['climate_risk']['score'] > 0.7:
        report['recommendations'].append(
            'High climate risk detected. Consider implementing climate adaptation strategies.'
        )
    if report['risk_factors']['land_risk']['score'] > 0.7:
        report['recommendations'].append(
            'High land risk detected. Review land utilization and irrigation practices.'
        )
    if report['risk_factors']['production_risk']['score'] > 0.7:
        report['recommendations'].append(
            'High production risk detected. Consider diversifying crops and improving resilience.'
        )
    if risk_metrics['model_disagreement'] > 0.5:
        report['recommendations'].append(
            'High forecast uncertainty detected. Consider collecting more data or reviewing assumptions.'
        )
    
    return report

def main():
    # Load data
    agriculture_df = pd.read_csv('Value of Agricultural Production_Cleaned.csv')
    temperature_df = pd.read_csv('Temperature_Cleaned.csv')
    land_df = pd.read_csv('LAND_DATA_CLEANED.csv')
    
    # Initialize and train model
    model = EnhancedRiskAnalysis()
    model.fit(agriculture_df, temperature_df, land_df)
    
    # Create feature matrix
    features = model.create_feature_matrix(agriculture_df, temperature_df, land_df)
    
    # Generate predictions and risk analysis
    predictions, risk_metrics, prophet_predictions = model.predict_risk(features)
    
    # Generate risk report
    report = generate_risk_report(predictions, risk_metrics, features)
    
    # Save results
    results_df = pd.DataFrame({
        'Date': features['Year'].unique()[-len(predictions):],
        'Predicted_Value': predictions,
        'Climate_Risk_Score': risk_metrics['climate_risk_score'],
        'Land_Risk_Score': risk_metrics['land_risk_score'],
        'Production_Risk_Score': risk_metrics['production_risk_score'],
        'Forecast_Uncertainty': risk_metrics['forecast_uncertainty']
    })
    
    results_df.to_csv('risk_analysis_results.csv', index=False)
    
    # Print summary report
    print("\nRisk Analysis Summary Report:")
    print(f"Overall Risk Score: {report['summary']['overall_risk_score']:.2f}")
    print(f"Trend Direction: {report['summary']['trend_direction']}")
    print(f"95% Value at Risk: {report['summary']['var_95']:.2f}")
    print(f"95% Conditional VaR: {report['summary']['cvar_95']:.2f}")
    print("\nRisk Factors:")
    print(f"Climate Risk Score: {report['risk_factors']['climate_risk']['score']:.2f}")
    print(f"Land Risk Score: {report['risk_factors']['land_risk']['score']:.2f}")
    print(f"Production Risk Score: {report['risk_factors']['production_risk']['score']:.2f}")
    print("\nRecommendations:")
    for rec in report['recommendations']:
        print(f"- {rec}")

if __name__ == "__main__":
    main()