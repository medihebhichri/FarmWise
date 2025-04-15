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
from arch import arch_model
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
            'prophet': {},
            'garch': None
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
    
    def calculate_extreme_events_probability(self, data, threshold=0.95):
        """Calculate probability of extreme events using EVT"""
        sorted_data = np.sort(data)
        threshold_value = np.percentile(data, threshold * 100)
        exceedances = sorted_data[sorted_data > threshold_value]
        
        # Fit Generalized Pareto Distribution
        try:
            shape, loc, scale = stats.genpareto.fit(exceedances)
            return shape, loc, scale
        except Exception as e:
            # Fallback to normal distribution if GPD fitting fails
            return stats.norm.fit(exceedances)
        
    def compute_dynamic_volatility(self, returns):
        """Compute dynamic volatility using GARCH(1,1)"""
        try:
            model = arch_model(returns, vol='Garch', p=1, q=1)
            res = model.fit(disp='off')
            return res.conditional_volatility
        except Exception as e:
            # Fallback to rolling standard deviation if GARCH fails
            return returns.rolling(window=12, min_periods=1).std()
        
    def calculate_dependency(self, data1, data2):
        """Calculate dependency structure using rank correlation"""
        return stats.spearmanr(data1, data2)[0]
        
    def calculate_climate_risk_metrics(self, temp_df):
        """Enhanced climate risk metrics calculation"""
        temp_df = temp_df[temp_df['Area'] == 'Tunisia'].copy()
        
        # Handle missing values
        temp_df['TemperatureChange'] = temp_df['TemperatureChange'].fillna(method='ffill')
        temp_df['StandardDeviation'] = temp_df['StandardDeviation'].fillna(method='ffill')
        
        # Calculate dynamic volatility using GARCH (or fallback)
        temp_df['temp_volatility'] = self.compute_dynamic_volatility(temp_df['TemperatureChange'])
        
        # Calculate extreme temperature events probability
        shape, loc, scale = self.calculate_extreme_events_probability(temp_df['TemperatureChange'])
        temp_df['extreme_temp_probability'] = 1 - stats.genpareto.cdf(
            temp_df['TemperatureChange'], 
            shape, 
            loc, 
            scale
        )
        
        # Enhanced seasonal decomposition
        temp_yearly = temp_df.set_index(pd.to_datetime(temp_df['Date']))['TemperatureChange']
        decomposition = sm.tsa.seasonal_decompose(temp_yearly.fillna(method='ffill'), period=12)
        temp_df['seasonal_factor'] = decomposition.seasonal
        temp_df['trend'] = decomposition.trend
        temp_df['residual'] = decomposition.resid
        
        # Calculate rolling statistics with adaptive window
        for window in [3, 6, 12]:
            # Volatility calculation
            temp_df[f'temp_volatility_{window}m'] = (
                temp_df['TemperatureChange']
                .rolling(window, min_periods=1)
                .std()
            )
            
            # Momentum calculation
            temp_df[f'temp_momentum_{window}m'] = (
                temp_df['TemperatureChange']
                .rolling(window, min_periods=1)
                .mean()
                .diff()
                .fillna(0)
            )
        
        return temp_df

    def create_advanced_features(self, features):
        """Create advanced feature interactions"""
        # Temperature-Production Interaction
        features['temp_prod_interaction'] = (
            features['temp_volatility_mean'].fillna(0) * 
            features['production_volatility'].fillna(0)
        )
        
        # Climate Sensitivity with Memory
        features['climate_memory'] = (
            features['temp_volatility_mean'].fillna(0).ewm(span=12).mean() * 
            features['production_volatility'].fillna(0).ewm(span=12).mean()
        )
        
        # Non-linear transformations
        features['temp_volatility_squared'] = features['temp_volatility_mean'].fillna(0) ** 2
        features['production_risk_squared'] = features['production_volatility'].fillna(0) ** 2
        
        # Cross-sectional ranking features
        features['temp_vol_rank'] = features.groupby('Year')['temp_volatility_mean'].rank(pct=True)
        features['prod_vol_rank'] = features.groupby('Year')['production_volatility'].rank(pct=True)
        
        return features

    def calculate_tail_risk_measures(self, returns, alpha=0.05):
        """Calculate sophisticated tail risk measures"""
        var = np.percentile(returns, alpha * 100)
        cvar = returns[returns <= var].mean()
        
        # Calculate Expected Shortfall using Gaussian and Student-t distributions
        norm_params = stats.norm.fit(returns)
        t_params = stats.t.fit(returns)
        
        gaussian_es = stats.norm.expect(
            lambda x: x, 
            args=(norm_params[0], norm_params[1]),
            lb=-np.inf, 
            ub=var
        )
        
        student_t_es = stats.t.expect(
            lambda x: x,
            args=t_params,
            lb=-np.inf,
            ub=var
        )
        
        return {
            'var': var,
            'cvar': cvar,
            'gaussian_es': gaussian_es,
            'student_t_es': student_t_es
        }

    def create_feature_matrix(self, agriculture_df, temp_df, land_df):
        """Create enhanced feature matrix with sophisticated risk metrics"""
        try:
            # Process temperature data with advanced metrics
            climate_features = self.calculate_climate_risk_metrics(temp_df)
            climate_features['Year'] = pd.to_datetime(climate_features['Date']).dt.year
            
            def safe_kurtosis(x):
                try:
                    return stats.kurtosis(x, nan_policy='omit')
                except:
                    return 0.0
            
            # Aggregate climate features yearly with flattened structure
            climate_yearly = climate_features.groupby('Year').agg({
                'temp_volatility': ['mean', 'max', 'std'],
                'extreme_temp_probability': 'max',
                'seasonal_factor': lambda x: safe_kurtosis(x),
                'trend': 'last',
                'temp_momentum_12m': 'last'  # Assuming 12-month momentum exists
            }).reset_index()
            
            # Flatten column names manually
            climate_yearly.columns = [
                'Year',
                'temp_volatility_mean',
                'temp_volatility_max',
                'temp_volatility_std',
                'extreme_temp_probability_max',
                'seasonal_factor_kurtosis',
                'trend_last',
                'temp_momentum_12m_last'
            ]
            
            # Ensure required columns exist with default values
            expected_cols = [
                'temp_volatility_mean', 'temp_volatility_max', 'temp_volatility_std',
                'extreme_temp_probability_max', 'seasonal_factor_kurtosis',
                'trend_last', 'temp_momentum_12m_last'
            ]
            for col in expected_cols:
                if col not in climate_yearly.columns:
                    climate_yearly[col] = 0.0
            
            return self._merge_and_process_features(climate_yearly, agriculture_df, land_df)
            
        except Exception as e:
            print(f"Error in feature matrix creation: {e}")
            raise

    def _merge_and_process_features(self, climate_yearly, agriculture_df, land_df):
        """Helper method to merge and process features with error handling"""
        try:
            # Process agriculture data with advanced volatility
            tunisia_ag = agriculture_df[agriculture_df['Area'] == 'Tunisia'].copy()
            
            # Create a default 'Item' column if not present
            if 'Item' not in tunisia_ag.columns:
                tunisia_ag['Item'] = 'Default'
            
            # Ensure Year column is in the correct format for both dataframes
            tunisia_ag['Year'] = pd.to_datetime(tunisia_ag['Year']).dt.year.astype(int)
            climate_yearly['Year'] = pd.to_datetime(climate_yearly['Year'].astype(str)).dt.year.astype(int)
            
            # Debug information
            print("\nDebug Information:")
            print("Agriculture Years dtype:", tunisia_ag['Year'].dtype)
            print("Climate Years dtype:", climate_yearly['Year'].dtype)
            print("Agriculture Years:", sorted(tunisia_ag['Year'].unique()))
            print("Climate Years:", sorted(climate_yearly['Year'].unique()))
            print("Agriculture DataFrame shape:", tunisia_ag.shape)
            print("Climate DataFrame shape:", climate_yearly.shape)
            
            # Calculate production volatility with error handling
            try:
                tunisia_ag['production_volatility'] = self.compute_dynamic_volatility(
                    tunisia_ag['Value'].pct_change().fillna(0)
                )
            except Exception as e:
                print(f"Warning: Dynamic volatility calculation failed: {e}")
                tunisia_ag['production_volatility'] = tunisia_ag['Value'].pct_change().rolling(12).std().fillna(0)
            
            # Merge features
            tunisia_ag = tunisia_ag.reset_index(drop=True)
            climate_yearly = climate_yearly.reset_index(drop=True)
            
            features = pd.merge(
                tunisia_ag,
                climate_yearly,
                on='Year',
                how='left',
                validate='many_to_one'
            )
            print("Merged DataFrame shape:", features.shape)
            
            # Process land data
            land_features = self.preprocess_land_data(land_df)
            for col in land_features.columns:
                if len(land_features[col]) > 0:
                    features[col] = land_features[col].values[0]
                else:
                    features[col] = 0.0
            
            # Create advanced features
            features = self.create_advanced_features(features)
            
            # Ensure required columns exist
            for col in ['land_utilization', 'irrigation_efficiency', 'crop_diversity']:
                if col not in features.columns:
                    features[col] = 0.0
            
            # Compute additional risk metric: land_climate_risk
            features['land_climate_risk'] = (
                features['temp_volatility_mean'].fillna(0) / (features['temp_volatility_max'].fillna(0) + 1e-5)
            ) * (1 - features['land_utilization'].fillna(0))
            
            # Fill any remaining NaNs after merge
            features = features.fillna(0)
            
            # Calculate risk scores
            features['climate_risk_score'] = self.calculate_climate_risk_score(features)
            features['production_risk_score'] = self.calculate_production_risk_score(features)
            features['systemic_risk_score'] = self.calculate_systemic_risk_score(features)
            
            return features
            
        except Exception as e:
            print(f"Error in feature merging and processing: {e}")
            raise

    def calculate_climate_risk_score(self, features):
        """Calculate climate risk score using multiple factors"""
        momentum = features.get('temp_momentum_12m_last', features['temp_volatility_mean'].diff().fillna(0)).fillna(0)
        score = (
            0.4 * features['temp_volatility_mean'].fillna(0) +
            0.3 * features['extreme_temp_probability_max'].fillna(0) +
            0.2 * momentum +
            0.1 * features['seasonal_factor_kurtosis'].fillna(0)
        ).clip(0, 1)
        return score
        
    def calculate_production_risk_score(self, features):
        """Calculate production risk score with non-linear relationships"""
        base_score = (
            features['production_volatility'].fillna(0) * 
            np.exp(-features['irrigation_efficiency'].fillna(0)) *
            (1 + features.get('temp_prod_interaction', 0))
        )
        return self.normalize_score(base_score)
        
    def calculate_systemic_risk_score(self, features):
        """Calculate systemic risk incorporating multiple risk factors"""
        return self.normalize_score(
            features['climate_risk_score'].fillna(0) * 
            features['production_risk_score'].fillna(0) * 
            (1 - features['land_utilization'].fillna(0)) *
            (1 + features['climate_memory'].fillna(0))
        )
        
    def normalize_score(self, series):
        """Normalize scores to 0-1 range"""
        s = series.fillna(0)
        return ((s - s.min()) / (s.max() - s.min() + 1e-5)).clip(0, 1)
    
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
        
        # Prepare data for deep learning using consistent feature names
        sequence_features = features[[
            'Value', 'temp_volatility_mean', 'trend_last', 'production_volatility',
            'climate_memory'
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
        
        # Train models with error handling
        try:
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
        except Exception as e:
            print(f"Warning: LSTM training failed: {e}")
            self.models['lstm'] = None

        try:
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
        except Exception as e:
            print(f"Warning: XGBoost training failed: {e}")
            self.models['xgb'] = None

        # Train Prophet models for each item
        for item in features['Item'].unique():
            try:
                item_data = features[features['Item'] == item].copy()
                
                # Prepare Prophet dataframe with required regressors
                prophet_df = pd.DataFrame({
                    'ds': pd.to_datetime(item_data['Year'], format='%Y'),
                    'y': item_data['Value'],
                    'temp_volatility': item_data['temp_volatility_mean'],
                    'production_risk': item_data['production_risk_score']
                })
                
                model = Prophet(
                    yearly_seasonality=True,
                    weekly_seasonality=False,
                    daily_seasonality=False,
                    seasonality_mode='multiplicative'
                )
                
                # Add the regressors
                model.add_regressor('temp_volatility')
                model.add_regressor('production_risk')
                
                # Fit the model
                model.fit(prophet_df)
                self.models['prophet'][item] = model
            except Exception as e:
                print(f"Warning: Prophet training failed for {item}: {e}")
                self.models['prophet'][item] = None
        
    def calculate_var_metrics(self, predictions, confidence_level=0.95):
        """Calculate Value at Risk metrics"""
        var = np.percentile(predictions, (1 - confidence_level) * 100)
        cvar = np.mean(predictions[predictions <= var])
        return var, cvar
    
    def calculate_risk_metrics(self, features, predictions):
        """Calculate comprehensive risk metrics"""
        # Calculate climate sensitivity first
        climate_sensitivity = features.get('climate_memory', 
            features['temp_volatility_mean'].fillna(0) * features['production_volatility'].fillna(0)
        ).mean()
        
        risk_metrics = {
            'volatility': np.std(predictions),
            'var_95': self.calculate_var_metrics(predictions)[0],
            'cvar_95': self.calculate_var_metrics(predictions)[1],
            'climate_risk_score': features['climate_risk_score'].fillna(0).mean(),
            'land_risk_score': features.get('land_climate_risk', pd.Series(0)).fillna(0).mean(),
            'production_risk_score': features['production_risk_score'].fillna(0).mean(),
            'trend': np.polyfit(np.arange(len(predictions)), predictions, 1)[0],
            'extreme_event_probability': features['extreme_temp_probability_max'].fillna(0).mean(),
            'climate_sensitivity': climate_sensitivity
        }
        return risk_metrics
    
    def predict_risk(self, features, forecast_periods=12):
        """Generate ensemble predictions and comprehensive risk analysis"""
        try:
            # Prepare data for prediction using consistent column names
            sequence_features = features[[
                'Value', 'temp_volatility_mean', 'trend_last', 'production_volatility',
                'climate_memory'
            ]].values
            sequence_features = self.scalers['features'].transform(sequence_features)
            
            aux_features = features[[
                'land_utilization', 'irrigation_efficiency', 'crop_diversity',
                'production_risk_score', 'land_climate_risk'
            ]].values
            aux_features = self.scalers['land'].transform(aux_features)
            
            # Prepare sequences for LSTM
            X_seq = []
            lookback = 12
            for i in range(len(sequence_features) - lookback):
                X_seq.append(sequence_features[i:(i + lookback)])
            X_seq = np.array(X_seq)
            
            predictions = []
            
            # LSTM predictions
            if self.models['lstm'] is not None:
                try:
                    lstm_pred = self.models['lstm'].predict([X_seq, aux_features[lookback:]])
                    predictions.append(lstm_pred)
                except Exception as e:
                    print(f"Warning: LSTM prediction failed: {e}")
            
            # XGBoost predictions
            if self.models['xgb'] is not None:
                try:
                    xgb_pred = self.models['xgb'].predict(
                        np.hstack([sequence_features, aux_features])
                    )
                    predictions.append(xgb_pred)
                except Exception as e:
                    print(f"Warning: XGBoost prediction failed: {e}")
            
            # Prophet predictions
            prophet_predictions = {}
            for item in features['Item'].unique():
                model = self.models['prophet'].get(item)
                if model is not None:
                    try:
                        future = pd.DataFrame({
                            'ds': pd.date_range(
                                start=pd.to_datetime(str(features['Year'].max())),
                                periods=forecast_periods,
                                freq='Y'
                            )
                        })
                        future['temp_volatility'] = features['temp_volatility_mean'].mean()
                        future['production_risk'] = features['production_risk_score'].mean()
                        forecast = model.predict(future)
                        prophet_predictions[item] = forecast['yhat'].values
                    except Exception as e:
                        print(f"Warning: Prophet prediction failed for {item}: {e}")
            
            if not predictions and not prophet_predictions:
                raise ValueError("No models were able to generate predictions")
            
            # Align predictions (use the minimum number of prediction points available)
            n_preds = forecast_periods
            aligned_predictions = []
            # Align model predictions
            for pred in predictions:
                aligned_predictions.append(pred[-n_preds:].flatten())
            
            # Align prophet predictions (if any)
            if prophet_predictions:
                prophet_mean = np.mean([
                    pred[-n_preds:] for pred in prophet_predictions.values()
                ], axis=0)
                aligned_predictions.append(prophet_mean)
            
            # Ensemble predictions with equal weights
            ensemble_pred = np.mean(aligned_predictions, axis=0)
            
            # Calculate risk metrics
            risk_metrics = self.calculate_risk_metrics(features, ensemble_pred)
            
            # Calculate prediction uncertainty
            if len(aligned_predictions) > 1:
                prediction_std = np.std(aligned_predictions, axis=0)
                risk_metrics['forecast_uncertainty'] = np.mean(prediction_std)
                risk_metrics['model_disagreement'] = stats.variation(prediction_std)
            else:
                risk_metrics['forecast_uncertainty'] = np.nan
                risk_metrics['model_disagreement'] = np.nan
            
            return ensemble_pred, risk_metrics, prophet_predictions
            
        except Exception as e:
            print(f"Error in risk prediction: {e}")
            raise

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
                'temperature_volatility': features['temp_volatility_mean'].mean(),
                'extreme_events_probability': features['extreme_temp_probability_max'].mean()
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
    
    # Save results to CSV
    results_df = pd.DataFrame({
        'Date': np.array(sorted(features['Year'].unique()))[-len(predictions):],
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
