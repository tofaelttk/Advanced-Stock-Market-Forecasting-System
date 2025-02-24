import numpy as np
import pandas as pd
import plotly.graph_objects as go
from sklearn.preprocessing import RobustScaler
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_absolute_error
from keras.models import Sequential
from keras.layers import Dense, LSTM, Dropout, BatchNormalization, Input, Bidirectional
from keras.callbacks import EarlyStopping, ReduceLROnPlateau
from prophet import Prophet
import xgboost as xgb
from datetime import datetime, timedelta
import ta
from arch import arch_model
from bayes_opt import BayesianOptimization
from scipy import stats
import logging

class EnhancedStockPredictor:
    def __init__(self, file_path):
        self.file_path = file_path
        self.look_back = 400
        self.scaler = RobustScaler()
        self.volatility_window = 30
        self.confidence_level = 0.99
        self.optimization_iterations = 15
        self.logger = self._setup_logger()
        self.z_score = stats.norm.ppf(self.confidence_level)

    def _setup_logger(self):
        logger = logging.getLogger('StockPredictor')
        logger.setLevel(logging.INFO)
        handler = logging.StreamHandler()
        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        return logger

    def load_data(self):
        try:
            df = pd.read_csv(self.file_path)
            df.rename(columns={'DATE': 'Date', 'CLOSING PRICE': 'Close'}, inplace=True)
            
            df['Close'] = df['Close'].str.replace('[^\d.]', '', regex=True).astype(float)
            df['Date'] = pd.to_datetime(df['Date'])
            
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            df[numeric_cols] = df[numeric_cols].ffill().bfill()
            
            df = df.sort_values('Date').reset_index(drop=True)
            return df
        except Exception as e:
            self.logger.error(f"Error loading data: {str(e)}")
            raise

    def add_enhanced_features(self, df):
        for window in [7, 21, 50]:
            df[f'MA{window}'] = ta.trend.sma_indicator(df['Close'], window=window)
        
        df['RSI'] = ta.momentum.rsi(df['Close'])
        df['MACD'] = ta.trend.macd_diff(df['Close'])
        df['BB_upper'] = ta.volatility.bollinger_hband(df['Close'])
        df['BB_lower'] = ta.volatility.bollinger_lband(df['Close'])
        
        returns = df['Close'].pct_change()
        df['volatility_30'] = returns.rolling(30).std()
        df['volatility_regime'] = returns.rolling(90).std().apply(
            lambda x: 2 if x > 0.05 else 1 if x > 0.03 else 0
        )
        
        df['trend_strength'] = abs(df['Close'] - df['MA50']) / df['MA50']
        df['momentum_ratio'] = df['MA7'] / df['MA21']
        
        df = df.ffill().bfill()
        return df

    def prepare_lstm_data(self, data):
        feature_columns = ['Close', 'MA50', 'RSI', 'MACD', 'volatility_30', 'trend_strength']
        scaled_data = self.scaler.fit_transform(data[feature_columns])
        
        X, y = [], []
        for i in range(self.look_back, len(scaled_data)):
            X.append(scaled_data[i-self.look_back:i, :])
            y.append(scaled_data[i, 0])  # Single target for Close price
            
        X, y = np.array(X), np.array(y)
        split_idx = int(len(X) * 0.8)
        return X[:split_idx], X[split_idx:], y[:split_idx], y[split_idx:]

    def build_lstm_model(self, input_shape):
        model = Sequential([
            Input(shape=input_shape),
            Bidirectional(LSTM(256, return_sequences=True, activation='tanh')),
            BatchNormalization(),
            Dropout(0.3),
            Bidirectional(LSTM(192, return_sequences=True, activation='tanh')),
            BatchNormalization(),
            Dropout(0.3),
            Bidirectional(LSTM(128, return_sequences=False, activation='tanh')),
            BatchNormalization(),
            Dense(64, activation='relu'),
            Dropout(0.2),
            Dense(32, activation='relu'),
            Dense(1, activation='linear')  # Fixed to single output
        ])
        model.compile(optimizer='adam', loss='huber', metrics=['mae'])
        return model

    def evaluate_model(self, model, X, y):
        """Critical missing method added back"""
        tscv = TimeSeriesSplit(n_splits=5)
        scores = []
        
        for train_idx, test_idx in tscv.split(X):
            X_train, X_test = X[train_idx], X[test_idx]
            y_train, y_test = y[train_idx], y[test_idx]
            
            model.fit(X_train, y_train)
            scores.append(mean_absolute_error(y_test, model.predict(X_test)))
            
        return np.mean(scores)

    def optimize_xgboost(self, X_train, y_train):
        def xgb_cv(max_depth, learning_rate, n_estimators, subsample):
            params = {
                'max_depth': int(max_depth),
                'learning_rate': learning_rate,
                'n_estimators': int(n_estimators),
                'subsample': subsample,
                'eval_metric': 'mae'
            }
            model = xgb.XGBRegressor(**params)
            return -self.evaluate_model(model, X_train, y_train)  # Now using the existing method
        
        optimizer = BayesianOptimization(
            f=xgb_cv,
            pbounds={
                'max_depth': (3, 10),
                'learning_rate': (0.001, 0.3),
                'n_estimators': (500, 2000),
                'subsample': (0.6, 1)
            },
            random_state=42
        )
        optimizer.maximize(init_points=5, n_iter=int(self.optimization_iterations))
        return {k: int(v) if k in ['max_depth', 'n_estimators'] else v 
                for k, v in optimizer.max['params'].items()}

    def calculate_volatility_forecast(self, returns, horizon=30):
        try:
            model = arch_model(returns.dropna()*100, vol='GARCH', p=1, q=1)  # Scaled returns
            model_fit = model.fit(disp='off')
            forecast = model_fit.forecast(horizon=horizon)
            return float(np.sqrt(forecast.variance.values[-1]).mean() / 100)
        except Exception as e:
            self.logger.warning(f"GARCH model failed: {str(e)}")
            return returns.std() * np.sqrt(252)

    def forecast(self, future_days=1305):
        try:
            future_days = int(future_days)
            df = self.load_data()
            df = self.add_enhanced_features(df)
            returns = np.log(df['Close']).diff().dropna()
            
            volatility_forecast = self.calculate_volatility_forecast(returns)
            self.logger.info(f"Adjusted Volatility Forecast: {volatility_forecast:.4f}")
            
            # LSTM Model with single output
            X_train_lstm, X_test_lstm, y_train_lstm, y_test_lstm = self.prepare_lstm_data(df)
            lstm_model = self.build_lstm_model((X_train_lstm.shape[1], X_train_lstm.shape[2]))
            
            history = lstm_model.fit(
                X_train_lstm, y_train_lstm,
                validation_data=(X_test_lstm, y_test_lstm),
                epochs=200,
                batch_size=32,
                callbacks=[
                    EarlyStopping(patience=20, restore_best_weights=True),
                    ReduceLROnPlateau(factor=0.5, patience=10)
                ],
                verbose=0
            )
            
            # Calculate LSTM prediction intervals
            lstm_val_pred = lstm_model.predict(X_test_lstm).flatten()
            lstm_mae = mean_absolute_error(y_test_lstm, lstm_val_pred)
            
            # Prophet Model
            prophet_df = df.rename(columns={'Date': 'ds', 'Close': 'y'})[['ds', 'y']]
            prophet_model = Prophet(yearly_seasonality=20, weekly_seasonality=True, 
                                  interval_width=self.confidence_level)
            prophet_model.fit(prophet_df)
            
            # XGBoost Model
            feature_cols = [col for col in df.columns if col not in ['Date', 'Close']]
            X = df[feature_cols].shift(1).dropna()
            y = df['Close'][1:]
            
            best_params = self.optimize_xgboost(X.values, y.values)
            xgb_model = xgb.XGBRegressor(**best_params)
            xgb_model.fit(X.values, y.values)
            
            # Generate predictions
            predictions = {
                'lstm': self._forecast_lstm(lstm_model, df, future_days, lstm_mae),
                'prophet': self._forecast_prophet(prophet_model, future_days),
                'xgboost': self._forecast_xgboost(xgb_model, df, feature_cols, future_days)
            }
            
            # Ensemble predictions
            final = self.ensemble_predictions(predictions, volatility_forecast, df)
            self.plot_predictions(df, final, future_days)
            
            return final
            
        except Exception as e:
            self.logger.error(f"Forecasting failed: {str(e)}")
            raise

    def _forecast_lstm(self, model, data, periods, mae):
        periods = int(periods)
        feature_columns = ['Close', 'MA50', 'RSI', 'MACD', 'volatility_30', 'trend_strength']
        last_sequence = self.scaler.transform(data[feature_columns].tail(self.look_back))
        
        avg_predictions = []
        for _ in range(periods):
            pred = model.predict(last_sequence[-self.look_back:].reshape(1, self.look_back, -1))
            avg_pred = pred.flatten()[0]
            new_row = np.append(avg_pred, last_sequence[-1, 1:]).reshape(1, -1)
            last_sequence = np.vstack([last_sequence, new_row])
            avg_predictions.append(avg_pred)
        
        predicted_sequence = last_sequence[-periods:]
        predicted_sequence_inverse = self.scaler.inverse_transform(predicted_sequence)
        lstm_avg = predicted_sequence_inverse[:, 0]
        
        # Create uncertainty bands using validation MAE
        lstm_low = lstm_avg - (mae * self.z_score)
        lstm_high = lstm_avg + (mae * self.z_score)
        
        return {'low': lstm_low, 'avg': lstm_avg, 'high': lstm_high}

    def _forecast_prophet(self, model, periods):
        periods = int(periods)
        future = model.make_future_dataframe(periods=periods)
        forecast = model.predict(future)
        return {
            'low': forecast['yhat_lower'].iloc[-periods:].values,
            'avg': forecast['yhat'].iloc[-periods:].values,
            'high': forecast['yhat_upper'].iloc[-periods:].values
        }

    def _forecast_xgboost(self, model, data, features, periods):
        periods = int(periods)
        current_data = data.copy()
        avg_predictions = []
        errors = []
        
        # First pass for average predictions
        for _ in range(periods):
            current_features = self.add_enhanced_features(current_data)
            X = current_features[features].iloc[[-1]].values
            pred = model.predict(X)[0]
            avg_predictions.append(pred)
            
            new_date = current_data['Date'].iloc[-1] + timedelta(days=1)
            new_row = {'Date': new_date, 'Close': pred}
            current_data = pd.concat([current_data, pd.DataFrame([new_row])], ignore_index=True)
        
        # Second pass for error estimation
        val_pred = model.predict(current_data[features].iloc[:-periods])
        val_true = current_data['Close'].iloc[:-periods]
        mae = mean_absolute_error(val_true, val_pred)
        
        return {
            'low': np.array(avg_predictions) - (mae * self.z_score),
            'avg': np.array(avg_predictions),
            'high': np.array(avg_predictions) + (mae * self.z_score)
        }

    def ensemble_predictions(self, predictions, volatility, historical_df):
        min_length = min(
            len(predictions['lstm']['avg']),
            len(predictions['prophet']['avg']),
            len(predictions['xgboost']['avg'])
        )
        
        # Dynamic weighting based on volatility
        base_weights = {
            'lstm': 0.5 - (volatility * 5),
            'prophet': 0.3,
            'xgboost': 0.2 + (volatility * 5)
        }
        base_weights = {k: np.clip(v, 0.1, 0.7) for k, v in base_weights.items()}
        total_weight = sum(base_weights.values())
        
        combined = {'low': [], 'avg': [], 'high': []}
        for i in range(min_length):
            # Weighted average combination
            lstm_w = base_weights['lstm'] / total_weight
            prophet_w = base_weights['prophet'] / total_weight
            xgb_w = base_weights['xgboost'] / total_weight
            
            combined['avg'].append(
                predictions['lstm']['avg'][i] * lstm_w +
                predictions['prophet']['avg'][i] * prophet_w +
                predictions['xgboost']['avg'][i] * xgb_w
            )
            
            combined['low'].append(
                predictions['lstm']['low'][i] * lstm_w +
                predictions['prophet']['low'][i] * prophet_w +
                predictions['xgboost']['low'][i] * xgb_w
            )
            
            combined['high'].append(
                predictions['lstm']['high'][i] * lstm_w +
                predictions['prophet']['high'][i] * prophet_w +
                predictions['xgboost']['high'][i] * xgb_w
            )
        
        return {
            'date': pd.date_range(start=historical_df['Date'].iloc[-1] + timedelta(days=1), periods=min_length),
            'low': np.array(combined['low']),
            'average': np.array(combined['avg']),
            'high': np.array(combined['high'])
        }

    def plot_predictions(self, historical, forecast, periods):
        fig = go.Figure()
        
        # Historical data
        fig.add_trace(go.Scatter(
            x=historical['Date'],
            y=historical['Close'],
            name='Historical Prices',
            line=dict(color='#1f77b4')
        ))
        
        # Forecast lines
        fig.add_trace(go.Scatter(
            x=forecast['date'],
            y=forecast['average'],
            name='Average Forecast',
            line=dict(color='#ff7f0e', width=2)
        ))
        
        fig.add_trace(go.Scatter(
            x=forecast['date'],
            y=forecast['high'],
            name='High Estimate',
            line=dict(color='#2ca02c', width=1, dash='dot')
        ))
        
        # fig.add_trace(go.Scatter(
        #     x=forecast['date'],
        #     y=forecast['low'],
        #     name='Low Estimate',
        #     line=dict(color='#d62728', width=1, dash='dot')
        # ))
        
        # Confidence interval
        fig.add_trace(go.Scatter(
            x=forecast['date'].tolist() + forecast['date'].tolist()[::-1],
            y=np.concatenate([forecast['high'], forecast['low'][::-1]]),
            fill='toself',
            fillcolor='rgba(255,127,14,0.2)',
            line=dict(color='rgba(255,255,255,0)'),
            name=f'{self.confidence_level:.0%} Confidence'
        ))
        
        fig.update_layout(
            title='Enhanced Stock Price Forecast with Probabilistic Bands',
            xaxis_title='Date',
            yaxis_title='Price',
            template='plotly_white',
            hovermode='x unified',
            legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1)
        )
        
        fig.show()

if __name__ == "__main__":
    try:
        print("Starting enhanced stock price prediction...")
        predictor = EnhancedStockPredictor('/Users/tofael/CRStocks.csv')
        predictions = predictor.forecast(future_days=1305)
        print("Enhanced prediction completed successfully!")
    except Exception as e:
        print(f"An error occurred: {e}")













# import numpy as np
# import pandas as pd
# import plotly.graph_objects as go
# from sklearn.preprocessing import MinMaxScaler, RobustScaler
# from sklearn.model_selection import train_test_split
# from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
# from keras.models import Sequential
# from keras.layers import Dense, LSTM, Dropout, BatchNormalization, Input, Bidirectional
# from keras.callbacks import EarlyStopping, ReduceLROnPlateau
# from prophet import Prophet
# import xgboost as xgb
# from datetime import datetime, timedelta
# import ta
# from scipy import stats
# from sklearn.ensemble import IsolationForest
# from statsmodels.tsa.stattools import adfuller
# from statsmodels.tsa.seasonal import seasonal_decompose

# class EnhancedStockPredictor:
#     def __init__(self, file_path):
#         self.file_path = file_path
#         self.look_back = 400
#         self.scaler = MinMaxScaler(feature_range=(0, 1))
#         self.feature_scaler = RobustScaler()
#         self.volatility_window = 30
#         self.confidence_level = 0.99

#     def load_data(self):
#         try:
#             df = pd.read_csv(self.file_path)
#             df.rename(columns={'DATE': 'Date', 'CLOSING PRICE': 'Close'}, inplace=True)
#             df['Close'] = df['Close'].str.replace('[^\d.]', '', regex=True).astype(float)
#             df['Date'] = pd.to_datetime(df['Date'])

#             numeric_cols = df.select_dtypes(include=[np.number]).columns

#             if df[numeric_cols].isnull().values.any():
#                 print("Handling NaN values in data...")
#                 df[numeric_cols] = df[numeric_cols].fillna(method='ffill').fillna(method='bfill')
#             if np.isinf(df[numeric_cols].values).any():
#                 print("Handling infinite values in data...")
#                 df[numeric_cols] = df[numeric_cols].replace([np.inf, -np.inf], np.nan).fillna(method='ffill').fillna(method='bfill')

#             return df
#         except Exception as e:
#             print(f"Error loading data: {e}")
#             raise

#     def add_technical_indicators(self, df):
#         df['MA7'] = ta.trend.sma_indicator(df['Close'], window=7)
#         df['MA21'] = ta.trend.sma_indicator(df['Close'], window=21)
#         df['MA50'] = ta.trend.sma_indicator(df['Close'], window=50)
#         df['RSI'] = ta.momentum.rsi(df['Close'])
#         df['MACD'] = ta.trend.macd_diff(df['Close'])
#         df['BB_upper'] = ta.volatility.bollinger_hband(df['Close'])
#         df['BB_lower'] = ta.volatility.bollinger_lband(df['Close'])
#         df['ROC'] = ta.momentum.roc(df['Close'])
#         df['TSI'] = ta.momentum.tsi(df['Close'])
#         df.fillna(method='ffill', inplace=True)
#         df.fillna(method='bfill', inplace=True)
#         return df

#     def add_enhanced_features(self, df):
#         df = self.add_technical_indicators(df)

#         for window in [5, 10, 20, 30]:
#             df[f'momentum_{window}'] = df['Close'].pct_change(window)
#             df[f'acceleration_{window}'] = df[f'momentum_{window}'].diff()

#         returns = df['Close'].pct_change()
#         for window in [5, 10, 20, 30]:
#             df[f'volatility_{window}'] = returns.rolling(window=window).std()

#         df['trend_strength'] = abs(df['Close'] - df['MA50']) / df['MA50']

#         decomposition = seasonal_decompose(df['Close'], period=30, extrapolate_trend='freq')
#         df['seasonal'] = decomposition.seasonal
#         df['trend'] = decomposition.trend
#         df['residual'] = decomposition.resid


#         df.ffill(inplace=True)
#         df.bfill(inplace=True)


#         return df

#     def prepare_lstm_data(self, data, split_ratio=0.8):
#         # Univariate setup using only Close price
#         feature_columns = ['Close']
#         scaled_data = self.scaler.fit_transform(data[feature_columns])
        
#         X, y = [], []
#         for i in range(self.look_back, len(scaled_data)):
#             X.append(scaled_data[i-self.look_back:i, 0])
#             y.append(scaled_data[i, 0])

#         X, y = np.array(X), np.array(y)
#         X = np.reshape(X, (X.shape[0], X.shape[1], 1))
#         split_idx = int(len(X) * split_ratio)
#         X_train, X_test = X[:split_idx], X[split_idx:]
#         y_train, y_test = y[:split_idx], y[split_idx:]

#         return X_train, X_test, y_train, y_test

#     def build_lstm_model(self):
#         model = Sequential([
#             Input(shape=(self.look_back, 1)),
#             Bidirectional(LSTM(256, return_sequences=True, activation='tanh')),
#             BatchNormalization(),
#             Dropout(0.3),
#             Bidirectional(LSTM(192, return_sequences=True, activation='tanh')),
#             BatchNormalization(),
#             Dropout(0.3),
#             Bidirectional(LSTM(128, return_sequences=False, activation='tanh')),
#             BatchNormalization(),
#             Dense(64, activation='relu'),
#             Dropout(0.2),
#             Dense(32, activation='relu'),
#             Dense(1, activation='linear')
#         ])
#         model.compile(optimizer='adam', loss='huber', metrics=['mae'])
#         return model

#     def prepare_prophet_data(self, df):
#         prophet_df = df.rename(columns={'Date': 'ds', 'Close': 'y'})
#         prophet_df['yearly_seasonality'] = prophet_df.ds.dt.year
#         prophet_df['monthly_seasonality'] = prophet_df.ds.dt.month
#         return prophet_df

#     def forecast(self, future_days=750):
#         try:
#             df = self.load_data()
#             df = self.add_enhanced_features(df)

#             # Calculate historical volatility
#             returns = np.log(df['Close'] / df['Close'].shift(1))
#             historical_volatility = returns.std() * np.sqrt(252)

#             predictions = {}

#             print("Training LSTM model...")
#             X_train, X_test, y_train, y_test = self.prepare_lstm_data(df)
#             lstm_model = self.build_lstm_model()
#             callbacks = [
#                 EarlyStopping(monitor='val_loss', patience=20, restore_best_weights=True),
#                 ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=10, min_lr=0.0001)
#             ]

#             lstm_model.fit(
#                 X_train, y_train,
#                 epochs=200,
#                 batch_size=32,
#                 validation_split=0.2,
#                 callbacks=callbacks,
#                 verbose=1
#             )
#             lstm_predictions = self.forecast_lstm(lstm_model, df, future_days)
#             predictions['lstm'] = lstm_predictions

#             print("\nTraining Prophet model...")
#             prophet_df = self.prepare_prophet_data(df)
#             prophet_model = Prophet(yearly_seasonality=20, weekly_seasonality=True, daily_seasonality=True)
#             prophet_model.add_regressor('yearly_seasonality')
#             prophet_model.add_regressor('monthly_seasonality')
#             prophet_model.fit(prophet_df)

#             future_dates = prophet_model.make_future_dataframe(periods=future_days)
#             future_dates['yearly_seasonality'] = future_dates.ds.dt.year
#             future_dates['monthly_seasonality'] = future_dates.ds.dt.month

#             prophet_forecast = prophet_model.predict(future_dates)
#             predictions['prophet'] = prophet_forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].iloc[-future_days:]

#             print("\nTraining XGBoost model...")
#             xgb_predictions = self.train_multiple_xgboost(df, future_days)
#             predictions['xgboost'] = xgb_predictions

#             print("\nCreating ensemble predictions...")
#             final_predictions = self.ensemble_predictions_with_uncertainty(
#                 predictions, 
#                 weights={'lstm': 0.35, 'prophet': 0.35, 'xgboost': 0.30},
#                 historical_volatility=historical_volatility
#             )

#             self.plot_predictions_with_bands(df, final_predictions, future_days)
#             return final_predictions

#         except Exception as e:
#             print(f"Error in forecast: {e}")
#             raise

#     def forecast_lstm(self, model, data, future_days):
#         last_sequence = self.scaler.transform(data[['Close']].tail(self.look_back).values)
#         future_predictions = []

#         current_sequence = last_sequence.copy()
#         for _ in range(future_days):
#             current_sequence_reshaped = current_sequence[-self.look_back:].reshape(1, self.look_back, 1)
#             next_pred = model.predict(current_sequence_reshaped, verbose=0)
#             future_predictions.append(next_pred[0, 0])
#             current_sequence = np.vstack([current_sequence, next_pred])

#         future_predictions = self.scaler.inverse_transform(
#             np.array(future_predictions).reshape(-1, 1)
#         ).flatten()

#         return future_predictions

#     def train_multiple_xgboost(self, df, future_days):
#         quantiles = [0.15, 0.5, 0.85]
#         models = []
#         predictions = {q: [] for q in ['low', 'average', 'high']}

#         recent_trend = df['Close'].pct_change(periods=30).mean()
#         trend_multiplier = 1 + max(0, recent_trend)
        
#         feature_cols = [col for col in df.columns if col not in ['Date', 'Close']]
#         X = df[feature_cols]
#         y = df['Close']

#         X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

#         for q in quantiles:
#             model = xgb.XGBRegressor(
#                 objective='reg:squarederror',
#                 n_estimators=1500,
#                 learning_rate=0.008,
#                 max_depth=7,
#                 min_child_weight=3,
#                 subsample=0.8,
#                 colsample_bytree=0.8,
#                 gamma=0.1,
#                 random_state=42
#             )
#             model.fit(X_train, y_train, eval_set=[(X_test, y_test)], verbose=False)
#             models.append(model)

#         current_data = df.copy()
#         trend_factor = trend_multiplier
#         for i in range(future_days):
#             current_features = self.add_technical_indicators(current_data)
#             feature_data = current_features[feature_cols].iloc[[-1]]

#             trend_factor *= 1.0005

#             for j, q in enumerate(['low', 'average', 'high']):
#                 base_pred = models[j].predict(feature_data)[0]
#                 predictions[q].append(base_pred * trend_factor)

#             if not isinstance(current_data.index[-1], pd.Timestamp):
#                 current_data.index = pd.to_datetime(current_data.index)

#             new_row = pd.DataFrame({
#                 'Close': [predictions['average'][-1]]
#             }, index=[current_data.index[-1] + pd.Timedelta(days=1)])

#             current_data = pd.concat([current_data, new_row])

#         return predictions


#     def ensemble_predictions_with_uncertainty(self, predictions, weights, historical_volatility):
#         ensemble = {
#             'low': (weights['lstm'] * predictions['lstm'] * 0.95 +
#                     weights['prophet'] * predictions['prophet']['yhat_lower'].values +
#                     weights['xgboost'] * np.array(predictions['xgboost']['low'])),
            
#             'average': (weights['lstm'] * predictions['lstm'] +
#                        weights['prophet'] * predictions['prophet']['yhat'].values +
#                        weights['xgboost'] * np.array(predictions['xgboost']['average'])),
            
#             'high': (weights['lstm'] * predictions['lstm'] * 1.05 +
#                      weights['prophet'] * predictions['prophet']['yhat_upper'].values +
#                      weights['xgboost'] * np.array(predictions['xgboost']['high']))
#         }

#         # Apply volatility-based smoothing
#         for key in ensemble:
#             ensemble[key] = np.array(ensemble[key]) * (1 + historical_volatility/252)

#         return ensemble

#     def plot_predictions_with_bands(self, historical_data, predictions, future_days):
#         future_dates = pd.date_range(
#             start=historical_data['Date'].iloc[-1],
#             periods=future_days,
#             freq='B'
#         )

#         fig = go.Figure()

#         fig.add_trace(go.Scatter(
#             x=historical_data['Date'],
#             y=historical_data['Close'],
#             mode='lines',
#             name='Historical Data',
#             line=dict(color='royalblue', width=1)
#         ))

#         fig.add_trace(go.Scatter(
#             x=future_dates,
#             y=predictions['high'],
#             mode='lines',
#             name='High',
#             line=dict(color='firebrick', width=1)
#         ))

#         fig.add_trace(go.Scatter(
#             x=future_dates,
#             y=predictions['average'],
#             mode='lines',
#             name='Average',
#             line=dict(color='green', width=1)
#         ))

#         fig.add_trace(go.Scatter(
#             x=future_dates,
#             y=predictions['low'],
#             mode='lines',
#             name='Worst',
#             line=dict(color='orange', width=1)
#         ))

#         fig.update_layout(
#             title={
#                 'text': 'Stock Price Forecast with Prediction Bands',
#                 'y': 0.9,
#                 'x': 0.5,
#                 'xanchor': 'center',
#                 'yanchor': 'top'
#             },
#             xaxis=dict(
#                 title='Date',
#                 showgrid=True,
#                 gridcolor='lightgray',
#                 zeroline=False,
#                 showline=True,
#                 linecolor='black'
#             ),
#             yaxis=dict(
#                 title='Price ($)',
#                 showgrid=True,
#                 gridcolor='lightgray',
#                 zeroline=False,
#                 showline=True,
#                 linecolor='black'
#             ),
#             template='none',
#             hovermode='x',
#             plot_bgcolor='white',
#             legend=dict(
#                 title='Legend',
#                 orientation='h',
#                 yanchor='bottom',
#                 y=1.02,
#                 xanchor='right',
#                 x=1
#             ),
#             margin=dict(l=40, r=40, t=60, b=40)
#         )

#         fig.show()

# # Usage
# if __name__ == "__main__":
#     try:
#         print("Starting enhanced stock price prediction...")
#         predictor = EnhancedStockPredictor('/Users/tofael/CRStocks.csv')
#         predictions = predictor.forecast(future_days=1310)
#         print("Enhanced prediction completed successfully!")
#     except Exception as e:
#         print(f"An error occurred: {e}")
