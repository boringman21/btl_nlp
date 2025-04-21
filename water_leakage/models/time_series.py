#!/usr/bin/env python3
"""
Time series prediction module for water leakage detection.
Provides tools for forecasting flow and pressure values.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Union, Tuple, Any
from sklearn.preprocessing import StandardScaler
import logging
import os

# Create logger
logger = logging.getLogger(__name__)

class TimeSeriesPredictor:
    """
    Time series predictor for water leakage data.
    Uses LSTM models to predict next time steps based on historical data.
    """
    
    def __init__(self, window_size: int = 24):
        """
        Initialize the time series predictor.
        
        Args:
            window_size (int): Number of time steps to use for prediction (default: 24, which is 6 hours with 15-min intervals)
        """
        self.window_size = window_size
        self.scaler = None
        self.model = None
        self._is_fitted = False
    
    def _check_tensorflow(self) -> bool:
        """Check if TensorFlow is available and load it if so."""
        try:
            import tensorflow as tf
            from tensorflow.keras.models import Sequential
            from tensorflow.keras.layers import LSTM, Dense, Bidirectional
            from tensorflow.keras.callbacks import EarlyStopping
            
            self.tf = tf
            self.Sequential = Sequential
            self.LSTM = LSTM
            self.Dense = Dense
            self.Bidirectional = Bidirectional
            self.EarlyStopping = EarlyStopping
            
            return True
        except ImportError:
            logger.error("TensorFlow not available. Please install TensorFlow to use LSTM functionality.")
            return False
    
    def _create_sequences(self, data: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Create sequences for time series prediction.
        
        Args:
            data (np.ndarray): Input data array
            
        Returns:
            Tuple of X (input sequences) and y (target values)
        """
        X, y = [], []
        for i in range(len(data) - self.window_size):
            X.append(data[i:i+self.window_size])
            y.append(data[i+self.window_size])
        return np.array(X), np.array(y)
    
    def preprocess_data(self, df: pd.DataFrame) -> np.ndarray:
        """
        Preprocess data for time series prediction.
        
        Args:
            df (pd.DataFrame): Input DataFrame with time series data
            
        Returns:
            np.ndarray: Preprocessed data
        """
        # Ensure timestamp is index
        if 'Timestamp' in df.columns:
            df = df.set_index('Timestamp')
        
        # Interpolate missing values
        df = df.interpolate(method='time')
        
        # Add time features
        if df.index.dtype.kind == 'M':  # Check if index is datetime
            df['hour'] = df.index.hour
            df['day_of_week'] = df.index.dayofweek
        
        # Scale data
        self.scaler = StandardScaler()
        scaled_data = self.scaler.fit_transform(df)
        
        return scaled_data
    
    def fit(self, df: pd.DataFrame, epochs: int = 50, batch_size: int = 32, validation_split: float = 0.2) -> Dict[str, Any]:
        """
        Fit the LSTM model to the time series data.
        
        Args:
            df (pd.DataFrame): Input DataFrame with time series data
            epochs (int): Number of training epochs
            batch_size (int): Batch size for training
            validation_split (float): Fraction of data to use for validation
            
        Returns:
            Dict containing training history and validation metrics
        """
        if not self._check_tensorflow():
            return {"error": "TensorFlow not available"}
        
        # Preprocess data
        scaled_data = self.preprocess_data(df)
        
        # Create sequences
        X, y = self._create_sequences(scaled_data)
        
        if len(X) == 0:
            raise ValueError("Not enough data for the specified window size")
        
        # Split data
        split_idx = int((1 - validation_split) * len(X))
        X_train, y_train = X[:split_idx], y[:split_idx]
        X_test, y_test = X[split_idx:], y[split_idx:]
        
        # Build model
        self.model = self.Sequential([
            self.Bidirectional(self.LSTM(64, activation='relu', input_shape=(self.window_size, X.shape[2]))),
            self.Dense(X.shape[2])  # Predict all features
        ])
        
        self.model.compile(optimizer='adam', loss='mse')
        
        # Train model with early stopping
        history = self.model.fit(
            X_train, y_train,
            epochs=epochs,
            batch_size=batch_size,
            validation_data=(X_test, y_test),
            shuffle=False,
            callbacks=[self.EarlyStopping(patience=5)]
        )
        
        # Evaluate on test set
        X_test_reshaped = np.array([X_test[i] for i in range(len(X_test))])
        y_pred_scaled = self.model.predict(X_test_reshaped)
        y_pred = self.scaler.inverse_transform(y_pred_scaled)
        actual = self.scaler.inverse_transform(y_test)
        
        # Calculate MAE
        mae = np.mean(np.abs(actual - y_pred))
        
        self._is_fitted = True
        
        return {
            "history": history.history,
            "mae": mae,
            "actual": actual,
            "predicted": y_pred
        }
    
    def predict_next(self, df: pd.DataFrame) -> np.ndarray:
        """
        Predict the next time step (15 min) based on the last 6 hours (window_size) of data.
        
        Args:
            df (pd.DataFrame): Input DataFrame with most recent data (at least window_size records)
            
        Returns:
            np.ndarray: Predicted values for the next time step
        """
        if not self._is_fitted:
            raise ValueError("Model not fitted. Call fit() first.")
        
        if len(df) < self.window_size:
            raise ValueError(f"Not enough data. Need at least {self.window_size} records.")
        
        # Take only the last window_size records
        last_data = df.iloc[-self.window_size:]
        
        # Scale data
        scaled_input = self.scaler.transform(last_data)
        
        # Reshape for LSTM input
        input_sequence = scaled_input.reshape(1, self.window_size, -1)
        
        # Predict
        predicted_scaled = self.model.predict(input_sequence)
        
        # Inverse transform
        prediction = self.scaler.inverse_transform(predicted_scaled)[0]
        
        return prediction
    
    def save_model(self, path: str) -> None:
        """
        Save the trained model to disk.
        
        Args:
            path (str): Path to save the model
        """
        if not self._is_fitted:
            raise ValueError("Model not fitted. Call fit() first.")
        
        self.model.save(path)
        
        # Save scaler with joblib
        try:
            import joblib
            scaler_path = f"{path}_scaler.pkl"
            joblib.dump(self.scaler, scaler_path)
            logger.info(f"Scaler saved to {scaler_path}")
        except ImportError:
            logger.warning("joblib not available. Scaler not saved.")
    
    def load_model(self, model_path: str, scaler_path: str) -> None:
        """
        Load a pretrained model from disk.
        
        Args:
            model_path (str): Path to the model file
            scaler_path (str): Path to the scaler file
        """
        if not self._check_tensorflow():
            return
        
        # Load model
        self.model = self.tf.keras.models.load_model(model_path)
        
        # Load scaler
        try:
            import joblib
            self.scaler = joblib.load(scaler_path)
            self._is_fitted = True
            logger.info(f"Model loaded from {model_path} and scaler from {scaler_path}")
        except ImportError:
            logger.error("joblib not available. Cannot load scaler.")


def predict_next_tick(df: pd.DataFrame, sensor_id: str, feature_cols: List[str], model_path: Optional[str] = None) -> Dict[str, float]:
    """
    High-level function to predict the next 15-minute tick for a given sensor.
    
    Args:
        df (pd.DataFrame): Input DataFrame containing sensor data
        sensor_id (str): Sensor ID to make predictions for
        feature_cols (List[str]): List of feature columns to include in prediction
        model_path (str, optional): Path to load a pre-trained model from
        
    Returns:
        Dict with predictions for each feature
    """
    # Filter dataframe for the specific sensor and features
    sensor_cols = [f"{sensor_id}_{col}" for col in feature_cols]
    
    # Check if columns exist
    missing_cols = [col for col in sensor_cols if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing columns: {missing_cols}")
    
    # Extract data
    sensor_df = df[['Timestamp'] + sensor_cols].copy()
    
    # Convert timestamp to datetime if it's not already
    sensor_df['Timestamp'] = pd.to_datetime(sensor_df['Timestamp'])
    sensor_df = sensor_df.set_index('Timestamp')
    
    # Create predictor
    predictor = TimeSeriesPredictor(window_size=24)  # 24 x 15min = 6 hours
    
    if model_path:
        try:
            predictor.load_model(f"{model_path}/{sensor_id}_model", f"{model_path}/{sensor_id}_scaler.pkl")
        except FileNotFoundError:
            # Train a new model if no saved model exists
            logger.info(f"No saved model found for {sensor_id}. Training a new model.")
            predictor.fit(sensor_df)
            
            # Save the model if a path was provided
            os.makedirs(model_path, exist_ok=True)
            predictor.save_model(f"{model_path}/{sensor_id}_model")
    else:
        # Just train a new model
        predictor.fit(sensor_df)
    
    # Predict next tick
    prediction = predictor.predict_next(sensor_df)
    
    # Create result dictionary
    result = {}
    for i, col in enumerate(sensor_cols):
        feature = col.split('_', 1)[1]  # Extract original feature name
        result[feature] = float(prediction[i])
    
    return result 