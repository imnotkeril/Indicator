"""
Hybrid model for volatility prediction.
Combines traditional time-series models, ML models, and market regimes.
"""
import os
import sys
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import ElasticNet
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# Ensure project root is in Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, project_root)

# Conditional imports for deep learning libraries
try:
    import tensorflow as tf
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import LSTM, Dense, Dropout
    TENSORFLOW_AVAILABLE = True
except ImportError:
    TENSORFLOW_AVAILABLE = False
    Sequential = None
    LSTM = None
    Dense = None
    Dropout = None

import joblib
from datetime import datetime

from crypto_volatility_indicator.utils.logger import get_logger
from crypto_volatility_indicator.engine.models.calibration import ModelCalibrator

logger = get_logger(__name__)


class VolatilityPredictionModel:
    """
    A hybrid model for predicting volatility that combines traditional 
    time-series analysis with machine learning models.
    """

    def __init__(self, config=None):
        """
        Initialize the model.

        Parameters:
        -----------
        config : dict, optional
            Configuration parameters for the model
        """
        self.config = config or {}
        self.models = {}
        self.ensemble_weights = {}
        self.scaler = StandardScaler()
        self.calibrator = ModelCalibrator(
            lookback_period=self.config.get('lookback_period', 30),
            recalibrate_freq=self.config.get('recalibrate_freq', 7)
        )
        self.feature_importance = {}
        self.default_features = [
            'micro_vol', 'meso_vol', 'macro_vol',
            'volume', 'rsi', 'hurst_exponent',
            'fractal_dimension', 'implied_vol',
            'funding_rate', 'open_interest'
        ]

    def initialize_models(self):
        """Initialize the component models."""
        # Random Forest model
        self.models['rf'] = Pipeline([
            ('scaler', StandardScaler()),
            ('model', RandomForestRegressor(
                n_estimators=self.config.get('rf_n_estimators', 100),
                max_depth=self.config.get('rf_max_depth', 10),
                random_state=42
            ))
        ])

        # Gradient Boosting model
        self.models['gb'] = Pipeline([
            ('scaler', StandardScaler()),
            ('model', GradientBoostingRegressor(
                n_estimators=self.config.get('gb_n_estimators', 100),
                learning_rate=self.config.get('gb_learning_rate', 0.1),
                max_depth=self.config.get('gb_max_depth', 3),
                random_state=42
            ))
        ])

        # ElasticNet model
        self.models['en'] = Pipeline([
            ('scaler', StandardScaler()),
            ('model', ElasticNet(
                alpha=self.config.get('en_alpha', 0.5),
                l1_ratio=self.config.get('en_l1_ratio', 0.5),
                random_state=42
            ))
        ])

        # LSTM model (if configured)
        if self.config.get('use_lstm', False):
            self._initialize_lstm_model()

        # Set initial ensemble weights
        self._set_initial_ensemble_weights()

        logger.info("Models initialized")

    def _initialize_lstm_model(self):
        if not TENSORFLOW_AVAILABLE:
            logger.warning("TensorFlow not available. LSTM model cannot be initialized.")
            return

        # Get sequence length from config
        seq_length = self.config.get('lstm_seq_length', 10)
        num_features = len(self.default_features)

        # Create the LSTM model
        model = Sequential()
        model.add(LSTM(
            units=self.config.get('lstm_units', 50),
            activation='tanh',
            input_shape=(seq_length, num_features),
            return_sequences=True
        ))
        model.add(Dropout(self.config.get('lstm_dropout', 0.2)))
        model.add(LSTM(units=self.config.get('lstm_units_2', 30)))
        model.add(Dropout(self.config.get('lstm_dropout_2', 0.2)))
        model.add(Dense(1))

        model.compile(
            optimizer=self.config.get('lstm_optimizer', 'adam'),
            loss=self.config.get('lstm_loss', 'mse')
        )

        self.models['lstm'] = {
            'model': model,
            'seq_length': seq_length
        }

        logger.info("LSTM model initialized")

    def _set_initial_ensemble_weights(self):
        """Set initial weights for ensemble prediction."""
        models = list(self.models.keys())
        # Default to equal weights
        weights = np.ones(len(models)) / len(models)

        # If custom weights provided in config, use those
        if 'ensemble_weights' in self.config:
            config_weights = self.config['ensemble_weights']
            for i, model_name in enumerate(models):
                if model_name in config_weights:
                    weights[i] = config_weights[model_name]

        # Normalize weights
        weights = weights / np.sum(weights)

        self.ensemble_weights = {
            model_name: weight for model_name, weight
            in zip(models, weights)
        }

        logger.info(f"Ensemble weights set: {self.ensemble_weights}")

    def prepare_features(self, data, target_column=None):
        """
        Prepare features for the model.

        Parameters:
        -----------
        data : pd.DataFrame
            Input data
        target_column : str, optional
            Name of the target column

        Returns:
        --------
        X : pd.DataFrame
            Feature matrix
        y : pd.Series or None
            Target values (if target_column is provided)
        """
        # Get feature columns from config or use defaults
        feature_cols = self.config.get('feature_columns', self.default_features)

        # Filter to include only columns that exist in the data
        available_features = [col for col in feature_cols if col in data.columns]

        if not available_features:
            raise ValueError("No valid features found in the data")

        if len(available_features) < len(feature_cols):
            missing = set(feature_cols) - set(available_features)
            logger.warning(f"Missing features: {missing}")

        X = data[available_features].copy()

        # Handle missing values
        X = X.fillna(method='ffill').fillna(method='bfill')

        # Extract target if provided
        y = None
        if target_column and target_column in data.columns:
            y = data[target_column]
            y = y.fillna(method='ffill').fillna(method='bfill')

        return X, y

    def prepare_lstm_sequences(self, X, y=None):
        """
        Prepare sequences for LSTM model.

        Parameters:
        -----------
        X : pd.DataFrame
            Feature matrix
        y : pd.Series, optional
            Target values

        Returns:
        --------
        X_seq : np.ndarray
            Sequence data for LSTM
        y_seq : np.ndarray or None
            Target values for each sequence
        """
        if 'lstm' not in self.models:
            raise ValueError("LSTM model is not initialized")

        seq_length = self.models['lstm']['seq_length']

        # Convert to numpy arrays
        X_values = X.values
        n_samples = len(X_values) - seq_length

        if n_samples <= 0:
            raise ValueError(f"Not enough samples for sequence length {seq_length}")

        # Create sequences
        X_seq = np.zeros((n_samples, seq_length, X.shape[1]))
        for i in range(n_samples):
            X_seq[i] = X_values[i:i + seq_length]

        # Prepare target values if provided
        y_seq = None
        if y is not None:
            y_values = y.values
            y_seq = y_values[seq_length:]

        return X_seq, y_seq

    def train(self, data, target_column='realized_vol', regime_column=None):
        """
        Train the model.

        Parameters:
        -----------
        data : pd.DataFrame
            Training data
        target_column : str
            Name of the target column
        regime_column : str, optional
            Name of the column with regime labels

        Returns:
        --------
        dict
            Training metrics
        """
        if 'lstm' in self.config.get('models', []) and not TENSORFLOW_AVAILABLE:
            logger.warning("TensorFlow not available. LSTM training will be skipped.")
            self.config['models'] = [m for m in self.config.get('models', []) if m != 'lstm']

        # Initialize models if not done already
        if not self.models:
            self.initialize_models()

        # Prepare features
        X, y = self.prepare_features(data, target_column)

        if X.empty or y is None or y.empty:
            raise ValueError("No valid training data")

        # Train each model
        metrics = {}
        for name, model in self.models.items():
            if name == 'lstm':
                # Special handling for LSTM
                X_seq, y_seq = self.prepare_lstm_sequences(X, y)
                history = model['model'].fit(
                    X_seq, y_seq,
                    epochs=self.config.get('lstm_epochs', 50),
                    batch_size=self.config.get('lstm_batch_size', 32),
                    validation_split=0.2,
                    verbose=0
                )
                # Store training history
                metrics[name] = {
                    'loss': history.history['loss'][-1],
                    'val_loss': history.history['val_loss'][-1]
                }
            else:
                # Standard models
                model.fit(X, y)
                predictions = model.predict(X)

                # Calculate metrics
                metrics[name] = {
                    'mse': mean_squared_error(y, predictions),
                    'mae': mean_absolute_error(y, predictions),
                    'r2': r2_score(y, predictions)
                }

                # Store feature importance for tree-based models
                if hasattr(model['model'], 'feature_importances_'):
                    self.feature_importance[name] = dict(
                        zip(X.columns, model['model'].feature_importances_)
                    )

        # Optimize ensemble weights if we have regime information
        if regime_column and regime_column in data.columns:
            self._optimize_ensemble_weights(X, y, data[regime_column])

        logger.info("Models trained successfully")
        return metrics

    def _optimize_ensemble_weights(self, X, y, regimes):
        """
        Optimize ensemble weights based on performance in different regimes.

        Parameters:
        -----------
        X : pd.DataFrame
            Feature matrix
        y : pd.Series
            Target values
        regimes : pd.Series
            Market regime labels
        """
        # Get unique regimes
        unique_regimes = regimes.unique()

        # Initialize weights for each regime
        regime_weights = {regime: {} for regime in unique_regimes}

        # For each regime, calculate model performance
        for regime in unique_regimes:
            # Get indices for this regime
            idx = regimes == regime
            if sum(idx) < 10:  # Skip if too few samples
                continue

            X_regime = X.loc[idx]
            y_regime = y.loc[idx]

            # Calculate error for each model
            errors = {}
            for name, model in self.models.items():
                if name == 'lstm':
                    # Skip LSTM for now (needs sequence data)
                    continue

                try:
                    predictions = model.predict(X_regime)
                    errors[name] = mean_squared_error(y_regime, predictions)
                except Exception as e:
                    logger.error(f"Error predicting with {name} for regime {regime}: {e}")
                    errors[name] = float('inf')

            # Convert errors to weights (inverse of error)
            if errors:
                # Avoid division by zero
                for name in errors:
                    if errors[name] == 0:
                        errors[name] = 1e-10

                # Calculate inverse error
                inverse_errors = {name: 1.0 / error for name, error in errors.items()}

                # Normalize to get weights
                total = sum(inverse_errors.values())
                weights = {name: err / total for name, err in inverse_errors.items()}

                regime_weights[regime] = weights

        # Store regime-specific weights
        self.regime_weights = regime_weights

        logger.info(f"Optimized weights for {len(unique_regimes)} regimes")

    def predict(self, data, regime=None):
        """
        Make predictions with the model.

        Parameters:
        -----------
        data : pd.DataFrame
            Input data
        regime : str, optional
            Current market regime

        Returns:
        --------
        pd.Series
            Predicted volatility
        """
        if not self.models:
            raise ValueError("Models are not trained")

        # Prepare features
        X, _ = self.prepare_features(data)

        # Get predictions from each model
        predictions = {}
        for name, model in self.models.items():
            if name == 'lstm':
                # Special handling for LSTM
                try:
                    X_seq, _ = self.prepare_lstm_sequences(X)
                    pred = model['model'].predict(X_seq)
                    # Align with original data (account for sequence length)
                    seq_length = model['seq_length']
                    pred_series = pd.Series(
                        index=data.index[seq_length:],
                        data=pred.flatten()
                    )
                    predictions[name] = pred_series
                except Exception as e:
                    logger.error(f"Error predicting with LSTM: {e}")
            else:
                try:
                    pred = model.predict(X)
                    predictions[name] = pd.Series(index=X.index, data=pred)
                except Exception as e:
                    logger.error(f"Error predicting with {name}: {e}")

        # Determine weights to use
        weights = self._get_weights_for_regime(regime)

        # Combine predictions
        result = self._combine_predictions(predictions, weights)

        return result

    def _get_weights_for_regime(self, regime=None):
        """
        Get appropriate weights for the current regime.

        Parameters:
        -----------
        regime : str, optional
            Current market regime

        Returns:
        --------
        dict
            Model weights
        """
        # If we have regime-specific weights and current regime is provided
        if hasattr(self, 'regime_weights') and regime and regime in self.regime_weights:
            return self.regime_weights[regime]

        # Otherwise use default weights
        return self.ensemble_weights

    def _combine_predictions(self, predictions, weights):
        """
        Combine predictions from multiple models.

        Parameters:
        -----------
        predictions : dict
            Dictionary with model predictions
        weights : dict
            Dictionary with model weights

        Returns:
        --------
        pd.Series
            Combined prediction
        """
        # Check if we have valid predictions
        if not predictions:
            raise ValueError("No valid predictions to combine")

        # Get common index
        common_models = set(predictions.keys()) & set(weights.keys())
        if not common_models:
            raise ValueError("No models with both predictions and weights")

        # Normalize weights for available models
        available_weights = {name: weights[name] for name in common_models}
        total_weight = sum(available_weights.values())
        if total_weight == 0:
            # If all weights are zero, use equal weights
            normalized_weights = {name: 1.0 / len(common_models) for name in common_models}
        else:
            normalized_weights = {name: w / total_weight for name, w in available_weights.items()}

        # Initialize result with zeros
        result = None

        # Add weighted predictions
        for name in common_models:
            weighted_pred = predictions[name] * normalized_weights[name]
            if result is None:
                result = weighted_pred
            else:
                # Align indices
                result = result.add(weighted_pred, fill_value=0)

        return result

    def save_model(self, directory='models'):
        """
        Save the trained model.

        Parameters:
        -----------
        directory : str
            Directory to save the model
        """
        os.makedirs(directory, exist_ok=True)
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

        # Save standard models
        for name, model in self.models.items():
            if name != 'lstm':
                joblib.dump(model, os.path.join(directory, f"{name}_{timestamp}.joblib"))

        # Save LSTM model if it exists
        if 'lstm' in self.models:
            self.models['lstm']['model'].save(os.path.join(directory, f"lstm_{timestamp}"))

        # Save weights and config
        joblib.dump(
            {
                'ensemble_weights': self.ensemble_weights,
                'feature_importance': self.feature_importance,
                'config': self.config,
                'regime_weights': self.regime_weights if hasattr(self, 'regime_weights') else {}
            },
            os.path.join(directory, f"metadata_{timestamp}.joblib")
        )

        logger.info(f"Model saved to {directory} with timestamp {timestamp}")

    def load_model(self, directory, timestamp=None):
        """
        Load a trained model.

        Parameters:
        -----------
        directory : str
            Directory where the model is saved
        timestamp : str, optional
            Specific timestamp to load. If None, loads the latest.
        """
        if not os.path.exists(directory):
            raise ValueError(f"Directory {directory} does not exist")

        # Find available timestamps
        files = os.listdir(directory)
        timestamps = set()
        for file in files:
            if '_' in file:
                ts = file.split('_', 1)[1].split('.')[0]
                timestamps.add(ts)

        if not timestamps:
            raise ValueError(f"No model files found in {directory}")

        # Use latest timestamp if not specified
        if timestamp is None:
            timestamp = sorted(timestamps)[-1]
        elif timestamp not in timestamps:
            raise ValueError(f"Timestamp {timestamp} not found")

        # Load models
        self.models = {}
        for file in files:
            if timestamp in file:
                if file.startswith('lstm_'):
                    # Load LSTM model
                    lstm_dir = os.path.join(directory, file)
                    if os.path.isdir(lstm_dir):
                        self.models['lstm'] = {
                            'model': tf.keras.models.load_model(lstm_dir),
                            'seq_length': 10  # Default, will be overridden from config
                        }
                elif file.startswith('metadata_'):
                    # Load metadata
                    metadata = joblib.load(os.path.join(directory, file))
                    self.ensemble_weights = metadata.get('ensemble_weights', {})
                    self.feature_importance = metadata.get('feature_importance', {})
                    self.config = metadata.get('config', {})
                    if 'regime_weights' in metadata:
                        self.regime_weights = metadata['regime_weights']

                    # Update LSTM sequence length if needed
                    if 'lstm' in self.models and 'lstm_seq_length' in self.config:
                        self.models['lstm']['seq_length'] = self.config['lstm_seq_length']
                else:
                    # Load standard models
                    model_name = file.split('_')[0]
                    self.models[model_name] = joblib.load(os.path.join(directory, file))

        if not self.models:
            raise ValueError(f"Failed to load models for timestamp {timestamp}")

        logger.info(f"Loaded model with timestamp {timestamp}")

    def get_feature_importance(self):
        """
        Get feature importance for the model.

        Returns:
        --------
        dict
            Feature importance for each model
        """
        return self.feature_importance