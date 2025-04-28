"""
Module for calibrating volatility prediction models.
Provides utilities for comparing implied and realized volatility,
and calibrating machine learning models based on historical performance.
"""
import os
import sys
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from sklearn.metrics import mean_squared_error, mean_absolute_error
import logging
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, project_root)
from crypto_volatility_indicator.utils.logger import get_logger

logger = get_logger(__name__)


class ModelCalibrator:
    """
    A class for calibrating volatility prediction models based on
    historical performance and market conditions.
    """

    def __init__(self, lookback_period=30, recalibrate_freq=7):
        """
        Initialize the ModelCalibrator.

        Parameters:
        -----------
        lookback_period : int
            The number of days to look back for calibration data
        recalibrate_freq : int
            How often to recalibrate the model (in days)
        """
        self.lookback_period = lookback_period
        self.recalibrate_freq = recalibrate_freq
        self.last_calibration = None
        self.calibration_history = []
        self.optimal_params = {}

    def compare_implied_realized(self, implied_vol_data, realized_vol_data):
        """
        Compare implied volatility to realized volatility and calculate various metrics.

        Parameters:
        -----------
        implied_vol_data : pd.DataFrame
            DataFrame with timestamp index and implied volatility values
        realized_vol_data : pd.DataFrame
            DataFrame with timestamp index and realized volatility values

        Returns:
        --------
        dict
            Dictionary with comparison metrics
        """
        # Align the data
        aligned_data = pd.merge(
            implied_vol_data,
            realized_vol_data,
            left_index=True,
            right_index=True,
            how='inner'
        )

        if aligned_data.empty:
            logger.warning("No overlapping data between implied and realized volatility")
            return {}

        # Calculate metrics
        metrics = {
            'mse': mean_squared_error(aligned_data.iloc[:, 0], aligned_data.iloc[:, 1]),
            'mae': mean_absolute_error(aligned_data.iloc[:, 0], aligned_data.iloc[:, 1]),
            'bias': np.mean(aligned_data.iloc[:, 0] - aligned_data.iloc[:, 1]),
            'correlation': aligned_data.iloc[:, 0].corr(aligned_data.iloc[:, 1]),
            'vol_premium': np.mean(aligned_data.iloc[:, 0] / aligned_data.iloc[:, 1] - 1)
        }

        return metrics

    def calculate_vol_premium(self, implied_vol_data, realized_vol_data):
        """
        Calculate the volatility premium (implied / realized - 1).

        Parameters:
        -----------
        implied_vol_data : pd.DataFrame
            DataFrame with timestamp index and implied volatility values
        realized_vol_data : pd.DataFrame
            DataFrame with timestamp index and realized volatility values

        Returns:
        --------
        pd.DataFrame
            DataFrame with volatility premium over time
        """
        # Align the data
        aligned_data = pd.merge(
            implied_vol_data,
            realized_vol_data,
            left_index=True,
            right_index=True,
            how='inner',
            suffixes=('_implied', '_realized')
        )

        if aligned_data.empty:
            logger.warning("No overlapping data between implied and realized volatility")
            return pd.DataFrame()

        # Calculate volatility premium
        aligned_data['vol_premium'] = (
                aligned_data.iloc[:, 0] / aligned_data.iloc[:, 1] - 1
        )

        return aligned_data[['vol_premium']]

    def calibrate_model_parameters(self, model, X_train, y_train, param_bounds):
        """
        Calibrate model parameters using training data.

        Parameters:
        -----------
        model : object
            The model to calibrate (must have a .set_params() method)
        X_train : pd.DataFrame
            Training features
        y_train : pd.Series
            Training target values
        param_bounds : dict
            Dictionary with parameter names as keys and (min, max) tuples as values

        Returns:
        --------
        dict
            Dictionary with optimized parameters
        """
        param_names = list(param_bounds.keys())
        initial_values = [
            (param_bounds[param][0] + param_bounds[param][1]) / 2
            for param in param_names
        ]

        bounds = [param_bounds[param] for param in param_names]

        def objective_function(params):
            param_dict = {name: value for name, value in zip(param_names, params)}
            model.set_params(**param_dict)
            model.fit(X_train, y_train)
            predictions = model.predict(X_train)
            return mean_squared_error(y_train, predictions)

        # Optimize the parameters
        result = minimize(
            objective_function,
            initial_values,
            bounds=bounds,
            method='L-BFGS-B'
        )

        # Return the optimized parameters
        optimized_params = {
            name: value for name, value in zip(param_names, result.x)
        }

        self.optimal_params = optimized_params
        self.last_calibration = pd.Timestamp.now()
        self.calibration_history.append({
            'timestamp': self.last_calibration,
            'parameters': optimized_params,
            'objective_value': result.fun
        })

        return optimized_params

    def calculate_calibration_weights(self, market_regimes, current_regime):
        """
        Calculate calibration weights based on market regimes.
        Gives higher weight to calibration data from similar market regimes.

        Parameters:
        -----------
        market_regimes : pd.Series
            Series with market regime labels indexed by timestamp
        current_regime : str
            The current market regime

        Returns:
        --------
        pd.Series
            Series with weights for each timestamp
        """
        weights = pd.Series(index=market_regimes.index, data=1.0)

        # Give higher weight to data points from the same regime
        weights[market_regimes == current_regime] = 2.0

        # Normalize weights to sum to 1
        weights = weights / weights.sum()

        return weights

    def needs_recalibration(self):
        """
        Check if the model needs recalibration based on the recalibration frequency.

        Returns:
        --------
        bool
            True if recalibration is needed, False otherwise
        """
        if self.last_calibration is None:
            return True

        days_since_calibration = (pd.Timestamp.now() - self.last_calibration).days
        return days_since_calibration >= self.recalibrate_freq

    def get_calibration_history(self):
        """
        Get the calibration history.

        Returns:
        --------
        pd.DataFrame
            DataFrame with calibration history
        """
        if not self.calibration_history:
            return pd.DataFrame()

        return pd.DataFrame(self.calibration_history)