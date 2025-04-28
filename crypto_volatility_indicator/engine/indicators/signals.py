"""
Module for generating trading signals based on volatility indicators.
Provides functionality to generate entry/exit signals, position sizing,
and risk management parameters.
"""
import os
import sys
import numpy as np
import pandas as pd
from enum import Enum
import logging

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, project_root)

from crypto_volatility_indicator.utils.logger import get_logger


logger = get_logger(__name__)


class SignalType(Enum):
    """Enumeration of signal types."""
    ENTRY_LONG = 'entry_long'
    EXIT_LONG = 'exit_long'
    ENTRY_SHORT = 'entry_short'
    EXIT_SHORT = 'exit_short'
    REDUCE_POSITION = 'reduce_position'
    INCREASE_POSITION = 'increase_position'
    VOLATILITY_BREAKOUT = 'volatility_breakout'
    VOLATILITY_CONTRACTION = 'volatility_contraction'
    REGIME_CHANGE = 'regime_change'


class SignalGenerator:
    """
    Class for generating trading signals based on volatility indicators.
    """

    def __init__(self, config=None):
        """
        Initialize the SignalGenerator.

        Parameters:
        -----------
        config : dict, optional
            Configuration parameters
        """
        self.config = config or {}

        # Default thresholds
        self.vol_breakout_threshold = self.config.get('vol_breakout_threshold', 2.0)
        self.vol_contraction_threshold = self.config.get('vol_contraction_threshold', 0.5)
        self.regime_change_sensitivity = self.config.get('regime_change_sensitivity', 0.7)
        self.position_sizing_factor = self.config.get('position_sizing_factor', 1.0)
        self.max_position_size = self.config.get('max_position_size', 1.0)
        self.min_position_size = self.config.get('min_position_size', 0.1)

        # Store historical signals
        self.signals_history = []

        # Volatility indicator reference (will be set later)
        self.volatility_indicator = None

    def set_volatility_indicator(self, indicator):
        """
        Set the volatility indicator to use for signal generation.

        Parameters:
        -----------
        indicator : ProgressiveAdaptiveVolatilityIndicator
            The volatility indicator to use
        """
        from engine.indicators.composite import ProgressiveAdaptiveVolatilityIndicator
        if not isinstance(indicator, ProgressiveAdaptiveVolatilityIndicator):
            raise TypeError("Indicator must be a ProgressiveAdaptiveVolatilityIndicator")

        self.volatility_indicator = indicator
        logger.info("Volatility indicator set")

    def generate_signals(self, price_data, volatility_data=None, regime_data=None):
        """
        Generate trading signals based on price and volatility data.

        Parameters:
        -----------
        price_data : pd.DataFrame
            DataFrame with price data (must have 'close' column)
        volatility_data : pd.DataFrame, optional
            DataFrame with volatility data
        regime_data : pd.DataFrame, optional
            DataFrame with regime data

        Returns:
        --------
        pd.DataFrame
            DataFrame with generated signals
        """
        if 'close' not in price_data.columns:
            raise ValueError("Price data must have a 'close' column")

        # Use indicator data if no explicit data provided
        if volatility_data is None and self.volatility_indicator is not None:
            volatility_data = self.volatility_indicator.get_volatility_data()

        if regime_data is None and self.volatility_indicator is not None:
            regime_data = self.volatility_indicator.get_regime_data()

        if volatility_data is None:
            raise ValueError("Volatility data is required")

        # Align data
        aligned_data = self._align_data(price_data, volatility_data, regime_data)

        # Generate signals
        signals = self._generate_volatility_breakout_signals(aligned_data)
        signals = pd.concat([signals, self._generate_regime_change_signals(aligned_data)], axis=1)
        signals = pd.concat([signals, self._generate_position_sizing_signals(aligned_data)], axis=1)

        # Add to history
        self.signals_history.append(signals.iloc[-1].to_dict())

        return signals

    def _align_data(self, price_data, volatility_data, regime_data=None):
        """
        Align all data to the same index.

        Parameters:
        -----------
        price_data : pd.DataFrame
            DataFrame with price data
        volatility_data : pd.DataFrame
            DataFrame with volatility data
        regime_data : pd.DataFrame, optional
            DataFrame with regime data

        Returns:
        --------
        pd.DataFrame
            Aligned data
        """
        # Start with price data
        aligned = price_data[['close']].copy()

        # Add volatility data
        for col in volatility_data.columns:
            aligned[col] = volatility_data[col]

        # Add regime data if provided
        if regime_data is not None:
            for col in regime_data.columns:
                aligned[col] = regime_data[col]

        # Forward fill any missing values
        aligned = aligned.ffill()

        return aligned

    def _generate_volatility_breakout_signals(self, data):
        """
        Generate signals based on volatility breakouts.

        Parameters:
        -----------
        data : pd.DataFrame
            Aligned data with price and volatility

        Returns:
        --------
        pd.DataFrame
            DataFrame with volatility breakout signals
        """
        signals = pd.DataFrame(index=data.index)

        # Assume 'composite_vol' is the main volatility indicator
        if 'composite_vol' not in data.columns:
            logger.warning("No composite_vol column found, using first volatility column")
            vol_cols = [col for col in data.columns if 'vol' in col.lower()]
            if not vol_cols:
                raise ValueError("No volatility columns found in data")
            vol_col = vol_cols[0]
        else:
            vol_col = 'composite_vol'

        # Calculate rolling volatility stats
        rolling_window = self.config.get('rolling_window', 20)
        data['vol_mean'] = data[vol_col].rolling(rolling_window).mean()
        data['vol_std'] = data[vol_col].rolling(rolling_window).std()

        # Generate breakout signals
        signals['volatility_breakout'] = 0
        breakout_idx = (
                (data[vol_col] > data['vol_mean'] + self.vol_breakout_threshold * data['vol_std']) &
                (data[vol_col].shift(1) <= data['vol_mean'].shift(1) + self.vol_breakout_threshold * data[
                    'vol_std'].shift(1))
        )
        signals.loc[breakout_idx, 'volatility_breakout'] = 1

        # Generate contraction signals
        signals['volatility_contraction'] = 0
        contraction_idx = (
                (data[vol_col] < data['vol_mean'] - self.vol_contraction_threshold * data['vol_std']) &
                (data[vol_col].shift(1) >= data['vol_mean'].shift(1) - self.vol_contraction_threshold * data[
                    'vol_std'].shift(1))
        )
        signals.loc[contraction_idx, 'volatility_contraction'] = 1

        return signals

    def _generate_regime_change_signals(self, data):
        """
        Generate signals based on regime changes.

        Parameters:
        -----------
        data : pd.DataFrame
            Aligned data with price, volatility, and regime info

        Returns:
        --------
        pd.DataFrame
            DataFrame with regime change signals
        """
        signals = pd.DataFrame(index=data.index)

        # Check if regime data is available
        if 'regime' not in data.columns:
            logger.warning("No regime column found, skipping regime change signals")
            signals['regime_change'] = 0
            return signals

        # Detect regime changes
        signals['regime_change'] = 0
        signals['regime'] = data['regime']
        signals.loc[data['regime'] != data['regime'].shift(1), 'regime_change'] = 1

        # Detect transition to extreme volatility regime
        if 'extreme_vol' in set(data['regime'].unique()):
            signals['extreme_vol_regime'] = 0
            extreme_vol_idx = (data['regime'] == 'extreme_vol') & (data['regime'].shift(1) != 'extreme_vol')
            signals.loc[extreme_vol_idx, 'extreme_vol_regime'] = 1

        # Detect transition to low volatility regime
        if 'low_vol' in set(data['regime'].unique()):
            signals['low_vol_regime'] = 0
            low_vol_idx = (data['regime'] == 'low_vol') & (data['regime'].shift(1) != 'low_vol')
            signals.loc[low_vol_idx, 'low_vol_regime'] = 1

        return signals

    def _generate_position_sizing_signals(self, data):
        """
        Generate position sizing signals based on volatility.

        Parameters:
        -----------
        data : pd.DataFrame
            Aligned data with price and volatility

        Returns:
        --------
        pd.DataFrame
            DataFrame with position sizing signals
        """
        signals = pd.DataFrame(index=data.index)

        # Assume 'composite_vol' is the main volatility indicator
        if 'composite_vol' not in data.columns:
            vol_cols = [col for col in data.columns if 'vol' in col.lower()]
            if not vol_cols:
                raise ValueError("No volatility columns found in data")
            vol_col = vol_cols[0]
        else:
            vol_col = 'composite_vol'

        # Calculate inverse volatility for position sizing
        # Higher volatility → smaller position
        # Lower volatility → larger position
        rolling_window = self.config.get('rolling_window', 20)
        data['vol_percentile'] = data[vol_col].rolling(rolling_window).apply(
            lambda x: pd.Series(x).rank(pct=True).iloc[-1]
        )

        # Calculate position size (inverse of volatility percentile)
        # Scale between min and max position size
        signals['position_size'] = (
                                           self.min_position_size +
                                           (self.max_position_size - self.min_position_size) *
                                           (1 - data['vol_percentile'])
                                   ) * self.position_sizing_factor

        # Generate increase/decrease position signals
        signals['increase_position'] = 0
        signals['reduce_position'] = 0

        # Increase position when volatility decreases significantly
        increase_idx = data['vol_percentile'] < data['vol_percentile'].shift(1) - 0.2
        signals.loc[increase_idx, 'increase_position'] = 1

        # Reduce position when volatility increases significantly
        reduce_idx = data['vol_percentile'] > data['vol_percentile'].shift(1) + 0.2
        signals.loc[reduce_idx, 'reduce_position'] = 1

        return signals

    def calculate_stop_loss(self, price_data, volatility_data=None, position_type='long'):
        """
        Calculate dynamic stop loss levels based on volatility.

        Parameters:
        -----------
        price_data : pd.DataFrame
            DataFrame with price data (must have 'close' column)
        volatility_data : pd.DataFrame, optional
            DataFrame with volatility data
        position_type : str
            'long' or 'short'

        Returns:
        --------
        pd.Series
            Series with stop loss prices
        """
        if 'close' not in price_data.columns:
            raise ValueError("Price data must have a 'close' column")

        # Use indicator data if no explicit data provided
        if volatility_data is None and self.volatility_indicator is not None:
            volatility_data = self.volatility_indicator.get_volatility_data()

        if volatility_data is None:
            raise ValueError("Volatility data is required")

        # Align data
        data = self._align_data(price_data, volatility_data)

        # Assume 'composite_vol' is the main volatility indicator
        if 'composite_vol' not in data.columns:
            vol_cols = [col for col in data.columns if 'vol' in col.lower()]
            if not vol_cols:
                raise ValueError("No volatility columns found in data")
            vol_col = vol_cols[0]
        else:
            vol_col = 'composite_vol'

        # Scale factor based on current volatility regime
        if 'regime' in data.columns:
            regime_factors = {
                'low_vol': 3.0,
                'normal_vol': 2.0,
                'high_vol': 1.5,
                'extreme_vol': 1.0
            }

            data['regime_factor'] = data['regime'].map(
                lambda x: regime_factors.get(x, 2.0)
            )
        else:
            data['regime_factor'] = 2.0

        # Calculate ATR-like volatility measure
        data['vol_atr'] = data[vol_col] * data['close']

        # Calculate stop loss distances
        stop_distances = data['vol_atr'] * data['regime_factor']

        # Calculate stop loss prices
        if position_type.lower() == 'long':
            stop_prices = data['close'] - stop_distances
        else:  # short position
            stop_prices = data['close'] + stop_distances

        return stop_prices

    def calculate_take_profit(self, price_data, volatility_data=None, position_type='long'):
        """
        Calculate dynamic take profit levels based on volatility.

        Parameters:
        -----------
        price_data : pd.DataFrame
            DataFrame with price data (must have 'close' column)
        volatility_data : pd.DataFrame, optional
            DataFrame with volatility data
        position_type : str
            'long' or 'short'

        Returns:
        --------
        pd.Series
            Series with take profit prices
        """
        if 'close' not in price_data.columns:
            raise ValueError("Price data must have a 'close' column")

        # Use indicator data if no explicit data provided
        if volatility_data is None and self.volatility_indicator is not None:
            volatility_data = self.volatility_indicator.get_volatility_data()

        if volatility_data is None:
            raise ValueError("Volatility data is required")

        # Align data
        data = self._align_data(price_data, volatility_data)

        # Assume 'composite_vol' is the main volatility indicator
        if 'composite_vol' not in data.columns:
            vol_cols = [col for col in data.columns if 'vol' in col.lower()]
            if not vol_cols:
                raise ValueError("No volatility columns found in data")
            vol_col = vol_cols[0]
        else:
            vol_col = 'composite_vol'

        # Scale factor based on risk-reward ratio
        risk_reward_ratio = self.config.get('risk_reward_ratio', 2.0)

        # Calculate ATR-like volatility measure
        data['vol_atr'] = data[vol_col] * data['close']

        # Calculate take profit distances (risk-reward times stop distance)
        if 'regime' in data.columns:
            regime_factors = {
                'low_vol': 3.0,
                'normal_vol': 2.0,
                'high_vol': 1.5,
                'extreme_vol': 1.0
            }

            data['regime_factor'] = data['regime'].map(
                lambda x: regime_factors.get(x, 2.0)
            )
        else:
            data['regime_factor'] = 2.0

        take_profit_distances = data['vol_atr'] * data['regime_factor'] * risk_reward_ratio

        # Calculate take profit prices
        if position_type.lower() == 'long':
            take_profit_prices = data['close'] + take_profit_distances
        else:  # short position
            take_profit_prices = data['close'] - take_profit_distances

        return take_profit_prices

    def get_signals_history(self):
        """
        Get the history of generated signals.

        Returns:
        --------
        list
            List of signal dictionaries
        """
        return self.signals_history

    def get_last_signals(self):
        """
        Get the most recent signals.

        Returns:
        --------
        dict
            Dictionary with the most recent signals
        """
        if not self.signals_history:
            return {}

        return self.signals_history[-1]