"""
Module for Kaufman's Adaptive Moving Average (KAMA).
Implements standard and enhanced versions of KAMA.
"""
import os
import sys
import numpy as np
import pandas as pd
import logging
import matplotlib.pyplot as plt

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, project_root)
from crypto_volatility_indicator.utils.logger import get_logger

# Set up logger
logger = get_logger(__name__)


class KAMAIndicator:
    """
    Kaufman's Adaptive Moving Average (KAMA).

    This class implements standard KAMA and enhanced versions with
    additional features like volume weighting and volatility adaptation.
    """

    def __init__(self, er_period=10, fast_ef=0.666, slow_ef=0.0645):
        """
        Initialize the KAMA indicator.

        Parameters:
        -----------
        er_period : int
            Efficiency Ratio period
        fast_ef : float
            Fast Efficiency Factor (default: 2/(2+1))
        slow_ef : float
            Slow Efficiency Factor (default: 2/(30+1))
        """
        self.er_period = er_period
        self.fast_ef = fast_ef
        self.slow_ef = slow_ef

        logger.info(f"KAMAIndicator initialized with er_period={er_period}, fast_ef={fast_ef}, slow_ef={slow_ef}")

    def calculate(self, df, price_col='close'):
        """
        Calculate standard KAMA.

        Parameters:
        -----------
        df : pd.DataFrame
            DataFrame with price data
        price_col : str
            Column name for price data

        Returns:
        --------
        pd.DataFrame
            DataFrame with KAMA values
        """
        if df is None or df.empty:
            logger.warning("Empty DataFrame provided to calculate KAMA")
            return df

        # Make a copy to avoid modifying the original
        df_kama = df.copy()

        # Check for required column
        if price_col not in df_kama.columns:
            logger.warning(f"Price column '{price_col}' not found in DataFrame")
            return df_kama

        try:
            # Calculate price change
            df_kama['price_change'] = abs(df_kama[price_col] - df_kama[price_col].shift(self.er_period))

            # Calculate volatility (sum of absolute price changes over the period)
            df_kama['volatility'] = abs(df_kama[price_col].diff()).rolling(window=self.er_period).sum()

            # Calculate Efficiency Ratio
            df_kama['er'] = df_kama['price_change'] / df_kama['volatility']
            df_kama['er'] = df_kama['er'].replace([np.inf, -np.inf], np.nan).fillna(0)

            # Calculate Smoothing Constant
            df_kama['sc'] = (df_kama['er'] * (self.fast_ef - self.slow_ef) + self.slow_ef) ** 2

            # Initialize KAMA
            kama = np.zeros(len(df_kama))

            # Set first KAMA value to first available price
            first_idx = df_kama[price_col].first_valid_index()
            if first_idx is not None:
                first_pos = df_kama.index.get_loc(first_idx)
                kama[first_pos] = df_kama[price_col].iloc[first_pos]

                # Calculate KAMA values
                for i in range(first_pos + 1, len(df_kama)):
                    kama[i] = kama[i - 1] + df_kama['sc'].iloc[i] * (df_kama[price_col].iloc[i] - kama[i - 1])

            # Add KAMA to DataFrame
            df_kama['kama'] = kama

            logger.info(f"Calculated KAMA with er_period={self.er_period}")
            return df_kama

        except Exception as e:
            logger.error(f"Error calculating KAMA: {e}")
            return df_kama

    def calculate_volume_weighted(self, df, price_col='close', volume_col='volume'):
        """
        Calculate Volume-Weighted KAMA.

        Parameters:
        -----------
        df : pd.DataFrame
            DataFrame with price and volume data
        price_col : str
            Column name for price data
        volume_col : str
            Column name for volume data

        Returns:
        --------
        pd.DataFrame
            DataFrame with Volume-Weighted KAMA values
        """
        if df is None or df.empty:
            logger.warning("Empty DataFrame provided to calculate Volume-Weighted KAMA")
            return df

        # Make a copy to avoid modifying the original
        df_vwkama = df.copy()

        # Check for required columns
        if price_col not in df_vwkama.columns:
            logger.warning(f"Price column '{price_col}' not found in DataFrame")
            return df_vwkama

        if volume_col not in df_vwkama.columns:
            logger.warning(f"Volume column '{volume_col}' not found in DataFrame")
            return df_vwkama

        try:
            # Normalize volume
            df_vwkama['volume_z'] = (df_vwkama[volume_col] - df_vwkama[volume_col].rolling(window=20).mean()) / \
                                    df_vwkama[volume_col].rolling(window=20).std()
            df_vwkama['volume_z'] = df_vwkama['volume_z'].fillna(0)

            # Calculate volume factor (higher volume = faster adaptation)
            df_vwkama['volume_factor'] = np.clip(1.0 + 0.5 * df_vwkama['volume_z'], 0.5, 2.0)

            # Calculate price change
            df_vwkama['price_change'] = abs(df_vwkama[price_col] - df_vwkama[price_col].shift(self.er_period))

            # Calculate volatility (sum of absolute price changes)
            df_vwkama['volatility'] = abs(df_vwkama[price_col].diff()).rolling(window=self.er_period).sum()

            # Calculate Efficiency Ratio
            df_vwkama['er'] = df_vwkama['price_change'] / df_vwkama['volatility']
            df_vwkama['er'] = df_vwkama['er'].replace([np.inf, -np.inf], np.nan).fillna(0)

            # Adjust Efficiency Ratio based on volume
            df_vwkama['adjusted_er'] = df_vwkama['er'] * df_vwkama['volume_factor']
            df_vwkama['adjusted_er'] = df_vwkama['adjusted_er'].clip(0, 1)

            # Calculate Smoothing Constant
            df_vwkama['sc'] = (df_vwkama['adjusted_er'] * (self.fast_ef - self.slow_ef) + self.slow_ef) ** 2

            # Initialize KAMA
            vwkama = np.zeros(len(df_vwkama))

            # Set first KAMA value to first available price
            first_idx = df_vwkama[price_col].first_valid_index()
            if first_idx is not None:
                first_pos = df_vwkama.index.get_loc(first_idx)
                vwkama[first_pos] = df_vwkama[price_col].iloc[first_pos]

                # Calculate KAMA values
                for i in range(first_pos + 1, len(df_vwkama)):
                    vwkama[i] = vwkama[i - 1] + df_vwkama['sc'].iloc[i] * (
                                df_vwkama[price_col].iloc[i] - vwkama[i - 1])

            # Add Volume-Weighted KAMA to DataFrame
            df_vwkama['vwkama'] = vwkama

            logger.info(f"Calculated Volume-Weighted KAMA with er_period={self.er_period}")
            return df_vwkama

        except Exception as e:
            logger.error(f"Error calculating Volume-Weighted KAMA: {e}")
            return df_vwkama

    def calculate_volatility_adaptive(self, df, price_col='close', vol_lookback=20):
        """
        Calculate Volatility-Adaptive KAMA.

        Parameters:
        -----------
        df : pd.DataFrame
            DataFrame with price data
        price_col : str
            Column name for price data
        vol_lookback : int
            Lookback period for volatility calculation

        Returns:
        --------
        pd.DataFrame
            DataFrame with Volatility-Adaptive KAMA values
        """
        if df is None or df.empty:
            logger.warning("Empty DataFrame provided to calculate Volatility-Adaptive KAMA")
            return df

        # Make a copy to avoid modifying the original
        df_vakama = df.copy()

        # Check for required column
        if price_col not in df_vakama.columns:
            logger.warning(f"Price column '{price_col}' not found in DataFrame")
            return df_vakama

        try:
            # Calculate returns
            df_vakama['return'] = df_vakama[price_col].pct_change()

            # Calculate rolling volatility
            df_vakama['volatility'] = df_vakama['return'].rolling(window=vol_lookback).std()

            # Calculate volatility z-score
            df_vakama['vol_z'] = (df_vakama['volatility'] - df_vakama['volatility'].rolling(
                window=vol_lookback * 2).mean()) / df_vakama['volatility'].rolling(window=vol_lookback * 2).std()
            df_vakama['vol_z'] = df_vakama['vol_z'].fillna(0)

            # Calculate volatility factor (higher volatility = slower adaptation)
            df_vakama['vol_factor'] = np.clip(1.0 - 0.3 * df_vakama['vol_z'], 0.5, 2.0)

            # Calculate price change
            df_vakama['price_change'] = abs(df_vakama[price_col] - df_vakama[price_col].shift(self.er_period))

            # Calculate price volatility (sum of absolute price changes)
            df_vakama['price_vol'] = abs(df_vakama[price_col].diff()).rolling(window=self.er_period).sum()

            # Calculate Efficiency Ratio
            df_vakama['er'] = df_vakama['price_change'] / df_vakama['price_vol']
            df_vakama['er'] = df_vakama['er'].replace([np.inf, -np.inf], np.nan).fillna(0)

            # Adjust Efficiency Ratio based on volatility
            df_vakama['adjusted_er'] = df_vakama['er'] * df_vakama['vol_factor']
            df_vakama['adjusted_er'] = df_vakama['adjusted_er'].clip(0, 1)

            # Calculate dynamic smoothing factors
            fast_sc = 2.0 / (2.0 + 1.0)
            slow_sc = 2.0 / (30.0 + 1.0)

            # Adjust smoothing factors based on volatility
            df_vakama['adjusted_fast'] = fast_sc * df_vakama['vol_factor']
            df_vakama['adjusted_slow'] = slow_sc / df_vakama['vol_factor']

            # Calculate Smoothing Constant
            df_vakama['sc'] = (df_vakama['adjusted_er'] * (
                        df_vakama['adjusted_fast'] - df_vakama['adjusted_slow']) + df_vakama['adjusted_slow']) ** 2

            # Initialize KAMA
            vakama = np.zeros(len(df_vakama))

            # Set first KAMA value to first available price
            first_idx = df_vakama[price_col].first_valid_index()
            if first_idx is not None:
                first_pos = df_vakama.index.get_loc(first_idx)
                vakama[first_pos] = df_vakama[price_col].iloc[first_pos]

                # Calculate KAMA values
                for i in range(first_pos + 1, len(df_vakama)):
                    vakama[i] = vakama[i - 1] + df_vakama['sc'].iloc[i] * (
                                df_vakama[price_col].iloc[i] - vakama[i - 1])

            # Add Volatility-Adaptive KAMA to DataFrame
            df_vakama['vakama'] = vakama

            logger.info(f"Calculated Volatility-Adaptive KAMA with er_period={self.er_period}")
            return df_vakama

        except Exception as e:
            logger.error(f"Error calculating Volatility-Adaptive KAMA: {e}")
            return df_vakama

    def calculate_time_adaptive(self, df, price_col='close'):
        """
        Calculate Time-Adaptive KAMA (adapts based on time of day/week).

        Parameters:
        -----------
        df : pd.DataFrame
            DataFrame with price data and datetime index
        price_col : str
            Column name for price data

        Returns:
        --------
        pd.DataFrame
            DataFrame with Time-Adaptive KAMA values
        """
        if df is None or df.empty:
            logger.warning("Empty DataFrame provided to calculate Time-Adaptive KAMA")
            return df

        # Make a copy to avoid modifying the original
        df_takama = df.copy()

        # Check for required column
        if price_col not in df_takama.columns:
            logger.warning(f"Price column '{price_col}' not found in DataFrame")
            return df_takama

        # Check if index is datetime
        if not isinstance(df_takama.index, pd.DatetimeIndex):
            logger.warning("DataFrame index is not datetime, cannot apply time adaptivity")
            return df_takama

        try:
            # Extract time features
            df_takama['hour'] = df_takama.index.hour
            df_takama['day_of_week'] = df_takama.index.dayofweek

            # Create time-based weight (market open hours get higher weight)
            df_takama['time_weight'] = 1.0

            # Higher weight for US market hours (approximately 13:30-20:00 UTC)
            df_takama.loc[(df_takama['hour'] >= 13) & (df_takama['hour'] <= 20), 'time_weight'] = 1.5

            # Higher weight for Asian market hours (approximately 0:00-8:00 UTC)
            df_takama.loc[(df_takama['hour'] >= 0) & (df_takama['hour'] <= 8), 'time_weight'] = 1.3

            # Lower weight for weekends
            df_takama.loc[df_takama['day_of_week'] >= 5, 'time_weight'] *= 0.7

            # Calculate price change
            df_takama['price_change'] = abs(df_takama[price_col] - df_takama[price_col].shift(self.er_period))

            # Calculate volatility (sum of absolute price changes)
            df_takama['volatility'] = abs(df_takama[price_col].diff()).rolling(window=self.er_period).sum()

            # Calculate Efficiency Ratio
            df_takama['er'] = df_takama['price_change'] / df_takama['volatility']
            df_takama['er'] = df_takama['er'].replace([np.inf, -np.inf], np.nan).fillna(0)

            # Adjust Efficiency Ratio based on time
            df_takama['adjusted_er'] = df_takama['er'] * df_takama['time_weight']
            df_takama['adjusted_er'] = df_takama['adjusted_er'].clip(0, 1)

            # Calculate Smoothing Constant
            df_takama['sc'] = (df_takama['adjusted_er'] * (self.fast_ef - self.slow_ef) + self.slow_ef) ** 2

            # Initialize KAMA
            takama = np.zeros(len(df_takama))

            # Set first KAMA value to first available price
            first_idx = df_takama[price_col].first_valid_index()
            if first_idx is not None:
                first_pos = df_takama.index.get_loc(first_idx)
                takama[first_pos] = df_takama[price_col].iloc[first_pos]

                # Calculate KAMA values
                for i in range(first_pos + 1, len(df_takama)):
                    takama[i] = takama[i - 1] + df_takama['sc'].iloc[i] * (
                                df_takama[price_col].iloc[i] - takama[i - 1])

            # Add Time-Adaptive KAMA to DataFrame
            df_takama['takama'] = takama

            logger.info(f"Calculated Time-Adaptive KAMA with er_period={self.er_period}")
            return df_takama

        except Exception as e:
            logger.error(f"Error calculating Time-Adaptive KAMA: {e}")
            return df_takama

    def calculate_hybrid(self, df, price_col='close', volume_col='volume', vol_lookback=20):
        """
        Calculate Hybrid KAMA (combines volume, volatility, and time adaptivity).

        Parameters:
        -----------
        df : pd.DataFrame
            DataFrame with price and volume data and datetime index
        price_col : str
            Column name for price data
        volume_col : str
            Column name for volume data
        vol_lookback : int
            Lookback period for volatility calculation

        Returns:
        --------
        pd.DataFrame
            DataFrame with Hybrid KAMA values
        """
        if df is None or df.empty:
            logger.warning("Empty DataFrame provided to calculate Hybrid KAMA")
            return df

        # Make a copy to avoid modifying the original
        df_hybrid = df.copy()

        # Check for required columns
        missing_cols = []

        if price_col not in df_hybrid.columns:
            missing_cols.append(price_col)

        if volume_col not in df_hybrid.columns:
            missing_cols.append(volume_col)

        if missing_cols:
            logger.warning(f"Missing columns: {missing_cols}")
            return df_hybrid

        # Check if index is datetime
        is_datetime_index = isinstance(df_hybrid.index, pd.DatetimeIndex)

        try:
            # Volume adaptivity
            df_hybrid['volume_z'] = (df_hybrid[volume_col] - df_hybrid[volume_col].rolling(window=20).mean()) / \
                                    df_hybrid[volume_col].rolling(window=20).std()
            df_hybrid['volume_z'] = df_hybrid['volume_z'].fillna(0)
            df_hybrid['volume_factor'] = np.clip(1.0 + 0.3 * df_hybrid['volume_z'], 0.7, 1.5)

            # Volatility adaptivity
            df_hybrid['return'] = df_hybrid[price_col].pct_change()
            df_hybrid['volatility'] = df_hybrid['return'].rolling(window=vol_lookback).std()
            df_hybrid['vol_z'] = (df_hybrid['volatility'] - df_hybrid['volatility'].rolling(
                window=vol_lookback * 2).mean()) / df_hybrid['volatility'].rolling(window=vol_lookback * 2).std()
            df_hybrid['vol_z'] = df_hybrid['vol_z'].fillna(0)
            df_hybrid['vol_factor'] = np.clip(1.0 - 0.2 * df_hybrid['vol_z'], 0.8, 1.3)

            # Time adaptivity (if datetime index)
            df_hybrid['time_factor'] = 1.0

            if is_datetime_index:
                df_hybrid['hour'] = df_hybrid.index.hour
                df_hybrid['day_of_week'] = df_hybrid.index.dayofweek

                # Higher weight for US market hours
                df_hybrid.loc[(df_hybrid['hour'] >= 13) & (df_hybrid['hour'] <= 20), 'time_factor'] = 1.2

                # Higher weight for Asian market hours
                df_hybrid.loc[(df_hybrid['hour'] >= 0) & (df_hybrid['hour'] <= 8), 'time_factor'] = 1.1

                # Lower weight for weekends
                df_hybrid.loc[df_hybrid['day_of_week'] >= 5, 'time_factor'] *= 0.8

            # Combined adaptivity factor
            df_hybrid['combined_factor'] = df_hybrid['volume_factor'] * df_hybrid['vol_factor'] * df_hybrid[
                'time_factor']
            df_hybrid['combined_factor'] = np.clip(df_hybrid['combined_factor'], 0.5, 2.0)

            # Calculate price change
            df_hybrid['price_change'] = abs(df_hybrid[price_col] - df_hybrid[price_col].shift(self.er_period))

            # Calculate volatility (sum of absolute price changes)
            df_hybrid['price_vol'] = abs(df_hybrid[price_col].diff()).rolling(window=self.er_period).sum()

            # Calculate Efficiency Ratio
            df_hybrid['er'] = df_hybrid['price_change'] / df_hybrid['price_vol']
            df_hybrid['er'] = df_hybrid['er'].replace([np.inf, -np.inf], np.nan).fillna(0)

            # Adjust Efficiency Ratio based on combined factor
            df_hybrid['adjusted_er'] = df_hybrid['er'] * df_hybrid['combined_factor']
            df_hybrid['adjusted_er'] = df_hybrid['adjusted_er'].clip(0, 1)

            # Calculate Smoothing Constant
            df_hybrid['sc'] = (df_hybrid['adjusted_er'] * (self.fast_ef - self.slow_ef) + self.slow_ef) ** 2

            # Initialize KAMA
            hybrid = np.zeros(len(df_hybrid))

            # Set first KAMA value to first available price
            first_idx = df_hybrid[price_col].first_valid_index()
            if first_idx is not None:
                first_pos = df_hybrid.index.get_loc(first_idx)
                hybrid[first_pos] = df_hybrid[price_col].iloc[first_pos]

                # Calculate KAMA values
                for i in range(first_pos + 1, len(df_hybrid)):
                    hybrid[i] = hybrid[i - 1] + df_hybrid['sc'].iloc[i] * (
                                df_hybrid[price_col].iloc[i] - hybrid[i - 1])

            # Add Hybrid KAMA to DataFrame
            df_hybrid['hybrid_kama'] = hybrid

            logger.info(f"Calculated Hybrid KAMA with er_period={self.er_period}")
            return df_hybrid

        except Exception as e:
            logger.error(f"Error calculating Hybrid KAMA: {e}")
            return df_hybrid

    def visualize(self, df, price_col='close', kama_cols=None, show_plot=True):
        """
        Visualize price and KAMA values.

        Parameters:
        -----------
        df : pd.DataFrame
            DataFrame with price and KAMA data
        price_col : str
            Column name for price data
        kama_cols : list, optional
            List of KAMA column names to plot
        show_plot : bool
            Whether to display the plot or return the figure

        Returns:
        --------
        matplotlib.figure.Figure or None
            Figure object if show_plot is False, None otherwise
        """
        if df is None or df.empty:
            logger.warning("Empty DataFrame provided to visualize")
            return None

        # Check for required column
        if price_col not in df.columns:
            logger.warning(f"Price column '{price_col}' not found in DataFrame")
            return None

        # Determine KAMA columns
        if kama_cols is None:
            kama_candidates = ['kama', 'vwkama', 'vakama', 'takama', 'hybrid_kama']
            kama_cols = [col for col in kama_candidates if col in df.columns]
        else:
            kama_cols = [col for col in kama_cols if col in df.columns]

        if not kama_cols:
            logger.warning("No KAMA columns found in DataFrame")
            return None

        try:
            import matplotlib.pyplot as plt

            # Create figure
            fig, ax = plt.subplots(figsize=(15, 8))

            # Plot price
            ax.plot(df.index, df[price_col], color='black', alpha=0.5, linewidth=1, label=price_col.capitalize())

            # Plot KAMA values
            colors = ['blue', 'red', 'green', 'purple', 'orange']

            for i, col in enumerate(kama_cols):
                color = colors[i % len(colors)]
                ax.plot(df.index, df[col], color=color, linewidth=2, label=col.upper())

            # Add labels and title
            ax.set_title(f'Price and KAMA Indicators')
            ax.set_ylabel('Price')
            ax.grid(True, alpha=0.3)
            ax.legend()

            # Add Efficiency Ratio subplot if available
            if 'er' in df.columns:
                ax2 = ax.twinx()
                ax2.fill_between(df.index, 0, df['er'], color='gray', alpha=0.2)
                ax2.set_ylabel('Efficiency Ratio')
                ax2.set_ylim(0, 1)

            plt.tight_layout()

            if show_plot:
                plt.show()
                return None
            else:
                return fig

        except Exception as e:
            logger.error(f"Error visualizing KAMA: {e}")
            return None

    def generate_signals(self, df, price_col='close', kama_col='kama', signal_type='crossover'):
        """
        Generate trading signals based on KAMA.

        Parameters:
        -----------
        df : pd.DataFrame
            DataFrame with price and KAMA data
        price_col : str
            Column name for price data
        kama_col : str
            Column name for KAMA values
        signal_type : str
            Signal type ('crossover', 'slope', 'er_threshold')

        Returns:
        --------
        pd.DataFrame
            DataFrame with trading signals
        """
        if df is None or df.empty:
            logger.warning("Empty DataFrame provided to generate_signals")
            return df

        # Check for required columns
        if price_col not in df.columns:
            logger.warning(f"Price column '{price_col}' not found in DataFrame")
            return df

        if kama_col not in df.columns:
            logger.warning(f"KAMA column '{kama_col}' not found in DataFrame")
            return df

        # Make a copy to avoid modifying the original
        df_signals = df.copy()

        try:
            if signal_type == 'crossover':
                # Price crosses above/below KAMA
                df_signals['signal'] = 0
                df_signals.loc[df_signals[price_col] > df_signals[kama_col], 'signal'] = 1
                df_signals.loc[df_signals[price_col] < df_signals[kama_col], 'signal'] = -1

                # Generate buy/sell signals on crossovers
                df_signals['buy_signal'] = (
                            (df_signals['signal'] == 1) & (df_signals['signal'].shift(1) == -1)).astype(int)
                df_signals['sell_signal'] = (
                            (df_signals['signal'] == -1) & (df_signals['signal'].shift(1) == 1)).astype(int)

            elif signal_type == 'slope':
                # Calculate KAMA slope
                df_signals['kama_slope'] = df_signals[kama_col].diff(5)

                # Generate signals based on slope
                df_signals['signal'] = 0
                df_signals.loc[df_signals['kama_slope'] > 0, 'signal'] = 1
                df_signals.loc[df_signals['kama_slope'] < 0, 'signal'] = -1

                # Generate buy/sell signals on slope change
                df_signals['buy_signal'] = (
                            (df_signals['signal'] == 1) & (df_signals['signal'].shift(1) == -1)).astype(int)
                df_signals['sell_signal'] = (
                            (df_signals['signal'] == -1) & (df_signals['signal'].shift(1) == 1)).astype(int)

            elif signal_type == 'er_threshold':
                # Ensure Efficiency Ratio is available
                if 'er' not in df_signals.columns:
                    logger.warning("Efficiency Ratio not available for er_threshold signals")
                    return df_signals

                # Generate signals based on Efficiency Ratio thresholds
                df_signals['signal'] = 0
                df_signals.loc[
                    (df_signals['er'] > 0.7) & (df_signals[price_col] > df_signals[kama_col]), 'signal'] = 1
                df_signals.loc[
                    (df_signals['er'] > 0.7) & (df_signals[price_col] < df_signals[kama_col]), 'signal'] = -1

                # Generate buy/sell signals on signal change
                df_signals['buy_signal'] = (
                            (df_signals['signal'] == 1) & (df_signals['signal'].shift(1) != 1)).astype(int)
                df_signals['sell_signal'] = (
                            (df_signals['signal'] == -1) & (df_signals['signal'].shift(1) != -1)).astype(int)

            else:
                logger.warning(f"Unknown signal type: {signal_type}")
                return df_signals

            logger.info(f"Generated {signal_type} signals based on {kama_col}")
            return df_signals

        except Exception as e:
            logger.error(f"Error generating signals: {e}")
            return df_signals

    def run_analysis(self, df, price_col='close', volume_col=None, include_variations=True):
        """
        Run a comprehensive KAMA analysis.

        Parameters:
        -----------
        df : pd.DataFrame
            DataFrame with price data
        price_col : str
            Column name for price data
        volume_col : str, optional
            Column name for volume data
        include_variations : bool
            Whether to include variations of KAMA

        Returns:
        --------
        tuple
            (DataFrame with KAMA values, dict with analysis results)
        """
        # Calculate standard KAMA
        df_kama = self.calculate(df, price_col=price_col)

        # Calculate variations if requested
        if include_variations:
            if volume_col is not None and volume_col in df.columns:
                df_kama = self.calculate_volume_weighted(df_kama, price_col=price_col, volume_col=volume_col)

            df_kama = self.calculate_volatility_adaptive(df_kama, price_col=price_col)

            if isinstance(df.index, pd.DatetimeIndex):
                df_kama = self.calculate_time_adaptive(df_kama, price_col=price_col)

            if volume_col is not None and volume_col in df.columns:
                df_kama = self.calculate_hybrid(df_kama, price_col=price_col, volume_col=volume_col)

        # Generate signals
        df_signals = self.generate_signals(df_kama, price_col=price_col, kama_col='kama')

        # Visualize
        self.visualize(df_kama, price_col=price_col)

        # Calculate performance metrics
        performance = {}

        if 'buy_signal' in df_signals.columns and 'sell_signal' in df_signals.columns:
            buy_indices = df_signals.index[df_signals['buy_signal'] == 1]
            sell_indices = df_signals.index[df_signals['sell_signal'] == 1]

            trades = []
            current_buy = None

            for idx in sorted(list(buy_indices) + list(sell_indices)):
                if idx in buy_indices and current_buy is None:
                    current_buy = idx
                elif idx in sell_indices and current_buy is not None:
                    buy_price = df_signals.loc[current_buy, price_col]
                    sell_price = df_signals.loc[idx, price_col]
                    return_pct = (sell_price / buy_price - 1) * 100

                    trades.append({
                        'buy_date': current_buy,
                        'sell_date': idx,
                        'buy_price': buy_price,
                        'sell_price': sell_price,
                        'return_pct': return_pct
                    })

                    current_buy = None

            if trades:
                returns = [trade['return_pct'] for trade in trades]

                performance['trade_count'] = len(trades)
                performance['win_rate'] = sum(1 for r in returns if r > 0) / len(returns)
                performance['avg_return'] = sum(returns) / len(returns)
                performance['max_return'] = max(returns)
                performance['min_return'] = min(returns)
                performance['profit_factor'] = sum(r for r in returns if r > 0) / abs(
                    sum(r for r in returns if r < 0)) if sum(r for r in returns if r < 0) != 0 else float('inf')

        # Prepare results
        results = {
            'kama_params': {
                'er_period': self.er_period,
                'fast_ef': self.fast_ef,
                'slow_ef': self.slow_ef
            },
            'performance': performance,
            'variations_included': include_variations
        }

        logger.info(f"Completed KAMA analysis")
        return df_signals, results

# Factory function to get a KAMA indicator
def get_kama_indicator(er_period=10, fast_ef=0.666, slow_ef=0.0645):
    """
    Get a configured KAMA indicator.

    Parameters:
    -----------
    er_period : int
        Efficiency Ratio period
    fast_ef : float
        Fast Efficiency Factor
    slow_ef : float
        Slow Efficiency Factor

    Returns:
    --------
    KAMAIndicator
        Configured indicator instance
    """
    return KAMAIndicator(er_period=er_period, fast_ef=fast_ef, slow_ef=slow_ef)