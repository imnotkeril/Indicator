"""
Module for meso-volatility analysis.
Handles analysis of volatility on medium timeframes (hours).
"""
import os
import sys
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import logging
from scipy import stats
from sklearn.cluster import KMeans
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, project_root)
from crypto_volatility_indicator.utils.logger import get_logger
from crypto_volatility_indicator.data.processors.normalizer import get_data_normalizer
from crypto_volatility_indicator.data.processors.filters import get_data_filter

# Set up logger
logger = get_logger(__name__)

class MesoVolatilityAnalyzer:
    """
    Analyzer for meso-volatility (hour timeframes).

    This class is responsible for analyzing and calculating volatility
    on medium timeframes, identifying volatility regimes, and detecting
    regime shifts.
    """

    def __init__(self, window_sizes=None, use_log_returns=True):
        """
        Initialize the meso-volatility analyzer.

        Parameters:
        -----------
        window_sizes : list, optional
            List of window sizes for rolling volatility calculation
        use_log_returns : bool
            If True, use log returns for volatility calculation
        """
        self.window_sizes = window_sizes or [6, 12, 24, 48, 96]
        self.use_log_returns = use_log_returns

        # Initialize data processing utilities
        self.normalizer = get_data_normalizer()
        self.filter = get_data_filter()

        logger.info(f"MesoVolatilityAnalyzer initialized with window sizes: {self.window_sizes}")

    def calculate_returns(self, df, price_col='close'):
        """
        Calculate returns from price data.

        Parameters:
        -----------
        df : pd.DataFrame
            DataFrame with price data
        price_col : str
            Column name for price data

        Returns:
        --------
        pd.DataFrame
            DataFrame with returns
        """
        if df is None or df.empty:
            logger.warning("Empty DataFrame provided to calculate_returns")
            return df

        # Make a copy to avoid modifying the original
        df_returns = df.copy()

        # Check for required column
        if price_col not in df_returns.columns:
            logger.warning(f"Price column '{price_col}' not found in DataFrame")
            return df_returns

        # Calculate returns
        try:
            # Ensure price column contains numeric values
            df_returns[price_col] = pd.to_numeric(df_returns[price_col], errors='coerce')

            if self.use_log_returns:
                # Prevent log(0) by adding a small constant
                df_returns['log_return'] = np.log(df_returns[price_col] / df_returns[price_col].shift(1).fillna(df_returns[price_col].iloc[0]) + 1e-10)
            else:
                df_returns['pct_return'] = df_returns[price_col].pct_change()

            return_col = 'log_return' if self.use_log_returns else 'pct_return'

            # Replace first row with 0
            if len(df_returns) > 1:
                df_returns.iloc[0, df_returns.columns.get_loc(return_col)] = 0

            # Remove infinite values
            df_returns[return_col] = df_returns[return_col].replace([np.inf, -np.inf], np.nan)

            logger.info(f"Calculated {'log' if self.use_log_returns else 'percentage'} returns")
            return df_returns

        except Exception as e:
            logger.error(f"Error calculating returns: {e}")
            return df_returns

    def calculate_historical_volatility(self, df, return_col=None):
        """
        Calculate historical volatility for different window sizes.

        Parameters:
        -----------
        df : pd.DataFrame
            DataFrame with returns
        return_col : str, optional
            Column name for returns (if None, uses 'log_return' or 'pct_return')

        Returns:
        --------
        pd.DataFrame
            DataFrame with historical volatility
        """
        if df is None or df.empty:
            logger.warning("Empty DataFrame provided to calculate_historical_volatility")
            return df

        # Make a copy to avoid modifying the original
        df_vol = df.copy()

        # Determine return column
        if return_col is None:
            return_col = 'log_return' if self.use_log_returns else 'pct_return'

        # Check for required column
        if return_col not in df_vol.columns:
            logger.warning(f"Return column '{return_col}' not found in DataFrame")
            return df_vol

        # Calculate historical volatility for each window size
        try:
            for window in self.window_sizes:
                # Remove NaN values to ensure accurate calculation
                clean_returns = df_vol[return_col].dropna()

                # Calculate standard deviation
                vol = clean_returns.rolling(window=window, min_periods=1).std()

                # Add column with volatility
                df_vol[f'volatility_{window}'] = vol

                # Annualize (assuming hourly data, multiply by sqrt(24 * 252))
                df_vol[f'volatility_{window}_annualized'] = vol * np.sqrt(24 * 252)

            logger.info(f"Calculated historical volatility for {len(self.window_sizes)} window sizes")
            return df_vol

        except Exception as e:
            logger.error(f"Error calculating historical volatility: {e}")
            return df_vol


    def calculate_rolling_volatility(self, df, return_col=None):
        """
        Calculate rolling volatility for different window sizes.

        Parameters:
        -----------
        df : pd.DataFrame
            DataFrame with returns
        return_col : str, optional
            Column name for returns (if None, uses 'log_return' or 'pct_return')

        Returns:
        --------
        pd.DataFrame
            DataFrame with rolling volatility
        """
        if df is None or df.empty:
            logger.warning("Empty DataFrame provided to calculate_rolling_volatility")
            return df

        # Make a copy to avoid modifying the original
        df_vol = df.copy()

        # Determine return column
        if return_col is None:
            return_col = 'log_return' if self.use_log_returns else 'pct_return'

        # Check for required column
        if return_col not in df_vol.columns:
            logger.warning(f"Return column '{return_col}' not found in DataFrame")
            return df_vol

        # Calculate rolling volatility for each window size
        try:
            for window in self.window_sizes:
                # Calculate standard deviation
                df_vol[f'volatility_{window}'] = df_vol[return_col].rolling(window=window).std()

                # Annualize (assuming hourly data, multiply by sqrt(24 * 252) for annual)
                df_vol[f'volatility_{window}_annualized'] = df_vol[f'volatility_{window}'] * np.sqrt(24 * 252)

            logger.info(f"Calculated rolling volatility for {len(self.window_sizes)} window sizes")
            return df_vol

        except Exception as e:
            logger.error(f"Error calculating rolling volatility: {e}")
            return df_vol

    def calculate_realized_volatility(self, df, window=24, return_col=None):
        """
        Calculate realized volatility for a specific window.

        Parameters:
        -----------
        df : pd.DataFrame
            DataFrame with returns
        window : int
            Window size for realized volatility calculation
        return_col : str, optional
            Column name for returns (if None, uses 'log_return' or 'pct_return')

        Returns:
        --------
        pd.DataFrame
            DataFrame with realized volatility
        """
        if df is None or df.empty:
            logger.warning("Empty DataFrame provided to calculate_realized_volatility")
            return df

        # Make a copy to avoid modifying the original
        df_rvol = df.copy()

        # Determine return column
        if return_col is None:
            return_col = 'log_return' if self.use_log_returns else 'pct_return'

        # Check for required column
        if return_col not in df_rvol.columns:
            logger.warning(f"Return column '{return_col}' not found in DataFrame")
            return df_rvol

        try:
            # Calculate realized volatility as squared returns
            df_rvol['return_squared'] = df_rvol[return_col] ** 2

            # Calculate rolling sum of squared returns
            df_rvol[f'realized_variance_{window}'] = df_rvol['return_squared'].rolling(window=window).sum()

            # Calculate realized volatility
            df_rvol[f'realized_volatility_{window}'] = np.sqrt(df_rvol[f'realized_variance_{window}'])

            # Annualize (assuming hourly data)
            df_rvol[f'realized_volatility_{window}_annualized'] = df_rvol[f'realized_volatility_{window}'] * np.sqrt(
                24 * 252 / window)

            logger.info(f"Calculated realized volatility with window {window}")
            return df_rvol

        except Exception as e:
            logger.error(f"Error calculating realized volatility: {e}")
            return df_rvol

    def detect_volatility_regimes(self, df, vol_col=None, n_regimes=3):
        """
        Detect volatility regimes using clustering.

        Parameters:
        -----------
        df : pd.DataFrame
            DataFrame with volatility data
        vol_col : str, optional
            Column name for volatility (if None, uses 'volatility_24')
        n_regimes : int
            Number of regimes to detect

        Returns:
        --------
        pd.DataFrame
            DataFrame with regime labels
        """
        if df is None or df.empty:
            logger.warning("Empty DataFrame provided to detect_volatility_regimes")
            return df

        # Make a copy to avoid modifying the original
        df_regimes = df.copy()

        # Determine volatility column
        if vol_col is None:
            # Try to find a suitable volatility column
            for w in self.window_sizes:
                col = f'volatility_{w}'
                if col in df_regimes.columns:
                    vol_col = col
                    break

            if vol_col is None:
                logger.warning("No volatility columns found in DataFrame")
                return df_regimes

        # Check for required column
        if vol_col not in df_regimes.columns:
            logger.warning(f"Volatility column '{vol_col}' not found in DataFrame")
            return df_regimes

        try:
            # Prepare data for clustering
            X = df_regimes[[vol_col]].dropna().values

            if len(X) < n_regimes:
                logger.warning(f"Not enough data points ({len(X)}) for {n_regimes} clusters")
                return df_regimes

            # Apply KMeans clustering
            kmeans = KMeans(n_clusters=n_regimes, random_state=42, n_init=10)
            labels = kmeans.fit_predict(X)

            # Get cluster centers
            centers = kmeans.cluster_centers_.flatten()

            # Order labels by volatility level (0 = lowest volatility)
            center_order = np.argsort(centers)
            remapping = {old_label: new_label for new_label, old_label in enumerate(center_order)}

            # Map the labels according to volatility level
            new_labels = np.array([remapping[label] for label in labels])

            # Create regime map for all data points (including those with NaN volatility)
            regime_map = pd.Series(index=df_regimes[vol_col].dropna().index, data=new_labels)

            # Add regime labels to the DataFrame
            df_regimes['volatility_regime'] = df_regimes.index.map(regime_map)

            # Fill NaN regimes with forward fill then backward fill
            df_regimes['volatility_regime'] = df_regimes['volatility_regime'].fillna(method='ffill').fillna(
                method='bfill')

            # Convert to integer
            df_regimes['volatility_regime'] = df_regimes['volatility_regime'].astype(int)

            # Add regime names
            regime_names = ['low', 'moderate', 'high']
            if n_regimes == 4:
                regime_names = ['low', 'moderate', 'high', 'extreme']
            elif n_regimes == 2:
                regime_names = ['low', 'high']

            if n_regimes <= len(regime_names):
                df_regimes['regime_name'] = df_regimes['volatility_regime'].apply(lambda x: regime_names[x])

            # Add regime levels (center values)
            for i, center in enumerate(sorted(centers)):
                df_regimes.loc[df_regimes['volatility_regime'] == i, 'regime_level'] = center

            logger.info(f"Detected {n_regimes} volatility regimes")
            return df_regimes

        except Exception as e:
            logger.error(f"Error detecting volatility regimes: {e}")
            return df_regimes

    def detect_regime_shifts(self, df, regime_col='volatility_regime', window=12):
        """
        Detect shifts in volatility regimes.

        Parameters:
        -----------
        df : pd.DataFrame
            DataFrame with regime labels
        regime_col : str
            Column name for regime labels
        window : int
            Window size for shift detection

        Returns:
        --------
        pd.DataFrame
            DataFrame with regime shift indicators
        """
        if df is None or df.empty:
            logger.warning("Empty DataFrame provided to detect_regime_shifts")
            return df

        # Make a copy to avoid modifying the original
        df_shifts = df.copy()

        # Check for required column
        if regime_col not in df_shifts.columns:
            logger.warning(f"Regime column '{regime_col}' not found in DataFrame")
            return df_shifts

        try:
            # Detect regime changes
            df_shifts['regime_change'] = df_shifts[regime_col] != df_shifts[regime_col].shift(1)

            # Calculate predominant regime over the window
            df_shifts['predominant_regime'] = df_shifts[regime_col].rolling(window=window).apply(
                lambda x: x.mode()[0] if len(x.mode()) > 0 else x.iloc[-1]
            )

            # Detect shifts in predominant regime
            df_shifts['regime_shift'] = df_shifts['predominant_regime'] != df_shifts['predominant_regime'].shift(1)
            # Calculate time since last regime change
            df_shifts['days_since_regime_change'] = 0

            current_count = 0
            for i in range(len(df_shifts)):
                if df_shifts['regime_change'].iloc[i]:
                    current_count = 0
                else:
                    current_count += 1
                df_shifts.iloc[i, df_shifts.columns.get_loc('days_since_regime_change')] = current_count

            # Convert to days (assuming hourly data)
            df_shifts['days_since_regime_change'] = df_shifts['days_since_regime_change'] / 24

            # Label regime shift points
            df_shifts['is_regime_shift_point'] = df_shifts['regime_shift'] & (
                        df_shifts['predominant_regime'] != df_shifts['predominant_regime'].shift(-1))

            # Calculate regime stability (percentage of the same regime in the rolling window)
            df_shifts['regime_stability'] = df_shifts[regime_col].rolling(window=window).apply(
                lambda x: x.value_counts().max() / len(x) if len(x) > 0 else 0
            )

            logger.info(f"Detected regime shifts with window {window}")
            return df_shifts

        except Exception as e:
            logger.error(f"Error detecting regime shifts: {e}")
            return df_shifts

    def analyze_regime_transitions(self, df, regime_col='volatility_regime'):
        """
        Analyze transitions between volatility regimes.

        Parameters:
        -----------
        df : pd.DataFrame
            DataFrame with regime labels
        regime_col : str
            Column name for regime labels

        Returns:
        --------
        pd.DataFrame, dict
            DataFrame with transition analysis and transition probability matrix
        """
        if df is None or df.empty:
            logger.warning("Empty DataFrame provided to analyze_regime_transitions")
            return df, {}

        # Make a copy to avoid modifying the original
        df_transitions = df.copy()

        # Check for required column
        if regime_col not in df_transitions.columns:
            logger.warning(f"Regime column '{regime_col}' not found in DataFrame")
            return df_transitions, {}

        try:
            # Calculate regime transitions
            df_transitions['next_regime'] = df_transitions[regime_col].shift(-1)

            # Get unique regimes
            regimes = sorted(df_transitions[regime_col].dropna().unique())
            n_regimes = len(regimes)

            # Initialize transition count matrix
            transition_counts = np.zeros((n_regimes, n_regimes))

            # Count transitions
            for i, current_regime in enumerate(regimes):
                for j, next_regime in enumerate(regimes):
                    transition_counts[i, j] = ((df_transitions[regime_col] == current_regime) &
                                               (df_transitions['next_regime'] == next_regime)).sum()

            # Calculate transition probabilities
            transition_probs = np.zeros_like(transition_counts)

            for i in range(n_regimes):
                row_sum = transition_counts[i].sum()
                if row_sum > 0:
                    transition_probs[i] = transition_counts[i] / row_sum

            # Convert to dictionary format
            transition_matrix = {
                'regimes': regimes.tolist(),
                'transition_counts': transition_counts.tolist(),
                'transition_probabilities': transition_probs.tolist()
            }

            # Calculate regime duration statistics
            regime_durations = {}

            for regime in regimes:
                # Find sequences of the same regime
                mask = df_transitions[regime_col] == regime

                # Calculate run lengths
                runs = np.diff(np.where(np.concatenate(([mask.iloc[0]],
                                                        mask.iloc[:-1] != mask.iloc[1:],
                                                        [True])))[0])

                if len(runs) > 0:
                    regime_durations[int(regime)] = {
                        'mean_duration': float(np.mean(runs)),
                        'median_duration': float(np.median(runs)),
                        'max_duration': int(np.max(runs)),
                        'min_duration': int(np.min(runs))
                    }

            transition_matrix['regime_durations'] = regime_durations

            logger.info(f"Analyzed regime transitions for {n_regimes} regimes")
            return df_transitions, transition_matrix

        except Exception as e:
            logger.error(f"Error analyzing regime transitions: {e}")
            return df_transitions, {}

    def calculate_adaptive_kama(self, df, price_col='close', er_period=10, fast_ef=0.666, slow_ef=0.0645):
        """
        Calculate Kaufman's Adaptive Moving Average (KAMA).

        Parameters:
        -----------
        df : pd.DataFrame
            DataFrame with price data
        price_col : str
            Column name for price data
        er_period : int
            Efficiency ratio period
        fast_ef : float
            Fast efficiency factor (default: 2/(2+1))
        slow_ef : float
            Slow efficiency factor (default: 2/(30+1))

        Returns:
        --------
        pd.DataFrame
            DataFrame with KAMA values
        """
        if df is None or df.empty:
            logger.warning("Empty DataFrame provided to calculate_adaptive_kama")
            return df

        # Make a copy to avoid modifying the original
        df_kama = df.copy()

        # Check for required column
        if price_col not in df_kama.columns:
            logger.warning(f"Price column '{price_col}' not found in DataFrame")
            return df_kama

        try:
            # Calculate price change
            df_kama['price_change'] = abs(df_kama[price_col] - df_kama[price_col].shift(er_period))

            # Calculate volatility (sum of absolute price changes over the period)
            df_kama['volatility'] = abs(df_kama[price_col].diff()).rolling(window=er_period).sum()

            # Calculate efficiency ratio
            df_kama['efficiency_ratio'] = df_kama['price_change'] / df_kama['volatility']
            df_kama['efficiency_ratio'] = df_kama['efficiency_ratio'].replace([np.inf, -np.inf], np.nan).fillna(0)

            # Calculate smoothing constant
            df_kama['smoothing_constant'] = (
                                                    df_kama['efficiency_ratio'] * (fast_ef - slow_ef) + slow_ef
                                            ) ** 2

            # Initialize KAMA
            kama = np.zeros(len(df_kama))

            # Set first KAMA value to first available price
            first_idx = df_kama[price_col].first_valid_index()
            if first_idx is not None:
                first_pos = df_kama.index.get_loc(first_idx)
                kama[first_pos] = df_kama[price_col].iloc[first_pos]

                # Calculate KAMA values
                for i in range(first_pos + 1, len(df_kama)):
                    kama[i] = kama[i - 1] + df_kama['smoothing_constant'].iloc[i] * (
                            df_kama[price_col].iloc[i] - kama[i - 1]
                    )

            # Add KAMA to DataFrame
            df_kama['kama'] = kama

            logger.info(f"Calculated adaptive KAMA with er_period={er_period}")
            return df_kama

        except Exception as e:
            logger.error(f"Error calculating adaptive KAMA: {e}")
            return df_kama

    def calculate_volume_weighted_kama(self, df, price_col='close', volume_col='volume', er_period=10):
        """
        Calculate Volume-Weighted Kaufman's Adaptive Moving Average.

        Parameters:
        -----------
        df : pd.DataFrame
            DataFrame with price and volume data
        price_col : str
            Column name for price data
        volume_col : str
            Column name for volume data
        er_period : int
            Efficiency ratio period

        Returns:
        --------
        pd.DataFrame
            DataFrame with Volume-Weighted KAMA values
        """
        if df is None or df.empty:
            logger.warning("Empty DataFrame provided to calculate_volume_weighted_kama")
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
            df_vwkama['price_change'] = abs(df_vwkama[price_col] - df_vwkama[price_col].shift(er_period))

            # Calculate volatility (sum of absolute price changes)
            df_vwkama['volatility'] = abs(df_vwkama[price_col].diff()).rolling(window=er_period).sum()

            # Calculate efficiency ratio
            df_vwkama['efficiency_ratio'] = df_vwkama['price_change'] / df_vwkama['volatility']
            df_vwkama['efficiency_ratio'] = df_vwkama['efficiency_ratio'].replace([np.inf, -np.inf], np.nan).fillna(
                0)

            # Adjust efficiency ratio based on volume
            df_vwkama['adjusted_er'] = df_vwkama['efficiency_ratio'] * df_vwkama['volume_factor']
            df_vwkama['adjusted_er'] = df_vwkama['adjusted_er'].clip(0, 1)

            # Calculate fast and slow EFs based on volume
            fast_ef = 2.0 / (2.0 + 1.0)
            slow_ef = 2.0 / (30.0 + 1.0)

            # Calculate smoothing constant
            df_vwkama['smoothing_constant'] = (
                                                      df_vwkama['adjusted_er'] * (fast_ef - slow_ef) + slow_ef
                                              ) ** 2

            # Initialize VW-KAMA
            vwkama = np.zeros(len(df_vwkama))

            # Set first KAMA value to first available price
            first_idx = df_vwkama[price_col].first_valid_index()
            if first_idx is not None:
                first_pos = df_vwkama.index.get_loc(first_idx)
                vwkama[first_pos] = df_vwkama[price_col].iloc[first_pos]

                # Calculate KAMA values
                for i in range(first_pos + 1, len(df_vwkama)):
                    vwkama[i] = vwkama[i - 1] + df_vwkama['smoothing_constant'].iloc[i] * (
                            df_vwkama[price_col].iloc[i] - vwkama[i - 1]
                    )

            # Add VW-KAMA to DataFrame
            df_vwkama['vw_kama'] = vwkama

            logger.info(f"Calculated volume-weighted KAMA with er_period={er_period}")
            return df_vwkama

        except Exception as e:
            logger.error(f"Error calculating volume-weighted KAMA: {e}")
            return df_vwkama

    def calculate_trend_strength(self, df, price_col='close', window=24):
        """
        Calculate trend strength using linear regression.

        Parameters:
        -----------
        df : pd.DataFrame
            DataFrame with price data
        price_col : str
            Column name for price data
        window : int
            Window size for trend analysis

        Returns:
        --------
        pd.DataFrame
            DataFrame with trend strength metrics
        """
        if df is None or df.empty:
            logger.warning("Empty DataFrame provided to calculate_trend_strength")
            return df

        # Make a copy to avoid modifying the original
        df_trend = df.copy()

        # Check for required column
        if price_col not in df_trend.columns:
            logger.warning(f"Price column '{price_col}' not found in DataFrame")
            return df_trend

        try:
            # Calculate linear regression for each rolling window
            df_trend['trend_slope'] = np.nan
            df_trend['trend_r2'] = np.nan
            df_trend['trend_pvalue'] = np.nan

            for i in range(window, len(df_trend)):
                # Get window data
                y = df_trend[price_col].iloc[i - window:i].values
                x = np.arange(window)

                # Calculate linear regression
                slope, intercept, r_value, p_value, std_err = stats.linregress(x, y)

                # Store results
                df_trend.iloc[i, df_trend.columns.get_loc('trend_slope')] = slope
                df_trend.iloc[i, df_trend.columns.get_loc('trend_r2')] = r_value ** 2
                df_trend.iloc[i, df_trend.columns.get_loc('trend_pvalue')] = p_value

            # Normalize trend slope by price level
            df_trend['price_level'] = df_trend[price_col].rolling(window=window).mean()
            df_trend['normalized_slope'] = df_trend['trend_slope'] / df_trend['price_level'] * 100

            # Classify trend direction and strength
            df_trend['trend_direction'] = np.where(df_trend['trend_slope'] > 0, 1, -1)
            df_trend['trend_direction'] = np.where(df_trend['trend_slope'] == 0, 0, df_trend['trend_direction'])

            # Trend strength based on R-squared and slope
            df_trend['trend_strength'] = df_trend['trend_r2'] * abs(df_trend['normalized_slope'])

            # Classify trend strength
            df_trend['trend_strength_category'] = pd.cut(
                df_trend['trend_strength'],
                bins=[-np.inf, 0.1, 0.3, 0.6, np.inf],
                labels=['weak', 'moderate', 'strong', 'very_strong']
            )

            logger.info(f"Calculated trend strength with window {window}")
            return df_trend

        except Exception as e:
            logger.error(f"Error calculating trend strength: {e}")
            return df_trend

    def analyze_volatility_trend_relation(self, df, vol_col=None, trend_col='trend_strength'):
        """
        Analyze the relationship between volatility and trend strength.

        Parameters:
        -----------
        df : pd.DataFrame
            DataFrame with volatility and trend data
        vol_col : str, optional
            Column name for volatility (if None, uses 'volatility_24')
        trend_col : str
            Column name for trend strength

        Returns:
        --------
        pd.DataFrame, dict
            DataFrame with analysis and correlation metrics
        """
        if df is None or df.empty:
            logger.warning("Empty DataFrame provided to analyze_volatility_trend_relation")
            return df, {}

        # Make a copy to avoid modifying the original
        df_analysis = df.copy()

        # Determine volatility column
        if vol_col is None:
            # Try to find a suitable volatility column
            for w in self.window_sizes:
                col = f'volatility_{w}'
                if col in df_analysis.columns:
                    vol_col = col
                    break

            if vol_col is None:
                logger.warning("No volatility columns found in DataFrame")
                return df_analysis, {}

        # Check for required columns
        if vol_col not in df_analysis.columns:
            logger.warning(f"Volatility column '{vol_col}' not found in DataFrame")
            return df_analysis, {}

        if trend_col not in df_analysis.columns:
            logger.warning(f"Trend column '{trend_col}' not found in DataFrame")
            return df_analysis, {}

        try:
            # Calculate correlation between volatility and trend strength
            correlation = df_analysis[vol_col].corr(df_analysis[trend_col])

            # Calculate volatility during uptrends and downtrends
            up_mask = df_analysis['trend_direction'] == 1
            down_mask = df_analysis['trend_direction'] == -1

            up_vol = df_analysis.loc[up_mask, vol_col].mean() if up_mask.any() else np.nan
            down_vol = df_analysis.loc[down_mask, vol_col].mean() if down_mask.any() else np.nan

            # Calculate volatility for different trend strength categories
            strength_categories = ['weak', 'moderate', 'strong', 'very_strong']
            strength_vol = {}

            for category in strength_categories:
                mask = df_analysis['trend_strength_category'] == category
                if mask.any():
                    strength_vol[category] = df_analysis.loc[mask, vol_col].mean()

            # Calculate volatility-adjusted trend strength
            df_analysis['vol_adjusted_trend'] = df_analysis[trend_col] / df_analysis[vol_col]

            metrics = {
                'volatility_trend_correlation': correlation,
                'uptrend_volatility': up_vol,
                'downtrend_volatility': down_vol,
                'volatility_by_trend_strength': strength_vol,
                'vol_trend_ratio': (up_vol / down_vol) if (down_vol and down_vol > 0) else np.nan
            }

            logger.info(f"Analyzed volatility-trend relation: correlation={correlation:.3f}")
            return df_analysis, metrics

        except Exception as e:
            logger.error(f"Error analyzing volatility-trend relation: {e}")
            return df_analysis, {}

    def run_analysis(self, df, price_col='close', volume_col='volume'):
        """
        Run a comprehensive meso-volatility analysis.

        Parameters:
        -----------
        df : pd.DataFrame
            DataFrame with OHLCV data
        price_col : str
            Column name for price data
        volume_col : str
            Column name for volume data

        Returns:
        --------
        tuple
            (DataFrame with analysis results, dict with metrics)
        """
        if df is None or df.empty:
            logger.warning("Empty DataFrame provided to run_analysis")
            return df, {}

        logger.info("Running comprehensive meso-volatility analysis")

        try:
            # Step 1: Clean data
            df_clean = self.normalizer.clean_ohlcv_data(df)

            # Step 2: Calculate returns
            df_returns = self.calculate_returns(df_clean, price_col=price_col)

            # Step 3: Calculate rolling volatility
            df_vol = self.calculate_rolling_volatility(df_returns)

            # Step 4: Calculate realized volatility
            df_vol = self.calculate_realized_volatility(df_vol, window=24)

            # Step 5: Detect volatility regimes
            df_vol = self.detect_volatility_regimes(df_vol, n_regimes=3)

            # Step 6: Detect regime shifts
            df_vol = self.detect_regime_shifts(df_vol)

            # Step 7: Analyze regime transitions
            df_vol, transition_matrix = self.analyze_regime_transitions(df_vol)

            # Step 8: Calculate adaptive KAMA
            df_vol = self.calculate_adaptive_kama(df_vol, price_col=price_col)

            # Step 9: Calculate volume-weighted KAMA if volume data is available
            if volume_col in df_vol.columns:
                df_vol = self.calculate_volume_weighted_kama(df_vol, price_col=price_col, volume_col=volume_col)

            # Step 10: Calculate trend strength
            df_vol = self.calculate_trend_strength(df_vol, price_col=price_col)

            # Step 11: Analyze volatility-trend relation
            df_vol, vol_trend_metrics = self.analyze_volatility_trend_relation(df_vol)

            # Compute overall metrics
            metrics = {
                'volatility_regimes': {
                    'regime_count': int(df_vol['volatility_regime'].nunique()),
                    'regime_distribution': df_vol['volatility_regime'].value_counts().to_dict(),
                    'current_regime': int(df_vol['volatility_regime'].iloc[-1]) if len(df_vol) > 0 else None,
                    'regime_stability': float(
                        df_vol['regime_stability'].iloc[-1]) if 'regime_stability' in df_vol.columns and len(
                        df_vol) > 0 else None
                },
                'regime_transitions': transition_matrix,
                'volatility_trend_relation': vol_trend_metrics,
                'current_volatility': {
                    'realized_vol_24h': float(df_vol['realized_volatility_24_annualized'].iloc[
                                                  -1]) if 'realized_volatility_24_annualized' in df_vol.columns and len(
                        df_vol) > 0 else None,
                    'trend_direction': int(
                        df_vol['trend_direction'].iloc[-1]) if 'trend_direction' in df_vol.columns and len(
                        df_vol) > 0 else None,
                    'trend_strength': float(
                        df_vol['trend_strength'].iloc[-1]) if 'trend_strength' in df_vol.columns and len(
                        df_vol) > 0 else None
                }
            }

            logger.info(
                f"Completed meso-volatility analysis with {metrics['volatility_regimes']['regime_count']} regimes")
            return df_vol, metrics

        except Exception as e:
            logger.error(f"Error running meso-volatility analysis: {e}")
            return df, {}

# Factory function to get a meso-volatility analyzer
def get_meso_volatility_analyzer(window_sizes=None, use_log_returns=True):
    """
    Get a configured meso-volatility analyzer.

    Parameters:
    -----------
    window_sizes : list, optional
        List of window sizes for rolling volatility calculation
    use_log_returns : bool
        If True, use log returns for volatility calculation

    Returns:
    --------
    MesoVolatilityAnalyzer
        Configured analyzer instance
    """
    return MesoVolatilityAnalyzer(window_sizes=window_sizes, use_log_returns=use_log_returns)