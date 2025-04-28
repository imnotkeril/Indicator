"""
Module for micro-volatility analysis.
Handles analysis of volatility on short timeframes (minutes).
"""
import os
import sys
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import logging
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, project_root)
from crypto_volatility_indicator.utils.logger import get_logger
from crypto_volatility_indicator.data.processors.normalizer import get_data_normalizer
from crypto_volatility_indicator.data.processors.filters import get_data_filter

# Set up logger
logger = get_logger(__name__)

class MicroVolatilityAnalyzer:
    """
    Analyzer for micro-volatility (minute timeframes).

    This class is responsible for analyzing and calculating volatility
    on very short timeframes, detecting microbursts, and identifying
    patterns in high-frequency volatility.
    """

    def __init__(self, window_sizes=None, use_log_returns=True):
        """
        Initialize the micro-volatility analyzer.

        Parameters:
        -----------
        window_sizes : list, optional
            List of window sizes for rolling volatility calculation
        use_log_returns : bool
            If True, use log returns for volatility calculation
        """
        self.window_sizes = window_sizes or [5, 10, 20, 50, 100]
        self.use_log_returns = use_log_returns

        # Initialize data processing utilities
        self.normalizer = get_data_normalizer()
        self.filter = get_data_filter()

        logger.info(f"MicroVolatilityAnalyzer initialized with window sizes: {self.window_sizes}")

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

                # Annualize (assuming minute data, multiply by sqrt(525600) for annual)
                # For simplicity, using sqrt(252 * 1440)
                df_vol[f'volatility_{window}_annualized'] = vol * np.sqrt(252 * 1440)

            logger.info(f"Calculated historical volatility for {len(self.window_sizes)} window sizes")
            return df_vol

        except Exception as e:
            logger.error(f"Error calculating historical volatility: {e}")
            return df_vol

    # Остальные методы класса остаются без изменений

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

                # Annualize (assuming minute data, multiply by sqrt(525600) for annual)
                # For simplicity, we'll use sqrt(252 * 1440) = sqrt(362880)
                df_vol[f'volatility_{window}_annualized'] = df_vol[f'volatility_{window}'] * np.sqrt(252 * 1440)

            logger.info(f"Calculated rolling volatility for {len(self.window_sizes)} window sizes")
            return df_vol

        except Exception as e:
            logger.error(f"Error calculating rolling volatility: {e}")
            return df_vol

    def calculate_parkinson_volatility(self, df, window=20):
        """
        Calculate Parkinson volatility using high-low price range.

        Parameters:
        -----------
        df : pd.DataFrame
            DataFrame with high and low prices
        window : int
            Window size for rolling calculation

        Returns:
        --------
        pd.DataFrame
            DataFrame with Parkinson volatility
        """
        if df is None or df.empty:
            logger.warning("Empty DataFrame provided to calculate_parkinson_volatility")
            return df

        # Make a copy to avoid modifying the original
        df_vol = df.copy()

        # Check for required columns
        if 'high' not in df_vol.columns or 'low' not in df_vol.columns:
            logger.warning("High and low price columns required for Parkinson volatility")
            return df_vol

        try:
            # Calculate log high-low range
            df_vol['hl_range'] = np.log(df_vol['high'] / df_vol['low'])

            # Calculate Parkinson volatility
            df_vol['parkinson_volatility'] = np.sqrt(
                1 / (4 * np.log(2)) *
                df_vol['hl_range'].pow(2).rolling(window=window).mean()
            )

            # Annualize (assuming minute data)
            df_vol['parkinson_volatility_annualized'] = df_vol['parkinson_volatility'] * np.sqrt(252 * 1440)

            logger.info(f"Calculated Parkinson volatility with window {window}")
            return df_vol

        except Exception as e:
            logger.error(f"Error calculating Parkinson volatility: {e}")
            return df_vol

    def calculate_gk_volatility(self, df, window=20):
        """
        Calculate Garman-Klass volatility using open, high, low, close prices.

        Parameters:
        -----------
        df : pd.DataFrame
            DataFrame with OHLC prices
        window : int
            Window size for rolling calculation

        Returns:
        --------
        pd.DataFrame
            DataFrame with Garman-Klass volatility
        """
        if df is None or df.empty:
            logger.warning("Empty DataFrame provided to calculate_gk_volatility")
            return df

        # Make a copy to avoid modifying the original
        df_vol = df.copy()

        # Check for required columns
        required_cols = ['open', 'high', 'low', 'close']
        missing_cols = [col for col in required_cols if col not in df_vol.columns]

        if missing_cols:
            logger.warning(f"Missing required columns for Garman-Klass volatility: {missing_cols}")
            return df_vol

        try:
            # Calculate log returns for each component
            df_vol['log_hl'] = np.log(df_vol['high'] / df_vol['low'])
            df_vol['log_co'] = np.log(df_vol['close'] / df_vol['open'])

            # Calculate Garman-Klass volatility
            df_vol['gk_volatility'] = np.sqrt(
                0.5 * df_vol['log_hl'].pow(2) -
                (2 * np.log(2) - 1) * df_vol['log_co'].pow(2)
            ).rolling(window=window).mean()

            # Annualize (assuming minute data)
            df_vol['gk_volatility_annualized'] = df_vol['gk_volatility'] * np.sqrt(252 * 1440)

            logger.info(f"Calculated Garman-Klass volatility with window {window}")
            return df_vol

        except Exception as e:
            logger.error(f"Error calculating Garman-Klass volatility: {e}")
            return df_vol

    def detect_volatility_bursts(self, df, vol_col=None, threshold=2.0):
        """
        Detect bursts in micro-volatility.

        Parameters:
        -----------
        df : pd.DataFrame
            DataFrame with volatility data
        vol_col : str, optional
            Column name for volatility (if None, uses 'volatility_20')
        threshold : float
            Threshold for burst detection (in standard deviations)

        Returns:
        --------
        pd.DataFrame
            DataFrame with volatility bursts
        """
        if df is None or df.empty:
            logger.warning("Empty DataFrame provided to detect_volatility_bursts")
            return df

        # Make a copy to avoid modifying the original
        df_bursts = df.copy()

        # Determine volatility column
        if vol_col is None:
            # Try to find a suitable volatility column
            vol_candidates = [f'volatility_{w}' for w in self.window_sizes]
            available_cols = [col for col in vol_candidates if col in df_bursts.columns]

            if not available_cols:
                logger.warning("No volatility columns found in DataFrame")
                return df_bursts

            vol_col = available_cols[0]

            # Check for required column
        if vol_col not in df_bursts.columns:
            logger.warning(f"Volatility column '{vol_col}' not found in DataFrame")
            return df_bursts

        try:
            # Calculate rolling mean and standard deviation of volatility
            vol_mean = df_bursts[vol_col].rolling(window=100).mean()
            vol_std = df_bursts[vol_col].rolling(window=100).std()

            # Detect bursts where volatility exceeds mean + threshold * std
            df_bursts['vol_zscore'] = (df_bursts[vol_col] - vol_mean) / vol_std
            df_bursts['is_burst'] = df_bursts['vol_zscore'] > threshold

            # Label burst episodes (consecutive burst periods)
            burst_groups = []
            current_burst = None

            for i, row in df_bursts.iterrows():
                if row['is_burst'] and current_burst is None:
                    # Start of a new burst
                    current_burst = {'start': i, 'max_zscore': row['vol_zscore']}

                elif row['is_burst'] and current_burst is not None:
                    # Continuing burst, update max_zscore if needed
                    if row['vol_zscore'] > current_burst['max_zscore']:
                        current_burst['max_zscore'] = row['vol_zscore']

                elif not row['is_burst'] and current_burst is not None:
                    # End of a burst
                    current_burst['end'] = i
                    burst_groups.append(current_burst)
                    current_burst = None

            # Add the last burst if it's still open
            if current_burst is not None:
                current_burst['end'] = df_bursts.index[-1]
                burst_groups.append(current_burst)

            # Create burst labels
            df_bursts['burst_id'] = 0

            for i, burst in enumerate(burst_groups, 1):
                df_bursts.loc[burst['start']:burst['end'], 'burst_id'] = i

            logger.info(f"Detected {len(burst_groups)} volatility bursts with threshold {threshold}")

            # Add burst metadata
            df_bursts['burst_start'] = False
            df_bursts['burst_end'] = False
            df_bursts['burst_max'] = False

            for burst in burst_groups:
                # Mark start and end
                df_bursts.loc[burst['start'], 'burst_start'] = True
                df_bursts.loc[burst['end'], 'burst_end'] = True

                # Mark max zscore in each burst
                burst_segment = df_bursts.loc[burst['start']:burst['end']]
                max_idx = burst_segment['vol_zscore'].idxmax()
                df_bursts.loc[max_idx, 'burst_max'] = True

            return df_bursts

        except Exception as e:
            logger.error(f"Error detecting volatility bursts: {e}")
            return df_bursts

    def calculate_micropatterns(self, df, return_col=None, window=5):
        """
        Detect micro-patterns in returns.

        Parameters:
        -----------
        df : pd.DataFrame
            DataFrame with returns
        return_col : str, optional
            Column name for returns (if None, uses 'log_return' or 'pct_return')
        window : int
            Window size for pattern detection

        Returns:
        --------
        pd.DataFrame
            DataFrame with micro-patterns
        """
        if df is None or df.empty:
            logger.warning("Empty DataFrame provided to calculate_micropatterns")
            return df

        # Make a copy to avoid modifying the original
        df_patterns = df.copy()

        # Determine return column
        if return_col is None:
            return_col = 'log_return' if self.use_log_returns else 'pct_return'

        # Check for required column
        if return_col not in df_patterns.columns:
            logger.warning(f"Return column '{return_col}' not found in DataFrame")
            return df_patterns

        try:
            # Calculate sign of returns
            df_patterns['return_sign'] = np.sign(df_patterns[return_col])

            # Calculate consecutive up and down moves
            df_patterns['streak'] = (df_patterns['return_sign'] != df_patterns['return_sign'].shift(1)).cumsum()

            # Calculate streak length for each point
            df_patterns['streak_length'] = df_patterns.groupby('streak').cumcount() + 1

            # Identify common micro-patterns
            # 1. Reversal after n consecutive moves in one direction
            for n in range(3, window + 1):
                # Up streak followed by down
                mask_up_reversal = (df_patterns['return_sign'] == -1) & (
                            df_patterns['return_sign'].shift(1) == 1) & (df_patterns['streak_length'].shift(1) >= n)
                df_patterns[f'up_{n}_reversal'] = mask_up_reversal

                # Down streak followed by up
                mask_down_reversal = (df_patterns['return_sign'] == 1) & (
                            df_patterns['return_sign'].shift(1) == -1) & (
                                                 df_patterns['streak_length'].shift(1) >= n)
                df_patterns[f'down_{n}_reversal'] = mask_down_reversal

            # 2. Micro-patterns: 3 consecutive moves
            df_patterns['pattern_3'] = ''

            for i in range(3, len(df_patterns)):
                pattern = ''.join(
                    ['U' if sign == 1 else 'D' for sign in df_patterns['return_sign'].iloc[i - 3:i].values])
                df_patterns.loc[df_patterns.index[i], 'pattern_3'] = pattern

            # Count occurrences of each pattern
            pattern_counts = df_patterns['pattern_3'].value_counts()
            logger.info(f"Most common 3-period patterns: {pattern_counts.head(3).to_dict()}")

            return df_patterns

        except Exception as e:
            logger.error(f"Error calculating micro-patterns: {e}")
            return df_patterns

    def analyze_volume_volatility_relation(self, df, vol_col=None, volume_col='volume'):
        """
        Analyze the relationship between volume and volatility.

        Parameters:
        -----------
        df : pd.DataFrame
            DataFrame with volatility and volume data
        vol_col : str, optional
            Column name for volatility (if None, uses 'volatility_20')
        volume_col : str
            Column name for volume

        Returns:
        --------
        pd.DataFrame, dict
            DataFrame with analysis and correlation metrics
        """
        if df is None or df.empty:
            logger.warning("Empty DataFrame provided to analyze_volume_volatility_relation")
            return df, {}

        # Make a copy to avoid modifying the original
        df_analysis = df.copy()

        # Determine volatility column
        if vol_col is None:
            # Try to find a suitable volatility column
            vol_candidates = [f'volatility_{w}' for w in self.window_sizes]
            available_cols = [col for col in vol_candidates if col in df_analysis.columns]

            if not available_cols:
                logger.warning("No volatility columns found in DataFrame")
                return df_analysis, {}

            vol_col = available_cols[0]

        # Check for required columns
        if vol_col not in df_analysis.columns:
            logger.warning(f"Volatility column '{vol_col}' not found in DataFrame")
            return df_analysis, {}

        if volume_col not in df_analysis.columns:
            logger.warning(f"Volume column '{volume_col}' not found in DataFrame")
            return df_analysis, {}

        try:
            # Log-transform volume
            df_analysis[f'{volume_col}_log'] = np.log1p(df_analysis[volume_col])

            # Calculate rolling correlation
            for window in [50, 100, 200]:
                df_analysis[f'vol_volume_corr_{window}'] = df_analysis[vol_col].rolling(window=window).corr(
                    df_analysis[f'{volume_col}_log'])

            # Normalize volume
            df_analysis[f'{volume_col}_z'] = (df_analysis[volume_col] - df_analysis[volume_col].rolling(
                window=100).mean()) / df_analysis[volume_col].rolling(window=100).std()

            # Normalize volatility
            df_analysis[f'{vol_col}_z'] = (df_analysis[vol_col] - df_analysis[vol_col].rolling(window=100).mean()) / \
                                          df_analysis[vol_col].rolling(window=100).std()

            # Calculate volume-volatility ratio
            df_analysis['vol_volume_ratio'] = df_analysis[vol_col] / (
                        df_analysis[volume_col] + 1)  # Add 1 to avoid division by zero

            # Calculate metrics
            overall_corr = df_analysis[vol_col].corr(df_analysis[f'{volume_col}_log'])
            recent_corr = df_analysis.iloc[-100:][vol_col].corr(df_analysis.iloc[-100:][f'{volume_col}_log'])

            metrics = {
                'overall_correlation': overall_corr,
                'recent_correlation': recent_corr,
                'volume_volatility_ratio_mean': df_analysis['vol_volume_ratio'].mean(),
                'volume_volatility_ratio_std': df_analysis['vol_volume_ratio'].std()
            }

            logger.info(f"Calculated volume-volatility relation metrics: correlation={overall_corr:.3f}")
            return df_analysis, metrics

        except Exception as e:
            logger.error(f"Error analyzing volume-volatility relation: {e}")
            return df_analysis, {}

    def analyze_intraday_patterns(self, df, vol_col=None):
        """
        Analyze intraday patterns in micro-volatility.

        Parameters:
        -----------
        df : pd.DataFrame
            DataFrame with volatility data and datetime index
        vol_col : str, optional
            Column name for volatility (if None, uses 'volatility_20')

        Returns:
        --------
        pd.DataFrame, dict
            DataFrame with hour-of-day analysis and metrics
        """
        if df is None or df.empty:
            logger.warning("Empty DataFrame provided to analyze_intraday_patterns")
            return df, {}

        # Make a copy to avoid modifying the original
        df_analysis = df.copy()

        # Determine volatility column
        if vol_col is None:
            # Try to find a suitable volatility column
            vol_candidates = [f'volatility_{w}' for w in self.window_sizes]
            available_cols = [col for col in vol_candidates if col in df_analysis.columns]

            if not available_cols:
                logger.warning("No volatility columns found in DataFrame")
                return df_analysis, {}

            vol_col = available_cols[0]

        # Check for required column
        if vol_col not in df_analysis.columns:
            logger.warning(f"Volatility column '{vol_col}' not found in DataFrame")
            return df_analysis, {}

        try:
            # Check if index is datetime
            if not isinstance(df_analysis.index, pd.DatetimeIndex):
                logger.warning("DataFrame index is not datetime, cannot analyze intraday patterns")
                return df_analysis, {}

            # Extract hour of day
            df_analysis['hour'] = df_analysis.index.hour

            # Calculate average volatility by hour
            hourly_vol = df_analysis.groupby('hour')[vol_col].mean()

            # Calculate volatility ratio relative to daily average
            daily_avg = df_analysis[vol_col].mean()
            hourly_vol_ratio = hourly_vol / daily_avg

            # Find peak and trough hours
            peak_hour = hourly_vol.idxmax()
            peak_value = hourly_vol.max()

            trough_hour = hourly_vol.idxmin()
            trough_value = hourly_vol.min()

            # Calculate relative volatility for each hour
            for hour in range(24):
                df_analysis.loc[df_analysis['hour'] == hour, 'vol_hour_ratio'] = hourly_vol_ratio.get(hour, 1.0)

            metrics = {
                'peak_hour': peak_hour,
                'peak_value': peak_value,
                'trough_hour': trough_hour,
                'trough_value': trough_value,
                'max_min_ratio': peak_value / trough_value if trough_value > 0 else float('inf'),
                'hourly_volatility': hourly_vol.to_dict(),
                'hourly_volatility_ratio': hourly_vol_ratio.to_dict()
            }

            logger.info(f"Analyzed intraday patterns: peak_hour={peak_hour}, trough_hour={trough_hour}")
            return df_analysis, metrics

        except Exception as e:
            logger.error(f"Error analyzing intraday patterns: {e}")
            return df_analysis, {}

    def run_analysis(self, df, price_col='close', volume_col='volume'):
        """
        Run a comprehensive micro-volatility analysis.

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

        logger.info("Running comprehensive micro-volatility analysis")

        try:
            # Step 1: Clean data
            df_clean = self.normalizer.clean_ohlcv_data(df)

            # Step 2: Calculate returns
            df_returns = self.calculate_returns(df_clean, price_col=price_col)

            # Step 3: Calculate rolling volatility
            df_vol = self.calculate_rolling_volatility(df_returns)

            # Step 4: Calculate Parkinson volatility
            if all(col in df_clean.columns for col in ['high', 'low']):
                df_vol = self.calculate_parkinson_volatility(df_vol)

            # Step 5: Calculate Garman-Klass volatility
            if all(col in df_clean.columns for col in ['open', 'high', 'low', 'close']):
                df_vol = self.calculate_gk_volatility(df_vol)

            # Step 6: Detect volatility bursts
            df_vol = self.detect_volatility_bursts(df_vol)

            # Step 7: Calculate micro-patterns
            df_vol = self.calculate_micropatterns(df_vol)

            # Step 8: Analyze volume-volatility relation
            df_vol, vol_volume_metrics = self.analyze_volume_volatility_relation(df_vol, volume_col=volume_col)

            # Step 9: Analyze intraday patterns
            df_vol, intraday_metrics = self.analyze_intraday_patterns(df_vol)

            # Compute overall metrics
            metrics = {
                'volume_volatility_relation': vol_volume_metrics,
                'intraday_patterns': intraday_metrics,
                'burst_count': int(df_vol['is_burst'].sum()),
                'burst_duration_avg': 0,
                'volatility_mean': df_vol[
                    f'volatility_{self.window_sizes[1]}'].mean() if f'volatility_{self.window_sizes[1]}' in df_vol.columns else None,
                'volatility_std': df_vol[
                    f'volatility_{self.window_sizes[1]}'].std() if f'volatility_{self.window_sizes[1]}' in df_vol.columns else None
            }

            # Calculate average burst duration if there are bursts
            if metrics['burst_count'] > 0:
                burst_durations = []
                for burst_id in range(1, int(df_vol['burst_id'].max()) + 1):
                    burst_df = df_vol[df_vol['burst_id'] == burst_id]
                    if not burst_df.empty:
                        burst_durations.append(len(burst_df))

                metrics['burst_duration_avg'] = sum(burst_durations) / len(
                    burst_durations) if burst_durations else 0

            logger.info(f"Completed micro-volatility analysis: {metrics['burst_count']} bursts detected")
            return df_vol, metrics

        except Exception as e:
            logger.error(f"Error running micro-volatility analysis: {e}")
            return df, {}

        # Factory function to get a micro-volatility analyzer

def get_micro_volatility_analyzer(window_sizes=None, use_log_returns=True):
    """
    Get a configured micro-volatility analyzer.

    Parameters:
    -----------
    window_sizes : list, optional
        List of window sizes for rolling volatility calculation
    use_log_returns : bool
        If True, use log returns for volatility calculation

    Returns:
    --------
    MicroVolatilityAnalyzer
        Configured analyzer instance
    """
    return MicroVolatilityAnalyzer(window_sizes=window_sizes, use_log_returns=use_log_returns)