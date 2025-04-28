"""
Module for macro-volatility analysis.
Handles analysis of volatility on long timeframes (days).
"""
import os
import sys
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import logging
from scipy import stats
from statsmodels.tsa.stattools import adfuller, acf
from statsmodels.tsa.seasonal import seasonal_decompose
import statsmodels.api as sm
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, project_root)
from crypto_volatility_indicator.utils.logger import get_logger
from crypto_volatility_indicator.data.processors.normalizer import get_data_normalizer
from crypto_volatility_indicator.data.processors.filters import get_data_filter

# Set up logger
logger = get_logger(__name__)

class MacroVolatilityAnalyzer:
    """
    Analyzer for macro-volatility (day timeframes).

    This class is responsible for analyzing and calculating volatility
    on long timeframes, identifying long-term trends, cycles, and
    structural changes in volatility.
    """

    def __init__(self, window_sizes=None, use_log_returns=True):
        """
        Initialize the macro-volatility analyzer.

        Parameters:
        -----------
        window_sizes : list, optional
            List of window sizes for rolling volatility calculation
        use_log_returns : bool
            If True, use log returns for volatility calculation
        """
        self.window_sizes = window_sizes or [5, 10, 20, 50, 100, 200]
        self.use_log_returns = use_log_returns

        # Initialize data processing utilities
        self.normalizer = get_data_normalizer()
        self.filter = get_data_filter()

        logger.info(f"MacroVolatilityAnalyzer initialized with window sizes: {self.window_sizes}")

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

                # Annualize (assuming daily data, multiply by sqrt(252))
                df_vol[f'volatility_{window}_annualized'] = vol * np.sqrt(252)

            logger.info(f"Calculated historical volatility for {len(self.window_sizes)} window sizes")
            return df_vol

        except Exception as e:
            logger.error(f"Error calculating historical volatility: {e}")
            return df_vol


    def calculate_garch_volatility(self, df, return_col=None, p=1, q=1):
        """
        Calculate GARCH volatility model.

        Parameters:
        -----------
        df : pd.DataFrame
            DataFrame with returns
        return_col : str, optional
            Column name for returns (if None, uses 'log_return' or 'pct_return')
        p : int
            GARCH lag order
        q : int
            ARCH lag order

        Returns:
        --------
        pd.DataFrame
            DataFrame with GARCH volatility
        """
        if df is None or df.empty:
            logger.warning("Empty DataFrame provided to calculate_garch_volatility")
            return df

        # Make a copy to avoid modifying the original
        df_garch = df.copy()

        # Determine return column
        if return_col is None:
            return_col = 'log_return' if self.use_log_returns else 'pct_return'

        # Check for required column
        if return_col not in df_garch.columns:
            logger.warning(f"Return column '{return_col}' not found in DataFrame")
            return df_garch

        try:
            # Use arch package for GARCH modeling
            from arch import arch_model

            # Remove NaN values
            returns = df_garch[return_col].dropna().values

            if len(returns) < 100:
                logger.warning("Not enough data points for GARCH model")
                return df_garch

            # Fit GARCH model
            model = arch_model(returns, vol='GARCH', p=p, q=q, mean='Zero', rescale=False)
            model_fit = model.fit(disp='off')

            # Get conditional volatility
            conditional_vol = model_fit.conditional_volatility

            # Create series with same index as original returns
            vol_index = df_garch[return_col].dropna().index
            garch_vol = pd.Series(conditional_vol, index=vol_index)

            # Annualize volatility
            garch_vol_annualized = garch_vol * np.sqrt(252)

            # Add to DataFrame
            df_garch['garch_volatility'] = garch_vol
            df_garch['garch_volatility_annualized'] = garch_vol_annualized

            # Add forecast if available
            try:
                # Forecast 30 days ahead
                forecast = model_fit.forecast(horizon=30)
                forecast_vol = forecast.variance.values[-1]  # Last value is 30-day forecast

                # Annualize the forecast
                forecast_vol_annualized = np.sqrt(forecast_vol) * np.sqrt(252)

                # Add to metrics
                df_garch['garch_forecast_30d'] = None
                df_garch.iloc[-1, df_garch.columns.get_loc('garch_forecast_30d')] = forecast_vol_annualized
            except:
                logger.warning("Could not generate GARCH forecast")

            # Add GARCH parameters
            df_garch['garch_params'] = None
            df_garch.iloc[-1, df_garch.columns.get_loc('garch_params')] = str(model_fit.params)

            logger.info(f"Calculated GARCH({p},{q}) volatility model")
            return df_garch

        except ImportError:
            logger.warning("arch package not available for GARCH modeling")
            return df_garch
        except Exception as e:
            logger.error(f"Error calculating GARCH volatility: {e}")
            return df_garch

    def detect_volatility_cycles(self, df, vol_col=None, max_lag=100):
        """
        Detect cycles in volatility time series.

        Parameters:
        -----------
        df : pd.DataFrame
            DataFrame with volatility data
        vol_col : str, optional
            Column name for volatility (if None, uses 'volatility_20_annualized')
        max_lag : int
            Maximum lag for autocorrelation calculation

        Returns:
        --------
        pd.DataFrame, dict
            DataFrame with cycle information and cycle metrics
        """
        if df is None or df.empty:
            logger.warning("Empty DataFrame provided to detect_volatility_cycles")
            return df, {}

        # Make a copy to avoid modifying the original
        df_cycles = df.copy()

        # Determine volatility column
        if vol_col is None:
            # Try to find a suitable volatility column
            for w in self.window_sizes:
                col = f'volatility_{w}_annualized'
                if col in df_cycles.columns:
                    vol_col = col
                    break

            if vol_col is None:
                logger.warning("No volatility columns found in DataFrame")
                return df_cycles, {}

        # Check for required column
        if vol_col not in df_cycles.columns:
            logger.warning(f"Volatility column '{vol_col}' not found in DataFrame")
            return df_cycles, {}

        try:
            # Calculate autocorrelation
            volatility = df_cycles[vol_col].dropna()

            if len(volatility) < max_lag * 2:
                logger.warning(f"Not enough data points ({len(volatility)}) for cycle detection")
                return df_cycles, {}

            # Calculate autocorrelation function
            acf_values = acf(volatility, nlags=max_lag)

            # Find peaks in autocorrelation (potential cycles)
            from scipy.signal import find_peaks
            peaks, _ = find_peaks(acf_values[1:], height=0.1)  # Skip lag 0

            # Add 1 to get the correct lag (since we skipped lag 0)
            peaks = peaks + 1

            # Calculate cycle strengths
            cycle_strengths = acf_values[peaks]

            # Get strongest cycles
            if len(peaks) > 0:
                # Sort by strength
                cycles_sorted = sorted(zip(peaks, cycle_strengths), key=lambda x: x[1], reverse=True)

                # Get top 3 cycles (or fewer if less available)
                top_cycles = cycles_sorted[:min(3, len(cycles_sorted))]

                # Store cycle information
                cycles = []

                for lag, strength in top_cycles:
                    cycles.append({
                        'length': int(lag),
                        'strength': float(strength),
                        'description': f"{lag}-day cycle"
                    })

                # Add cycle decomposition
                try:
                    # Decompose the volatility series into trend, seasonal, and residual components
                    if len(volatility) >= 2 * max(peak for peak, _ in top_cycles):
                        # Use the strongest cycle for decomposition
                        strongest_cycle = top_cycles[0][0]

                        # Decompose time series
                        decomposition = seasonal_decompose(
                            volatility,
                            model='multiplicative',
                            period=strongest_cycle,
                            extrapolate_trend='freq'
                        )

                        # Add decomposed components to DataFrame
                        df_cycles['volatility_trend'] = None
                        df_cycles['volatility_seasonal'] = None
                        df_cycles['volatility_residual'] = None

                        # Map decomposed values back to original DataFrame
                        # (handling potentially different indices)
                        trend_series = decomposition.trend
                        seasonal_series = decomposition.seasonal
                        residual_series = decomposition.resid

                        for idx in volatility.index:
                            if idx in trend_series.index:
                                df_cycles.loc[idx, 'volatility_trend'] = trend_series.loc[idx]
                                df_cycles.loc[idx, 'volatility_seasonal'] = seasonal_series.loc[idx]
                                df_cycles.loc[idx, 'volatility_residual'] = residual_series.loc[idx]
                except Exception as decomp_error:
                    logger.warning(f"Error in cycle decomposition: {decomp_error}")

                # Calculate current cycle phase for each detected cycle
                for i, (lag, strength) in enumerate(top_cycles):
                    # Calculate phase (0 to 1, where 0/1 = cycle start/end)
                    if len(volatility) >= lag:
                        # Create a synthetic cycle with the detected length
                        synthetic_cycle = np.sin(np.arange(lag) * 2 * np.pi / lag)

                        # Get the last 'lag' values of volatility
                        recent_volatility = volatility.iloc[-lag:].values

                        # Normalize
                        recent_volatility = (recent_volatility - recent_volatility.mean()) / recent_volatility.std()

                        # Find the phase by correlating with shifts of the synthetic cycle
                        correlations = []
                        for shift in range(lag):
                            shifted_cycle = np.roll(synthetic_cycle, shift)
                            correlation = np.corrcoef(recent_volatility, shifted_cycle)[0, 1]
                            correlations.append(correlation)

                        # Best shift is the one with highest correlation
                        best_shift = np.argmax(correlations)

                        # Convert to phase (0 to 1)
                        phase = (lag - best_shift) % lag / lag

                        # Add to cycles
                        cycles[i]['current_phase'] = float(phase)
                        cycles[i]['phase_description'] = self._describe_cycle_phase(phase)

                # Create metrics dictionary
                cycle_metrics = {
                    'detected_cycles': cycles,
                    'primary_cycle_length': int(top_cycles[0][0]) if top_cycles else None,
                    'primary_cycle_strength': float(top_cycles[0][1]) if top_cycles else None,
                    'autocorrelation': acf_values.tolist()
                }

                logger.info(f"Detected {len(cycles)} volatility cycles")
                return df_cycles, cycle_metrics
            else:
                logger.info("No significant volatility cycles detected")
                return df_cycles, {'detected_cycles': []}

        except Exception as e:
            logger.error(f"Error detecting volatility cycles: {e}")
            return df_cycles, {}

    def _describe_cycle_phase(self, phase):
        """
        Provide a descriptive label for a cycle phase.

        Parameters:
        -----------
        phase : float
            Cycle phase (0 to 1)

        Returns:
        --------
        str
            Descriptive label
        """
        if phase < 0.125:
            return "starting new cycle"
        elif phase < 0.25:
            return "early rising phase"
        elif phase < 0.375:
            return "accelerating rise"
        elif phase < 0.5:
            return "approaching peak"
        elif phase < 0.625:
            return "just past peak"
        elif phase < 0.75:
            return "declining phase"
        elif phase < 0.875:
            return "accelerating decline"
        else:
            return "approaching cycle bottom"

    def analyze_volatility_clustering(self, df, return_col=None, window=50):
        """
        Analyze volatility clustering and persistence.

        Parameters:
        -----------
        df : pd.DataFrame
            DataFrame with returns
        return_col : str, optional
            Column name for returns (if None, uses 'log_return' or 'pct_return')
        window : int
            Window size for rolling analysis

        Returns:
        --------
        pd.DataFrame, dict
            DataFrame with clustering metrics and summary statistics
        """
        if df is None or df.empty:
            logger.warning("Empty DataFrame provided to analyze_volatility_clustering")
            return df, {}

        # Make a copy to avoid modifying the original
        df_cluster = df.copy()

        # Determine return column
        if return_col is None:
            return_col = 'log_return' if self.use_log_returns else 'pct_return'

        # Check for required column
        if return_col not in df_cluster.columns:
            logger.warning(f"Return column '{return_col}' not found in DataFrame")
            return df_cluster, {}

        try:
            # Calculate squared returns (proxy for volatility)
            df_cluster['return_squared'] = df_cluster[return_col] ** 2

            # Calculate autocorrelation of squared returns
            autocorr_lag1 = df_cluster['return_squared'].autocorr(lag=1)

            # Calculate rolling autocorrelation
            df_cluster['squared_return_autocorr'] = df_cluster['return_squared'].rolling(window=window).apply(
                lambda x: x.autocorr(lag=1) if len(x) > 1 else np.nan
            )

            # Calculate volatility persistence using ARCH LM test
            from statsmodels.stats.diagnostic import het_arch

            # Need at least 100 observations for a reliable test
            if len(df_cluster) >= 100:
                returns = df_cluster[return_col].dropna().values
                arch_lm, p_value, _, _ = het_arch(returns, nlags=5)

                # High arch_lm statistic and low p-value indicate ARCH effects (volatility clustering)
                has_arch_effect = p_value < 0.05
            else:
                arch_lm = np.nan
                p_value = np.nan
                has_arch_effect = None

            # Calculate runs test for volatility persistence
            # A run is a sequence of consecutive returns with the same sign
            df_cluster['return_sign'] = np.sign(df_cluster[return_col])

            # Count runs
            sign_changes = (df_cluster['return_sign'] != df_cluster['return_sign'].shift(1)).fillna(0).sum()
            run_count = sign_changes + 1  # Number of runs

            # Expected number of runs under randomness
            n = len(df_cluster.dropna())
            n_pos = (df_cluster['return_sign'] > 0).sum()
            n_neg = (df_cluster['return_sign'] < 0).sum()

            expected_runs = (2 * n_pos * n_neg) / n + 1 if n > 0 else np.nan
            runs_test_statistic = (run_count - expected_runs) / np.sqrt(
                (2 * n_pos * n_neg * (2 * n_pos * n_neg - n)) / (n ** 2 * (n - 1))) if n > 1 else np.nan

            # Negative runs_test_statistic indicates fewer runs than expected (persistence)
            has_persistence = runs_test_statistic < -1.96 if not np.isnan(runs_test_statistic) else None

            # Calculate GARCH volatility persistence if arch package is available
            garch_persistence = None

            try:
                from arch import arch_model

                # Remove NaN values
                returns = df_cluster[return_col].dropna().values

                if len(returns) >= 100:
                    # Fit GARCH(1,1) model
                    model = arch_model(returns, vol='GARCH', p=1, q=1, mean='Zero', rescale=False)
                    model_fit = model.fit(disp='off')

                    # Get GARCH parameters
                    alpha = model_fit.params['alpha[1]']
                    beta = model_fit.params['beta[1]']

                    # Calculate persistence (alpha + beta)
                    garch_persistence = alpha + beta
            except ImportError:
                pass
            except Exception as garch_error:
                logger.warning(f"Error calculating GARCH persistence: {garch_error}")

            # Create metrics dictionary
            clustering_metrics = {
                'autocorrelation_lag1': float(autocorr_lag1),
                'arch_lm_statistic': float(arch_lm) if not np.isnan(arch_lm) else None,
                'arch_lm_p_value': float(p_value) if not np.isnan(p_value) else None,
                'has_arch_effect': has_arch_effect,
                'runs_test_statistic': float(runs_test_statistic) if not np.isnan(runs_test_statistic) else None,
                'has_persistence': has_persistence,
                'garch_persistence': float(garch_persistence) if garch_persistence is not None else None
            }

            logger.info(
                f"Analyzed volatility clustering: autocorr={autocorr_lag1:.3f}, has_persistence={has_persistence}")
            return df_cluster, clustering_metrics

        except Exception as e:
            logger.error(f"Error analyzing volatility clustering: {e}")
            return df_cluster, {}

    def detect_structural_breaks(self, df, vol_col=None, return_col=None):
        """
        Detect structural breaks in volatility.

        Parameters:
        -----------
        df : pd.DataFrame
            DataFrame with volatility data
        vol_col : str, optional
            Column name for volatility (if None, uses 'volatility_20_annualized')
        return_col : str, optional
            Column name for returns (if None, uses 'log_return' or 'pct_return')

        Returns:
        --------
        pd.DataFrame, dict
            DataFrame with break points and break information
        """
        if df is None or df.empty:
            logger.warning("Empty DataFrame provided to detect_structural_breaks")
            return df, {}

        # Make a copy to avoid modifying the original
        df_breaks = df.copy()

        # Determine volatility column
        if vol_col is None:
            # Try to find a suitable volatility column
            for w in self.window_sizes:
                col = f'volatility_{w}_annualized'
                if col in df_breaks.columns:
                    vol_col = col
                    break

            if vol_col is None:
                if return_col is None:
                    return_col = 'log_return' if self.use_log_returns else 'pct_return'

                if return_col in df_breaks.columns:
                    # Use squared returns as proxy for volatility
                    df_breaks['return_squared'] = df_breaks[return_col] ** 2
                    vol_col = 'return_squared'
                else:
                    logger.warning("No volatility or return columns found in DataFrame")
                    return df_breaks, {}

        # Check for required column
        if vol_col not in df_breaks.columns:
            logger.warning(f"Volatility column '{vol_col}' not found in DataFrame")
            return df_breaks, {}

        try:
            # Import structural break detection libraries
            import ruptures as rpt

            # Get volatility data
            volatility = df_breaks[vol_col].dropna().values

            if len(volatility) < 100:
                logger.warning(f"Not enough data points ({len(volatility)}) for structural break detection")
                return df_breaks, {}

            # Apply PELT algorithm for break detection
            detector = rpt.Pelt(model="rbf", min_size=20).fit(volatility.reshape(-1, 1))
            break_points = detector.predict(pen=10)

            # Remove the last break point (end of the series)
            if break_points and break_points[-1] == len(volatility):
                break_points = break_points[:-1]

            # Add break points to the DataFrame
            df_breaks['structural_break'] = False

            # Map break points to original index
            vol_index = df_breaks[vol_col].dropna().index

            break_indices = [vol_index[bp] for bp in break_points if bp < len(vol_index)]

            if break_indices:
                df_breaks.loc[break_indices, 'structural_break'] = True

            # Calculate segment statistics
            segments = []

            if break_points:
                # Add start of series
                all_breaks = [0] + break_points

                for i in range(len(all_breaks) - 1):
                    start_idx = all_breaks[i]
                    end_idx = all_breaks[i + 1]

                    segment_vol = volatility[start_idx:end_idx]

                    segments.append({
                        'start_idx': int(start_idx),
                        'end_idx': int(end_idx),
                        'start_date': vol_index[start_idx] if start_idx < len(vol_index) else None,
                        'end_date': vol_index[end_idx - 1] if end_idx <= len(vol_index) else None,
                        'length': int(end_idx - start_idx),
                        'mean_volatility': float(np.mean(segment_vol)),
                        'std_volatility': float(np.std(segment_vol))
                    })

                # Add last segment
                last_break = break_points[-1]

                if last_break < len(volatility):
                    segment_vol = volatility[last_break:]

                    segments.append({
                        'start_idx': int(last_break),
                        'end_idx': int(len(volatility)),
                        'start_date': vol_index[last_break] if last_break < len(vol_index) else None,
                        'end_date': vol_index[-1] if len(vol_index) > 0 else None,
                        'length': int(len(volatility) - last_break),
                        'mean_volatility': float(np.mean(segment_vol)),
                        'std_volatility': float(np.std(segment_vol))
                    })

            # Create metrics dictionary
            break_metrics = {
                'break_points': break_points,
                'break_dates': [vol_index[bp].strftime('%Y-%m-%d') if bp < len(vol_index) else None for bp in
                                break_points],
                'segment_count': len(segments),
                'segments': segments
            }

            logger.info(f"Detected {len(break_points)} structural breaks in volatility")
            return df_breaks, break_metrics

        except ImportError:
            logger.warning("ruptures package not available for structural break detection")
            return df_breaks, {}
        except Exception as e:
            logger.error(f"Error detecting structural breaks: {e}")
            return df_breaks, {}

    def analyze_volatility_seasonality(self, df, vol_col=None):
        """
        Analyze seasonality patterns in volatility.

        Parameters:
        -----------
        df : pd.DataFrame
            DataFrame with volatility data
        vol_col : str, optional
            Column name for volatility (if None, uses 'volatility_20_annualized')

        Returns:
        --------
        pd.DataFrame, dict
            DataFrame with seasonality indicators and seasonality metrics
        """
        if df is None or df.empty:
            logger.warning("Empty DataFrame provided to analyze_volatility_seasonality")
            return df, {}

        # Make a copy to avoid modifying the original
        df_seasonality = df.copy()

        # Determine volatility column
        if vol_col is None:
            # Try to find a suitable volatility column
            for w in self.window_sizes:
                col = f'volatility_{w}_annualized'
                if col in df_seasonality.columns:
                    vol_col = col
                    break

            if vol_col is None:
                logger.warning("No volatility columns found in DataFrame")
                return df_seasonality, {}

        # Check for required column
        if vol_col not in df_seasonality.columns:
            logger.warning(f"Volatility column '{vol_col}' not found in DataFrame")
            return df_seasonality, {}

        # Check if index is datetime
        if not isinstance(df_seasonality.index, pd.DatetimeIndex):
            logger.warning("DataFrame index is not datetime, cannot analyze seasonality")
            return df_seasonality, {}

        try:
            # Extract date components
            df_seasonality['year'] = df_seasonality.index.year
            df_seasonality['month'] = df_seasonality.index.month
            df_seasonality['day_of_week'] = df_seasonality.index.dayofweek
            df_seasonality['day_of_month'] = df_seasonality.index.day
            df_seasonality['week_of_year'] = df_seasonality.index.isocalendar().week

            # Calculate monthly volatility
            monthly_vol = df_seasonality.groupby('month')[vol_col].mean()

            # Calculate day-of-week volatility
            dow_vol = df_seasonality.groupby('day_of_week')[vol_col].mean()

            # Calculate month volatility ratio (relative to annual average)
            annual_avg = df_seasonality[vol_col].mean()
            monthly_vol_ratio = monthly_vol / annual_avg

            # Calculate day-of-week volatility ratio
            dow_vol_ratio = dow_vol / annual_avg

            # Extract monthly seasonality
            month_names = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
            monthly_pattern = []

            for month in range(1, 13):
                if month in monthly_vol_ratio.index:
                    monthly_pattern.append({
                        'month': month,
                        'month_name': month_names[month - 1],
                        'volatility_ratio': float(monthly_vol_ratio[month]),
                        'volatility': float(monthly_vol[month])
                    })

            # Extract day-of-week seasonality
            dow_names = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
            dow_pattern = []

            for dow in range(7):
                if dow in dow_vol_ratio.index:
                    dow_pattern.append({
                        'day_of_week': dow,
                        'day_name': dow_names[dow],
                        'volatility_ratio': float(dow_vol_ratio[dow]),
                        'volatility': float(dow_vol[dow])
                    })

            # Identify months with highest/lowest volatility
            high_vol_months = [m['month'] for m in
                               sorted(monthly_pattern, key=lambda x: x['volatility_ratio'], reverse=True)[:3]]
            low_vol_months = [m['month'] for m in sorted(monthly_pattern, key=lambda x: x['volatility_ratio'])[:3]]

            # Identify days with highest/lowest volatility
            high_vol_days = [d['day_of_week'] for d in
                             sorted(dow_pattern, key=lambda x: x['volatility_ratio'], reverse=True)[:2]]
            low_vol_days = [d['day_of_week'] for d in sorted(dow_pattern, key=lambda x: x['volatility_ratio'])[:2]]

            # Label high/low volatility periods
            df_seasonality['is_high_vol_month'] = df_seasonality['month'].isin(high_vol_months)
            df_seasonality['is_low_vol_month'] = df_seasonality['month'].isin(low_vol_months)
            df_seasonality['is_high_vol_day'] = df_seasonality['day_of_week'].isin(high_vol_days)
            df_seasonality['is_low_vol_day'] = df_seasonality['day_of_week'].isin(low_vol_days)

            # Create seasonality metrics
            seasonality_metrics = {
                'monthly_pattern': monthly_pattern,
                'day_of_week_pattern': dow_pattern,
                'high_volatility_months': high_vol_months,
                'low_volatility_months': low_vol_months,
                'high_volatility_days': high_vol_days,
                'low_volatility_days': low_vol_days,
                'month_max_min_ratio': float(
                    monthly_vol.max() / monthly_vol.min()) if monthly_vol.min() > 0 else None,
                'dow_max_min_ratio': float(dow_vol.max() / dow_vol.min()) if dow_vol.min() > 0 else None
            }

            logger.info(f"Analyzed volatility seasonality patterns")
            return df_seasonality, seasonality_metrics

        except Exception as e:
            logger.error(f"Error analyzing volatility seasonality: {e}")
            return df_seasonality, {}

    def run_analysis(self, df, price_col='close', volume_col='volume'):
        """
        Run a comprehensive macro-volatility analysis.

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

        logger.info("Running comprehensive macro-volatility analysis")

        try:
            # Step 1: Clean data
            df_clean = self.normalizer.clean_ohlcv_data(df)

            # Step 2: Calculate returns
            df_returns = self.calculate_returns(df_clean, price_col=price_col)

            # Step 3: Calculate historical volatility
            df_vol = self.calculate_historical_volatility(df_returns)

            # Step 4: Calculate GARCH volatility
            df_vol = self.calculate_garch_volatility(df_vol)

            # Step 5: Detect volatility cycles
            df_vol, cycle_metrics = self.detect_volatility_cycles(df_vol)

            # Step 6: Analyze volatility clustering
            df_vol, clustering_metrics = self.analyze_volatility_clustering(df_vol)

            # Step 7: Detect structural breaks
            df_vol, break_metrics = self.detect_structural_breaks(df_vol)

            # Step 8: Analyze seasonality
            df_vol, seasonality_metrics = self.analyze_volatility_seasonality(df_vol)

            # Compute overall metrics
            metrics = {
                'volatility_cycles': cycle_metrics,
                'volatility_clustering': clustering_metrics,
                'structural_breaks': break_metrics,
                'seasonality': seasonality_metrics,
                'current_volatility': {
                    'historical_vol_20d': float(df_vol['volatility_20_annualized'].iloc[
                                                    -1]) if 'volatility_20_annualized' in df_vol.columns and len(
                        df_vol) > 0 else None,
                    'garch_vol': float(df_vol['garch_volatility_annualized'].iloc[
                                           -1]) if 'garch_volatility_annualized' in df_vol.columns and len(
                        df_vol) > 0 else None,
                    'garch_forecast_30d': float(
                        df_vol['garch_forecast_30d'].iloc[-1]) if 'garch_forecast_30d' in df_vol.columns and len(
                        df_vol) > 0 and not pd.isna(df_vol['garch_forecast_30d'].iloc[-1]) else None
                }
            }

            logger.info(f"Completed macro-volatility analysis")
            return df_vol, metrics

        except Exception as e:
            logger.error(f"Error running macro-volatility analysis: {e}")
            return df, {}

# Factory function to get a macro-volatility analyzer
def get_macro_volatility_analyzer(window_sizes=None, use_log_returns=True):
    """
    Get a configured macro-volatility analyzer.

    Parameters:
    -----------
    window_sizes : list, optional
        List of window sizes for rolling volatility calculation
    use_log_returns : bool
        If True, use log returns for volatility calculation

    Returns:
    --------
    MacroVolatilityAnalyzer
        Configured analyzer instance
    """
    return MacroVolatilityAnalyzer(window_sizes=window_sizes, use_log_returns=use_log_returns)