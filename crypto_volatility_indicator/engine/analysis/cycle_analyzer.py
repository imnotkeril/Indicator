"""
Module for cycle analysis of time series.
Identifies and analyzes cycles in price and volatility data.
"""
import os
import sys
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import logging
from scipy import signal, stats
from scipy.fftpack import fft, fftfreq
import matplotlib.pyplot as plt
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, project_root)
from crypto_volatility_indicator.utils.logger import get_logger

# Set up logger
logger = get_logger(__name__)


class CycleAnalyzer:
    """
    Analyzer for cycles in time series data.

    This class identifies and analyzes cycles in price, volatility,
    and other time series data using various methods including FFT,
    autocorrelation, and peak detection.
    """

    def __init__(self, max_cycle_length=None, min_cycle_length=5):
        """
        Initialize the cycle analyzer.

        Parameters:
        -----------
        max_cycle_length : int, optional
            Maximum cycle length to detect (in days)
            If None, will be set to 1/3 of the data length
        min_cycle_length : int
            Minimum cycle length to detect (in days)
        """
        self.max_cycle_length = max_cycle_length
        self.min_cycle_length = min_cycle_length

        logger.info(f"CycleAnalyzer initialized with min_cycle_length={min_cycle_length}")

    def analyze_fft(self, time_series, sampling_freq=1.0):
        """
        Analyze cycles using Fast Fourier Transform.

        Parameters:
        -----------
        time_series : array-like
            Time series data
        sampling_freq : float
            Sampling frequency in data points per day

        Returns:
        --------
        dict
            FFT analysis results
        """
        # Convert to numpy array if needed
        if isinstance(time_series, pd.Series):
            values = time_series.values
            dates = time_series.index
            is_series = True
        else:
            values = np.array(time_series)
            dates = None
            is_series = False

        # Remove NaN values
        mask = ~np.isnan(values)
        values = values[mask]
        if is_series:
            dates = dates[mask]

        if len(values) < 10:
            logger.warning("Time series too short for FFT analysis")
            return {'error': 'Time series too short'}

        try:
            # Calculate FFT
            fft_values = fft(values - np.mean(values))

            # Get the absolute values of the complex FFT results
            magnitudes = np.abs(fft_values)

            # Calculate frequencies
            freq = fftfreq(len(values), d=1.0 / sampling_freq)

            # Consider only positive frequencies
            pos_mask = freq > 0
            freqs = freq[pos_mask]
            mags = magnitudes[pos_mask]

            # Convert frequencies to periods (in days)
            periods = 1.0 / freqs

            # Set defaults for max_cycle_length if not provided
            if self.max_cycle_length is None:
                max_cycle_length = len(values) // 3
            else:
                max_cycle_length = self.max_cycle_length

            # Filter out periods outside the min-max range
            period_mask = (periods >= self.min_cycle_length) & (periods <= max_cycle_length)
            filtered_periods = periods[period_mask]
            filtered_mags = mags[period_mask]

            if len(filtered_periods) == 0:
                logger.warning("No cycles found within the specified length range")
                return {'error': 'No cycles found within the specified length range'}

            # Find peaks in the magnitude spectrum
            peaks, _ = signal.find_peaks(filtered_mags, height=np.mean(filtered_mags))

            if len(peaks) == 0:
                logger.warning("No significant peaks found in FFT analysis")
                return {'error': 'No significant peaks found'}

            # Sort peaks by magnitude
            peak_indices = sorted([(i, filtered_mags[i]) for i in peaks], key=lambda x: x[1], reverse=True)

            # Extract top cycles
            cycles = []

            for idx, magnitude in peak_indices[:5]:  # Get top 5 cycles
                period = filtered_periods[idx]

                # Calculate cycle strength (normalized magnitude)
                strength = magnitude / np.max(mags)

                cycles.append({
                    'period': float(period),
                    'frequency': float(freqs[period_mask][idx]),
                    'magnitude': float(magnitude),
                    'strength': float(strength),
                    'description': f"{period:.1f}-day cycle"
                })

            # Calculate current phase for dominant cycle
            if len(cycles) > 0:
                dominant_cycle = cycles[0]
                period = dominant_cycle['period']

                # Get the complex value at the dominant frequency
                freq_idx = np.where(freq == dominant_cycle['frequency'])[0][0]
                phase_angle = np.angle(fft_values[freq_idx])

                # Convert to phase (0 to 1, where 0/1 = cycle start)
                phase = (phase_angle / (2 * np.pi)) % 1

                # Adjust phase based on current position in time series
                if is_series:
                    # Calculate days since start
                    if isinstance(dates[0], (datetime, pd.Timestamp)):
                        days_elapsed = (dates[-1] - dates[0]).total_seconds() / (24 * 3600)
                    else:
                        days_elapsed = len(values) / sampling_freq

                    # Adjust phase by days elapsed
                    adjusted_phase = (phase + (days_elapsed % period) / period) % 1

                    dominant_cycle['current_phase'] = float(adjusted_phase)
                    dominant_cycle['phase_description'] = self._describe_cycle_phase(adjusted_phase)

                    # Calculate next cycle peak and trough
                    cycle_start_date = dates[-1] - timedelta(days=days_elapsed % period)
                    next_peak_date = cycle_start_date + timedelta(days=period * 0.25)
                    next_trough_date = cycle_start_date + timedelta(days=period * 0.75)

                    # Adjust if already passed
                    if next_peak_date <= dates[-1]:
                        next_peak_date += timedelta(days=period)
                    if next_trough_date <= dates[-1]:
                        next_trough_date += timedelta(days=period)

                    dominant_cycle['next_peak'] = next_peak_date.strftime('%Y-%m-%d')
                    dominant_cycle['next_trough'] = next_trough_date.strftime('%Y-%m-%d')

            results = {
                'method': 'fft',
                'cycles': cycles,
                'dominant_cycle': cycles[0] if cycles else None,
                'data': {
                    'periods': filtered_periods.tolist(),
                    'magnitudes': filtered_mags.tolist()
                }
            }

            logger.info(f"FFT analysis found {len(cycles)} significant cycles")
            return results

        except Exception as e:
            logger.error(f"Error in FFT analysis: {e}")
            return {'error': str(e)}

    def analyze_autocorrelation(self, time_series, max_lag=None):
        """
        Analyze cycles using autocorrelation.

        Parameters:
        -----------
        time_series : array-like
            Time series data
        max_lag : int, optional
            Maximum lag to consider

        Returns:
        --------
        dict
            Autocorrelation analysis results
        """
        # Convert to numpy array if needed
        if isinstance(time_series, pd.Series):
            values = time_series.values
            dates = time_series.index
            is_series = True
        else:
            values = np.array(time_series)
            dates = None
            is_series = False

        # Remove NaN values
        mask = ~np.isnan(values)
        values = values[mask]
        if is_series:
            dates = dates[mask]

        if len(values) < 10:
            logger.warning("Time series too short for autocorrelation analysis")
            return {'error': 'Time series too short'}

        # Set max_lag if not provided
        if max_lag is None:
            if self.max_cycle_length is None:
                max_lag = len(values) // 3
            else:
                max_lag = self.max_cycle_length

        try:
            # Calculate autocorrelation
            acf_values = self._calculate_acf(values, max_lag)

            # Find peaks in autocorrelation
            peaks, _ = signal.find_peaks(acf_values[self.min_cycle_length:], height=0.1)

            # Account for minimum lag offset
            peaks = peaks + self.min_cycle_length

            if len(peaks) == 0:
                logger.warning("No significant peaks found in autocorrelation analysis")
                return {'error': 'No significant peaks found'}

            # Sort peaks by correlation value
            peak_indices = sorted([(i, acf_values[i]) for i in peaks], key=lambda x: x[1], reverse=True)

            # Extract top cycles
            cycles = []

            for idx, correlation in peak_indices[:5]:  # Get top 5 cycles
                period = idx

                # Calculate cycle strength (normalized correlation)
                strength = correlation

                cycles.append({
                    'period': float(period),
                    'correlation': float(correlation),
                    'strength': float(strength),
                    'description': f"{period}-day cycle"
                })

            # Calculate current phase for dominant cycle
            if len(cycles) > 0 and is_series:
                dominant_cycle = cycles[0]
                period = dominant_cycle['period']

                # Create a synthetic cycle with the detected period
                synthetic_cycle = np.sin(np.arange(period) * 2 * np.pi / period)

                # Compare with recent values to determine phase
                if len(values) >= period:
                    recent_values = values[-period:]
                    normalized_values = (recent_values - np.mean(recent_values)) / (np.std(recent_values) + 1e-10)

                    # Calculate correlation with shifted synthetic cycle
                    correlations = []
                    for shift in range(period):
                        shifted_cycle = np.roll(synthetic_cycle, shift)
                        correlation = np.corrcoef(normalized_values, shifted_cycle)[0, 1]
                        correlations.append(correlation)

                    # Best shift is the one with highest correlation
                    best_shift = np.argmax(correlations)

                    # Convert to phase (0 to 1, where 0 = cycle start)
                    phase = (period - best_shift) % period / period

                    dominant_cycle['current_phase'] = float(phase)
                    dominant_cycle['phase_description'] = self._describe_cycle_phase(phase)

                    # Calculate next cycle peak and trough
                    if isinstance(dates[0], (datetime, pd.Timestamp)):
                        current_date = dates[-1]
                        cycle_position = period * phase
                        days_to_peak = (period * 0.25 - cycle_position) % period
                        days_to_trough = (period * 0.75 - cycle_position) % period

                        next_peak_date = current_date + timedelta(days=days_to_peak)
                        next_trough_date = current_date + timedelta(days=days_to_trough)

                        dominant_cycle['next_peak'] = next_peak_date.strftime('%Y-%m-%d')
                        dominant_cycle['next_trough'] = next_trough_date.strftime('%Y-%m-%d')

            results = {
                'method': 'autocorrelation',
                'cycles': cycles,
                'dominant_cycle': cycles[0] if cycles else None,
                'data': {
                    'lags': list(range(len(acf_values))),
                    'acf_values': acf_values.tolist()
                }
            }

            logger.info(f"Autocorrelation analysis found {len(cycles)} significant cycles")
            return results

        except Exception as e:
            logger.error(f"Error in autocorrelation analysis: {e}")
            return {'error': str(e)}

    def _calculate_acf(self, time_series, max_lag):
        """
        Calculate autocorrelation function.

        Parameters:
        -----------
        time_series : array-like
            Time series data
        max_lag : int
            Maximum lag

        Returns:
        --------
        array-like
            Autocorrelation values
        """
        # Ensure the time series is normalized
        ts = time_series - np.mean(time_series)
        ts = ts / (np.std(ts) + 1e-10)

        # Calculate autocorrelation using numpy's correlate
        acf = np.correlate(ts, ts, mode='full')

        # Extract the positive lags
        acf = acf[len(ts) - 1:]

        # Normalize
        acf = acf / acf[0]

        # Trim to max_lag
        acf = acf[:min(max_lag + 1, len(acf))]

        return acf

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

    def analyze_seasonal(self, time_series, frequency=None):
        """
        Analyze seasonal patterns in time series.

        Parameters:
        -----------
        time_series : pd.Series
            Time series data with datetime index
        frequency : str, optional
            Frequency to analyze ('daily', 'weekly', 'monthly', 'quarterly', 'annual')
            If None, tries to determine from data

        Returns:
        --------
        dict
            Seasonal analysis results
        """
        if not isinstance(time_series, pd.Series):
            logger.warning("Seasonal analysis requires a pandas Series with DatetimeIndex")
            return {'error': 'Input must be a pandas Series with DatetimeIndex'}

        if not isinstance(time_series.index, pd.DatetimeIndex):
            logger.warning("Seasonal analysis requires a datetime index")
            return {'error': 'Series must have a DatetimeIndex'}

        try:
            # Determine frequency if not provided
            if frequency is None:
                # Check index frequency
                if time_series.index.freq:
                    if time_series.index.freq.name in ['D', 'B']:
                        frequency = 'daily'
                    elif time_series.index.freq.name in ['W', 'W-SUN', 'W-MON']:
                        frequency = 'weekly'
                    elif time_series.index.freq.name in ['M', 'MS', 'BM', 'BMS']:
                        frequency = 'monthly'
                    elif time_series.index.freq.name in ['Q', 'QS', 'BQ', 'BQS']:
                        frequency = 'quarterly'
                    elif time_series.index.freq.name in ['A', 'AS', 'BA', 'BAS']:
                        frequency = 'annual'
                    else:
                        # Default to daily for unknown frequencies
                        frequency = 'daily'
                else:
                    # Guess based on time difference between observations
                    time_diff = (time_series.index[-1] - time_series.index[0]).total_seconds()
                    n_obs = len(time_series)
                    avg_seconds_per_obs = time_diff / (n_obs - 1)

                    if avg_seconds_per_obs < 24 * 3600 * 2:  # Less than 2 days
                        frequency = 'daily'
                    elif avg_seconds_per_obs < 7 * 24 * 3600 * 2:  # Less than 2 weeks
                        frequency = 'weekly'
                    elif avg_seconds_per_obs < 31 * 24 * 3600 * 2:  # Less than 2 months
                        frequency = 'monthly'
                    elif avg_seconds_per_obs < 91 * 24 * 3600 * 2:  # Less than 2 quarters
                        frequency = 'quarterly'
                    else:
                        frequency = 'annual'

            # Analyze based on frequency
            if frequency == 'daily':
                # Day of week patterns
                time_series = time_series.copy()
                time_series.index = pd.DatetimeIndex(time_series.index)
                day_of_week = time_series.groupby(time_series.index.dayofweek).mean()

                # Map day numbers to names
                day_names = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
                day_of_week.index = [day_names[i] for i in day_of_week.index]

                # Find days with highest and lowest values
                sorted_days = day_of_week.sort_values(ascending=False)
                best_day = sorted_days.index[0]
                worst_day = sorted_days.index[-1]

                # Calculate day ratios
                daily_avg = day_of_week.mean()
                day_ratios = day_of_week / daily_avg

                # Format for output
                day_of_week_data = [
                    {'day': day, 'value': float(day_of_week[day]), 'ratio': float(day_ratios[day])}
                    for day in day_of_week.index
                ]

                results = {
                    'frequency': frequency,
                    'day_of_week': day_of_week_data,
                    'best_day': {
                        'day': best_day,
                        'value': float(day_of_week[best_day]),
                        'ratio': float(day_ratios[best_day])
                    },
                    'worst_day': {
                        'day': worst_day,
                        'value': float(day_of_week[worst_day]),
                        'ratio': float(day_ratios[worst_day])
                    }
                }

            elif frequency == 'weekly':
                # Week of month patterns
                time_series = time_series.copy()
                time_series.index = pd.DatetimeIndex(time_series.index)

                # Assign week of month (1-5)
                time_series = time_series.reset_index()
                time_series['week_of_month'] = time_series['index'].apply(
                    lambda x: (x.day - 1) // 7 + 1
                )

                week_of_month = time_series.groupby('week_of_month').mean().iloc[:, 0]

                # Find weeks with highest and lowest values
                sorted_weeks = week_of_month.sort_values(ascending=False)
                best_week = sorted_weeks.index[0]
                worst_week = sorted_weeks.index[-1]

                # Calculate week ratios
                weekly_avg = week_of_month.mean()
                week_ratios = week_of_month / weekly_avg

                # Format for output
                week_of_month_data = [
                    {'week': int(week), 'value': float(week_of_month[week]), 'ratio': float(week_ratios[week])}
                    for week in week_of_month.index
                ]

                results = {
                    'frequency': frequency,
                    'week_of_month': week_of_month_data,
                    'best_week': {
                        'week': int(best_week),
                        'value': float(week_of_month[best_week]),
                        'ratio': float(week_ratios[best_week])
                    },
                    'worst_week': {
                        'week': int(worst_week),
                        'value': float(week_of_month[worst_week]),
                        'ratio': float(week_ratios[worst_week])
                    }
                }

            elif frequency == 'monthly':
                # Month of year patterns
                time_series = time_series.copy()
                time_series.index = pd.DatetimeIndex(time_series.index)
                month_of_year = time_series.groupby(time_series.index.month).mean()

                # Map month numbers to names
                month_names = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
                month_of_year.index = [month_names[i - 1] for i in month_of_year.index]

                # Find months with highest and lowest values
                sorted_months = month_of_year.sort_values(ascending=False)
                best_month = sorted_months.index[0]
                worst_month = sorted_months.index[-1]

                # Calculate month ratios
                monthly_avg = month_of_year.mean()
                month_ratios = month_of_year / monthly_avg

                # Format for output
                month_of_year_data = [
                    {'month': month, 'value': float(month_of_year[month]), 'ratio': float(month_ratios[month])}
                    for month in month_of_year.index
                ]

                results = {
                    'frequency': frequency,
                    'month_of_year': month_of_year_data,
                    'best_month': {
                        'month': best_month,
                        'value': float(month_of_year[best_month]),
                        'ratio': float(month_ratios[best_month])
                    },
                    'worst_month': {
                        'month': worst_month,
                        'value': float(month_of_year[worst_month]),
                        'ratio': float(month_ratios[worst_month])
                    }
                }

            elif frequency == 'quarterly':
                # Quarter patterns
                time_series = time_series.copy()
                time_series.index = pd.DatetimeIndex(time_series.index)
                quarter_of_year = time_series.groupby(time_series.index.quarter).mean()

                # Map quarter numbers to names
                quarter_names = ['Q1', 'Q2', 'Q3', 'Q4']
                quarter_of_year.index = [quarter_names[i - 1] for i in quarter_of_year.index]

                # Find quarters with highest and lowest values
                sorted_quarters = quarter_of_year.sort_values(ascending=False)
                best_quarter = sorted_quarters.index[0]
                worst_quarter = sorted_quarters.index[-1]

                # Calculate quarter ratios
                quarterly_avg = quarter_of_year.mean()
                quarter_ratios = quarter_of_year / quarterly_avg

                # Format for output
                quarter_of_year_data = [
                    {'quarter': quarter, 'value': float(quarter_of_year[quarter]),
                     'ratio': float(quarter_ratios[quarter])}
                    for quarter in quarter_of_year.index
                ]

                results = {
                    'frequency': frequency,
                    'quarter_of_year': quarter_of_year_data,
                    'best_quarter': {
                        'quarter': best_quarter,
                        'value': float(quarter_of_year[best_quarter]),
                        'ratio': float(quarter_ratios[best_quarter])
                    },
                    'worst_quarter': {
                        'quarter': worst_quarter,
                        'value': float(quarter_of_year[worst_quarter]),
                        'ratio': float(quarter_ratios[worst_quarter])
                    }
                }

            elif frequency == 'annual':
                # Year patterns
                time_series = time_series.copy()
                time_series.index = pd.DatetimeIndex(time_series.index)
                yearly_avg = time_series.groupby(time_series.index.year).mean()

                # Calculate year-over-year percentage change
                yearly_pct_change = yearly_avg.pct_change() * 100

                # Format for output
                yearly_data = [
                    {
                        'year': int(year),
                        'value': float(yearly_avg.loc[year]),
                        'pct_change': float(yearly_pct_change.loc[year]) if year in yearly_pct_change.index else None
                    }
                    for year in yearly_avg.index
                ]

                results = {
                    'frequency': frequency,
                    'yearly': yearly_data
                }

            else:
                logger.warning(f"Unknown frequency: {frequency}")
                return {'error': f'Unknown frequency: {frequency}'}

            logger.info(f"Seasonal analysis completed for {frequency} frequency")
            return results

        except Exception as e:
            logger.error(f"Error in seasonal analysis: {e}")
            return {'error': str(e)}

    def detect_dominant_cycles(self, time_series, price_col='close', date_col=None, sampling_freq=1.0):
        """
        Detect dominant cycles in time series using multiple methods.

        Parameters:
        -----------
        time_series : pd.DataFrame or pd.Series
            Time series data
        price_col : str
            Column name for price data (if DataFrame)
        date_col : str, optional
            Column name for date data (if DataFrame)
        sampling_freq : float
            Sampling frequency in data points per day

        Returns:
        --------
        dict
            Combined cycle analysis results
        """
        # Handle different input types
        if isinstance(time_series, pd.DataFrame):
            if price_col not in time_series.columns:
                logger.warning(f"Price column '{price_col}' not found in DataFrame")
                return {'error': f"Price column '{price_col}' not found in DataFrame"}

            # Extract price series
            price_series = time_series[price_col]

            # Extract dates if available
            if date_col is not None and date_col in time_series.columns:
                price_series.index = time_series[date_col]

        elif isinstance(time_series, pd.Series):
            price_series = time_series
        else:
            logger.warning("Input must be a pandas DataFrame or Series")
            return {'error': "Input must be a pandas DataFrame or Series"}

        # Run different analyses
        fft_results = self.analyze_fft(price_series, sampling_freq)
        ac_results = self.analyze_autocorrelation(price_series)

        # Run seasonal analysis if index is datetime
        seasonal_results = None

        if isinstance(price_series.index, pd.DatetimeIndex):
            seasonal_results = self.analyze_seasonal(price_series)

        # Combine results
        cycles = []

        # Add FFT cycles
        if 'cycles' in fft_results:
            for cycle in fft_results['cycles']:
                cycle['method'] = 'fft'
                cycles.append(cycle)

        # Add autocorrelation cycles
        if 'cycles' in ac_results:
            for cycle in ac_results['cycles']:
                cycle['method'] = 'autocorrelation'
                cycles.append(cycle)

        # Find the strongest cycle overall
        if cycles:
            # Sort by strength
            sorted_cycles = sorted(cycles, key=lambda x: x['strength'], reverse=True)
            dominant_cycle = sorted_cycles[0]
        else:
            dominant_cycle = None

        results = {
            'methods': {
                'fft': fft_results,
                'autocorrelation': ac_results,
                'seasonal': seasonal_results
            },
            'all_cycles': cycles,
            'dominant_cycle': dominant_cycle,
            'time_based_cycles': seasonal_results
        }

        logger.info(f"Detected {len(cycles)} cycles across all methods")
        return results

    def visualize_cycles(self, time_series, analysis_results=None, price_col='close', show_plot=True):
        """
        Visualize cycles in time series data.

        Parameters:
        -----------
        time_series : pd.DataFrame or pd.Series
            Time series data
        analysis_results : dict, optional
            Results from cycle analysis
            If None, will run detect_dominant_cycles
        price_col : str
            Column name for price data (if DataFrame)
        show_plot : bool
            Whether to display the plot or return the figure

        Returns:
        --------
        matplotlib.figure.Figure or None
            Figure object if show_plot is False, None otherwise
        """
        # Handle different input types
        if isinstance(time_series, pd.DataFrame):
            if price_col not in time_series.columns:
                logger.warning(f"Price column '{price_col}' not found in DataFrame")
                return None

            # Extract price series
            price_series = time_series[price_col]
        elif isinstance(time_series, pd.Series):
            price_series = time_series
        else:
            logger.warning("Input must be a pandas DataFrame or Series")
            return None

        # Run analysis if not provided
        if analysis_results is None:
            analysis_results = self.detect_dominant_cycles(time_series, price_col)

        # Check if analysis was successful
        if 'error' in analysis_results:
            logger.warning(f"Cycle analysis failed: {analysis_results['error']}")
            return None

        try:
            import matplotlib.pyplot as plt

            # Create plot
            fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(15, 15), sharex=False,
                                                gridspec_kw={'height_ratios': [3, 2, 2]})

            # Plot 1: Price with cycles overlay
            ax1.plot(price_series.index, price_series.values, color='black', label='Price')
            ax1.set_title('Price with Cycle Overlay')
            ax1.set_ylabel('Price')
            ax1.legend(loc='upper left')
            ax1.grid(True, alpha=0.3)

            # Add dominant cycle if available
            dominant_cycle = analysis_results.get('dominant_cycle')

            if dominant_cycle:
                period = dominant_cycle['period']

                # Generate a synthetic cycle
                x = np.linspace(0, len(price_series), len(price_series))

                # Scale to price range
                price_range = price_series.max() - price_series.min()
                price_mid = (price_series.max() + price_series.min()) / 2

                # Create sine wave with proper period
                cycle = np.sin(2 * np.pi * x / period) * price_range * 0.1 + price_mid

                # Plot cycle
                ax1.plot(price_series.index, cycle, color='red', linestyle='--',
                         label=f"{period:.1f}-day cycle", alpha=0.7)

                # Annotate cycle information
                cycle_desc = f"Dominant Cycle: {period:.1f} days"
                if 'phase_description' in dominant_cycle:
                    cycle_desc += f" ({dominant_cycle['phase_description']})"

                ax1.annotate(cycle_desc, xy=(0.02, 0.05), xycoords='axes fraction',
                             bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="red", alpha=0.7))

                # Add next peak/trough if available
                if 'next_peak' in dominant_cycle and 'next_trough' in dominant_cycle:
                    ax1.annotate(f"Next Peak: {dominant_cycle['next_peak']}",
                                 xy=(0.02, 0.95), xycoords='axes fraction',
                                 bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="green", alpha=0.7))

                    ax1.annotate(f"Next Trough: {dominant_cycle['next_trough']}",
                                 xy=(0.02, 0.88), xycoords='axes fraction',
                                 bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="red", alpha=0.7))

            # Plot 2: FFT results
            fft_results = analysis_results['methods']['fft']

            if 'error' not in fft_results and 'data' in fft_results:
                periods = fft_results['data']['periods']
                mags = fft_results['data']['magnitudes']

                ax2.plot(periods, mags, color='blue')
                ax2.set_title('FFT Analysis (Period Spectrum)')
                ax2.set_xlabel('Period (days)')
                ax2.set_ylabel('Magnitude')
                ax2.grid(True, alpha=0.3)

                # Mark top cycles
                if 'cycles' in fft_results:
                    for cycle in fft_results['cycles'][:3]:  # Top 3 cycles
                        period = cycle['period']
                        idx = np.abs(np.array(periods) - period).argmin()
                        mag = mags[idx]

                        ax2.plot(period, mag, 'ro')
                        ax2.annotate(f"{period:.1f}d", xy=(period, mag),
                                     xytext=(5, 5), textcoords='offset points')

            # Plot 3: Autocorrelation results
            ac_results = analysis_results['methods']['autocorrelation']

            if 'error' not in ac_results and 'data' in ac_results:
                lags = ac_results['data']['lags']
                acf_values = ac_results['data']['acf_values']

                ax3.plot(lags, acf_values, color='green')
                ax3.set_title('Autocorrelation Analysis')
                ax3.set_xlabel('Lag (days)')
                ax3.set_ylabel('Autocorrelation')
                ax3.axhline(y=0, color='black', linestyle='-', alpha=0.3)
                ax3.grid(True, alpha=0.3)

                # Mark top cycles
                if 'cycles' in ac_results:
                    for cycle in ac_results['cycles'][:3]:  # Top 3 cycles
                        lag = cycle['period']
                        acf = cycle['correlation']

                        ax3.plot(lag, acf, 'ro')
                        ax3.annotate(f"{lag:.1f}d", xy=(lag, acf),
                                     xytext=(5, 5), textcoords='offset points')

            plt.tight_layout()

            if show_plot:
                plt.show()
                return None
            else:
                return fig

        except Exception as e:
            logger.error(f"Error visualizing cycles: {e}")
            return None

    def forecast_cycle(self, time_series, forecast_periods=30, price_col='close', method='combined'):
        """
        Forecast future values based on detected cycles.

        Parameters:
        -----------
        time_series : pd.DataFrame or pd.Series
            Time series data
        forecast_periods : int
            Number of periods to forecast
        price_col : str
            Column name for price data (if DataFrame)
        method : str
            Forecasting method ('fft', 'autocorrelation', 'combined')

        Returns:
        --------
        pd.Series
            Forecasted values
        """
        # Handle different input types
        if isinstance(time_series, pd.DataFrame):
            if price_col not in time_series.columns:
                logger.warning(f"Price column '{price_col}' not found in DataFrame")
                return None

            # Extract price series
            price_series = time_series[price_col]
        elif isinstance(time_series, pd.Series):
            price_series = time_series
        else:
            logger.warning("Input must be a pandas DataFrame or Series")
            return None

        # Detect cycles
        analysis_results = self.detect_dominant_cycles(time_series, price_col)

        # Check if analysis was successful
        if 'error' in analysis_results:
            logger.warning(f"Cycle analysis failed: {analysis_results['error']}")
            return None

        try:
            # Determine which cycles to use based on method
            cycles_to_use = []

            if method == 'fft':
                if 'fft' in analysis_results['methods'] and 'cycles' in analysis_results['methods']['fft']:
                    cycles_to_use = analysis_results['methods']['fft']['cycles']
            elif method == 'autocorrelation':
                if 'autocorrelation' in analysis_results['methods'] and 'cycles' in analysis_results['methods'][
                    'autocorrelation']:
                    cycles_to_use = analysis_results['methods']['autocorrelation']['cycles']
            else:  # combined
                cycles_to_use = analysis_results.get('all_cycles', [])

            if not cycles_to_use:
                logger.warning("No cycles available for forecasting")
                return None

            # Use top 3 cycles or all if less than 3
            cycles_to_use = sorted(cycles_to_use, key=lambda x: x.get('strength', 0), reverse=True)[
                            :min(3, len(cycles_to_use))]

            # Get current trend
            n = len(price_series)

            # Linear regression on recent data (last 20% of data)
            start_idx = max(0, int(n * 0.8))
            x = np.arange(n - start_idx)
            y = price_series.values[start_idx:]

            slope, intercept, r_value, p_value, std_err = stats.linregress(x, y)
            trend = lambda x: slope * x + intercept

            # Forecast future values
            forecast_index = pd.date_range(
                start=price_series.index[-1] + pd.Timedelta(days=1),
                periods=forecast_periods,
                freq=pd.infer_freq(price_series.index) or 'D'
            )

            # Initialize forecast with trend
            forecast_values = np.array([trend(i) for i in range(forecast_periods)])

            # Add cycle components
            for cycle in cycles_to_use:
                period = cycle['period']
                strength = cycle.get('strength', 0.5)

                # Get last known phase
                phase = cycle.get('current_phase', 0)

                # Generate cycle values
                for i in range(forecast_periods):
                    # Calculate phase at this point
                    current_phase = (phase + (i / period)) % 1

                    # Convert phase to angle (0 to 2π)
                    angle = current_phase * 2 * np.pi

                    # Add sine wave component (scaled by strength and adjusted for amplitude)
                    price_std = np.std(price_series.values)
                    forecast_values[i] += np.sin(angle) * price_std * 0.1 * strength

            # Create forecast series
            forecast = pd.Series(forecast_values, index=forecast_index)

            logger.info(f"Generated {forecast_periods} period forecast using {len(cycles_to_use)} cycles")
            return forecast

        except Exception as e:
            logger.error(f"Error generating forecast: {e}")
            return None

    def run_analysis(self, time_series, price_col='close', sampling_freq=1.0):
        """
        Run a comprehensive cycle analysis.

        Parameters:
        -----------
        time_series : pd.DataFrame or pd.Series
            Time series data
        price_col : str
            Column name for price data (if DataFrame)
        sampling_freq : float
            Sampling frequency in data points per day

        Returns:
        --------
        tuple
            (dict with analysis results, forecasted values)
        """
        # Detect cycles
        analysis_results = self.detect_dominant_cycles(time_series, price_col, sampling_freq=sampling_freq)

        # Generate forecast
        forecast = self.forecast_cycle(time_series, forecast_periods=30, price_col=price_col)

        # Visualize
        self.visualize_cycles(time_series, analysis_results, price_col)

        logger.info("Completed cycle analysis")
        return analysis_results, forecast

        # Factory function to get a cycle analyzer

def get_cycle_analyzer(max_cycle_length=None, min_cycle_length=5):
    """
    Get a configured cycle analyzer.

    Parameters:
    -----------
    max_cycle_length : int, optional
        Maximum cycle length to detect (in days)
    min_cycle_length : int
        Minimum cycle length to detect (in days)

    Returns:
    --------
    CycleAnalyzer
        Configured analyzer instance
    """
    return CycleAnalyzer(max_cycle_length=max_cycle_length, min_cycle_length=min_cycle_length)