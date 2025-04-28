"""
Module for fractal analysis of time series.
Analyzes fractal properties of price and volatility data.
"""
import os
import sys
import numpy as np
import pandas as pd
from scipy import signal
from scipy.stats import linregress
import matplotlib.pyplot as plt
import pywt
import logging

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
from crypto_volatility_indicator.utils.logger import get_logger

sys.path.insert(0, project_root)
# Set up logger
logger = get_logger(__name__)


class FractalAnalyzer:
    """
    Advanced fractal analysis module for cryptocurrency price data.

    This module implements several fractal analysis techniques:
    1. Hurst Exponent calculation
    2. Fractal Dimension estimation
    3. Wavelet Multi-Resolution Analysis
    4. Self-similarity detection
    5. Multifractal spectrum analysis
    """

    def __init__(self, config=None):
        """
        Initialize the FractalAnalyzer.

        Parameters:
        -----------
        config : dict, optional
            Configuration parameters
        """
        # Set default configuration
        self.config = config or {}

        # Configure analysis parameters
        self.max_lag = self.config.get('max_lag', 20)
        self.max_scale = self.config.get('max_scale', 1000)

        logger.info(f"FractalAnalyzer initialized with max_lag={self.max_lag}")

    def calculate_hurst_exponent(self, time_series, max_lag=None):
        """
        Calculate the Hurst exponent using the rescaled range (R/S) method.

        Parameters:
        -----------
        time_series : array-like
            Time series data
        max_lag : int, optional
            Maximum lag for R/S calculation

        Returns:
        --------
        float
            Hurst exponent
        dict
            Additional metrics
        """
        # Use class-level max_lag if not provided
        if max_lag is None:
            max_lag = self.max_lag

        ts = np.asarray(time_series)
        if len(ts) < 100:
            logger.warning("Warning: Time series too short for reliable Hurst estimation")
            return 0.5, {'error': 'Time series too short'}

        # Remove NaN values
        ts = ts[~np.isnan(ts)]

        # Calculate returns
        returns = np.diff(np.log(ts))

        # Remove zeros to avoid log issues
        returns = returns[returns != 0]

        if len(returns) < max_lag:
            return 0.5, {'error': 'Time series too short'}

        # Calculate R/S values for different lags
        lags = range(10, max_lag)
        rs_values = []

        for lag in lags:
            rs = self._calculate_rs(returns, lag)
            rs_values.append(rs)

        # Calculate Hurst exponent using linear regression
        log_lags = np.log10(lags)
        log_rs = np.log10(rs_values)

        slope, intercept, r_value, p_value, std_err = linregress(log_lags, log_rs)

        # Interpret the Hurst exponent
        if slope < 0.4:
            interpretation = "anti-persistent (mean-reverting)"
        elif slope > 0.6:
            interpretation = "persistent (trending)"
        else:
            interpretation = "random walk"

        return slope, {
            'intercept': intercept,
            'r_squared': r_value ** 2,
            'p_value': p_value,
            'std_err': std_err,
            'interpretation': interpretation
        }

    def _calculate_rs(self, time_series, lag):
        """
        Calculate the Rescaled Range (R/S) for a given lag.

        Parameters:
        -----------
        time_series : array-like
            Time series data
        lag : int
            Lag for R/S calculation

        Returns:
        --------
        float
            R/S value
        """
        # Split the time series into chunks of size 'lag'
        remainder = len(time_series) % lag
        if remainder != 0:
            time_series = time_series[:-remainder]

        n_chunks = len(time_series) // lag

        if n_chunks == 0:
            return np.nan

        rs_values = []

        for i in range(n_chunks):
            chunk = time_series[i * lag:(i + 1) * lag]

            # Calculate mean and standard deviation
            mean = np.mean(chunk)
            std = np.std(chunk)

            if std == 0:
                continue

            # Calculate cumulative deviation
            cum_dev = np.cumsum(chunk - mean)

            # Calculate range
            r = np.max(cum_dev) - np.min(cum_dev)

            # Calculate rescaled range
            rs = r / std

            rs_values.append(rs)

        # Return average R/S value
        if len(rs_values) == 0:
            return np.nan

        return np.mean(rs_values)

    def calculate_fractal_dimension(self, time_series, eps_values=None):
        """
        Calculate the fractal dimension using box-counting method.

        Parameters:
        -----------
        time_series : array-like
            Time series data
        eps_values : array-like, optional
            Box sizes to use for counting

        Returns:
        --------
        float
            Fractal dimension
        dict
            Additional metrics
        """
        ts = np.asarray(time_series)

        # Remove NaN values
        ts = ts[~np.isnan(ts)]

        # Normalize to [0, 1] range
        ts_norm = (ts - np.min(ts)) / (np.max(ts) - np.min(ts))

        # Set default eps values if not provided
        if eps_values is None:
            # Use logarithmically spaced values
            eps_values = np.logspace(-3, 0, 10)

        # Calculate box counts for different box sizes
        counts = []

        for eps in eps_values:
            count = self._count_boxes(ts_norm, eps)
            counts.append(count)

        # Calculate fractal dimension using linear regression
        log_eps = np.log(eps_values)
        log_counts = np.log(counts)

        # Linear regression to find the slope
        slope, intercept, r_value, p_value, std_err = linregress(log_eps, log_counts)

        # Fractal dimension is negative of the slope
        fractal_dim = -slope

        return fractal_dim, {
            'intercept': intercept,
            'r_squared': r_value ** 2,
            'p_value': p_value,
            'std_err': std_err,
            'complexity': 'high' if fractal_dim > 1.5 else 'medium' if fractal_dim > 1.2 else 'low'
        }

    def _count_boxes(self, time_series, eps):
        """
        Count the number of boxes of size eps needed to cover the time series.

        Parameters:
        -----------
        time_series : array-like
            Time series data (normalized to [0, 1])
        eps : float
            Box size

        Returns:
        --------
        int
            Number of boxes
        """
        # Create a grid of boxes
        x_boxes = int(np.ceil(1.0 / eps))
        y_boxes = int(np.ceil(1.0 / eps))

        # Initialize box occupancy
        box_occupancy = np.zeros((x_boxes, y_boxes), dtype=bool)

        # Fill boxes
        for i in range(len(time_series) - 1):
            x1, y1 = i / len(time_series), time_series[i]
            x2, y2 = (i + 1) / len(time_series), time_series[i + 1]

            # Find boxes containing the line segment (x1,y1) to (x2,y2)
            # This is a simplified approach using Bresenham's line algorithm
            box_x1, box_y1 = int(x1 / eps), int(y1 / eps)
            box_x2, box_y2 = int(x2 / eps), int(y2 / eps)

            # Mark boxes as occupied
            if box_x1 == box_x2 and box_y1 == box_y2:
                # Same box
                box_occupancy[box_x1, box_y1] = True
            else:
                # Use Bresenham's algorithm to find boxes intersected by the line segment
                dx = abs(box_x2 - box_x1)
                dy = abs(box_y2 - box_y1)
                sx = 1 if box_x1 < box_x2 else -1
                sy = 1 if box_y1 < box_y2 else -1
                err = dx - dy

                while box_x1 != box_x2 or box_y1 != box_y2:
                    if 0 <= box_x1 < x_boxes and 0 <= box_y1 < y_boxes:
                        box_occupancy[box_x1, box_y1] = True

                    e2 = 2 * err
                    if e2 > -dy:
                        err -= dy
                        box_x1 += sx
                    if e2 < dx:
                        err += dx
                        box_y1 += sy

                # Mark final box
                if 0 <= box_x2 < x_boxes and 0 <= box_y2 < y_boxes:
                    box_occupancy[box_x2, box_y2] = True

        # Count occupied boxes
        return np.sum(box_occupancy)

    def wavelet_analysis(self, time_series, wavelet='db4', max_level=5, mode='soft'):
        """
        Perform wavelet analysis to detect patterns at different scales.

        Parameters:
        -----------
        time_series : array-like
            Time series data
        wavelet : str
            Wavelet type (e.g., 'db4', 'haar', 'sym4')
        max_level : int
            Maximum decomposition level
        mode : str
            Thresholding mode ('soft' or 'hard')

        Returns:
        --------
        dict
            Wavelet analysis results
        """
        ts = np.asarray(time_series)

        # Remove NaN values
        ts = ts[~np.isnan(ts)]

        # Ensure the length is a power of 2 (required for some wavelet transforms)
        # by padding or truncating
        power = int(np.log2(len(ts)))
        new_length = 2 ** power
        if new_length < len(ts):
            ts = ts[:new_length]
        else:
            ts = np.pad(ts, (0, new_length - len(ts)), 'constant', constant_values=(0, ts[-1]))

        # Perform discrete wavelet transform
        coeffs = pywt.wavedec(ts, wavelet, level=max_level)

        # Extract approximation and detail coefficients
        approx = coeffs[0]
        details = coeffs[1:]

        # Calculate energy of detail coefficients at each level
        energies = [np.sum(d ** 2) for d in details]
        total_energy = sum(energies)

        # Normalize energies
        if total_energy > 0:
            norm_energies = [e / total_energy for e in energies]
        else:
            norm_energies = [0] * len(energies)

        # Calculate wavelet entropy (measure of complexity)
        wavelet_entropy = -sum([e * np.log2(e) if e > 0 else 0 for e in norm_energies])

        # Detect dominance of specific scales
        dominant_scale = np.argmax(norm_energies) + 1 if norm_energies else None

        # Wavelet variance (variance of wavelet coefficients at each level)
        wavelet_variances = [np.var(d) for d in details]

        # Apply thresholding to coefficients if requested
        if mode is not None:
            # Threshold calculation
            sigma = np.median(np.abs(coeffs[-1])) / 0.6745
            threshold = sigma * np.sqrt(2 * np.log(len(ts)))

            # Apply thresholding
            new_coeffs = list(coeffs)

            for i in range(1, len(new_coeffs)):
                new_coeffs[i] = pywt.threshold(new_coeffs[i], threshold, mode=mode)

            # Reconstruct denoised signal
            denoised = pywt.waverec(new_coeffs, wavelet)

            # Make sure the denoised signal has the same length as the input
            denoised = denoised[:len(ts)]
        else:
            denoised = None

        return {
            'wavelet_type': wavelet,
            'max_level': max_level,
            'normalized_energies': norm_energies,
            'wavelet_entropy': wavelet_entropy,
            'dominant_scale': dominant_scale,
            'wavelet_variances': wavelet_variances,
            'denoised_signal': denoised
        }

    def detect_self_similarity(self, time_series, window_sizes=None, threshold=0.7):
        """
        Detect self-similarity in the time series by comparing patterns at different scales.

        Parameters:
        -----------
        time_series : array-like
            Time series data
        window_sizes : list, optional
            List of window sizes to consider
        threshold : float
            Correlation threshold for identifying similar patterns

        Returns:
        --------
        dict
            Self-similarity analysis results
        """
        ts = np.asarray(time_series)

        # Remove NaN values
        ts = ts[~np.isnan(ts)]

        # Set default window sizes if not provided
        if window_sizes is None:
            window_sizes = [int(len(ts) / 10), int(len(ts) / 5), int(len(ts) / 3)]
            window_sizes = [w for w in window_sizes if w > 10]  # Ensure reasonable sizes

        if not window_sizes:
            return {'error': 'Time series too short for self-similarity analysis'}

        # Calculate self-similarity for each window size
        similarities = []

        for window_size in window_sizes:
            # Calculate autocorrelation for lags up to window_size
            autocorr = self._autocorrelation(ts, window_size)

            # Find peaks in autocorrelation
            peaks, _ = signal.find_peaks(autocorr, height=threshold)

            # Calculate mean peak value
            mean_peak_value = np.mean(autocorr[peaks]) if len(peaks) > 0 else 0

            similarities.append({
                'window_size': window_size,
                'peak_lags': peaks.tolist(),
                'peak_values': autocorr[peaks].tolist() if len(peaks) > 0 else [],
                'mean_peak_value': mean_peak_value,
                'has_significant_similarity': mean_peak_value > threshold
            })

        # Overall self-similarity metric
        significant_counts = sum(1 for s in similarities if s['has_significant_similarity'])
        overall_self_similarity = significant_counts / len(similarities) if similarities else 0

        return {
            'window_analyses': similarities,
            'overall_self_similarity': overall_self_similarity,
            'has_self_similarity': overall_self_similarity > 0.5
        }

    def _autocorrelation(self, time_series, max_lag):
        """
        Calculate autocorrelation for lags up to max_lag.

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

        # Calculate variance (for normalization)
        var = np.var(ts)

        if var == 0:
            return np.zeros(max_lag + 1)

        # Calculate autocorrelation for lags up to max_lag
        autocorr = np.zeros(max_lag + 1)

        for lag in range(max_lag + 1):
            # Correlation between ts[:-lag] and ts[lag:]
            if lag == 0:
                autocorr[lag] = 1.0  # Autocorrelation at lag 0 is 1
            else:
                autocorr[lag] = np.corrcoef(ts[:-lag], ts[lag:])[0, 1]

        return autocorr

    def analyze_multifractal_spectrum(self, time_series, q_values=None, max_scale=1000):
        """
        Analyze the multifractal spectrum of the time series.

        Parameters:
        -----------
        time_series : array-like
            Time series data
        q_values : array-like, optional
            Range of q values for multifractal analysis
        max_scale : int
            Maximum scale for analysis

        Returns:
        --------
        dict
            Multifractal analysis results
        """
        # This is a simplified implementation of multifractal analysis
        # For a full implementation, consider using specialized libraries

        ts = np.asarray(time_series)

        # Remove NaN values
        ts = ts[~np.isnan(ts)]

        # Set default q values if not provided
        if q_values is None:
            q_values = np.linspace(-5, 5, 11)

        # Calculate fluctuation function for different scales and q values
        scales = np.logspace(1, np.log10(min(max_scale, len(ts) // 4)), 10).astype(int)

        # Initialize arrays for results
        hq = np.zeros(len(q_values))

        # For each q, calculate the generalized Hurst exponent
        for i, q in enumerate(q_values):
            fluctuations = []

            for scale in scales:
                fluct = self._dfa_fluctuation(ts, scale, q)
                fluctuations.append(fluct)

            # Calculate the slope (generalized Hurst exponent)
            log_scales = np.log(scales)
            log_fluctuations = np.log(fluctuations)

            # Linear regression
            slope, _, _, _, _ = linregress(log_scales, log_fluctuations)

            hq[i] = slope

        # Calculate multifractal spectrum
        tq = q_values * hq - 1
        hq_diff = np.diff(hq) / np.diff(q_values)

        # Calculate width of multifractal spectrum
        if len(hq) > 2:
            spectrum_width = np.max(hq) - np.min(hq)
        else:
            spectrum_width = 0

        return {
            'q_values': q_values.tolist(),
            'generalized_hurst': hq.tolist(),
            'tau_q': tq.tolist(),
            'hq_derivative': hq_diff.tolist() if len(hq_diff) > 0 else [],
            'spectrum_width': spectrum_width,
            'is_multifractal': spectrum_width > 0.2  # Arbitrary threshold
        }

    def _dfa_fluctuation(self, time_series, scale, q):
        """
        Calculate the detrended fluctuation function for a given scale and q.

        Parameters:
        -----------
        time_series : array-like
            Time series data
        scale : int
            Scale for detrended fluctuation analysis
        q : float
            Order of the moment

        Returns:
        --------
        float
            Fluctuation value
        """
        # Cumulative sum
        y = np.cumsum(time_series - np.mean(time_series))

        # Calculate number of segments
        n_segments = len(y) // scale

        if n_segments == 0:
            return np.nan

        fluctuations = []

        for i in range(n_segments):
            # Extract segment
            segment = y[i * scale:(i + 1) * scale]

            # Calculate trend (polynomial fit)
            x = np.arange(len(segment))
            coeffs = np.polyfit(x, segment, 1)
            trend = np.polyval(coeffs, x)

            # Calculate fluctuation (RMS of detrended segment)
            fluct = np.sqrt(np.mean((segment - trend) ** 2))
            fluctuations.append(fluct)

        # Return the q-th order fluctuation
        fluctuations = np.array(fluctuations)

        # Handle the case q=0 separately (geometric mean)
        if q == 0:
            return np.exp(np.mean(np.log(fluctuations)))
        else:
            return np.mean(fluctuations ** q) ** (1 / q)

    def analyze_volatility_scaling(self, returns, time_scales=None):
        """
        Analyze how volatility scales with time (volatility signature plot).

        Parameters:
        -----------
        returns : array-like
            Returns time series
        time_scales : list, optional
            List of time scales to analyze

        Returns:
        --------
        dict
            Scaling analysis results
        """
        if time_scales is None:
            # Default time scales (powers of 2)
            time_scales = [1, 2, 4, 8, 16, 32, 64]

        # Calculate volatility at different time scales
        volatilities = []

        for scale in time_scales:
            # Aggregate returns to the given time scale
            agg_returns = self._aggregate_returns(returns, scale)

            # Calculate volatility
            vol = np.std(agg_returns)

            # Scale to annualized value assuming daily returns input
            # Adjust as needed for other frequencies
            annualized_vol = vol * np.sqrt(252 / scale)

            volatilities.append(annualized_vol)

        # Calculate scaling exponent using linear regression
        log_scales = np.log(time_scales)
        log_vols = np.log(volatilities)

        scaling_exponent, intercept, r_value, p_value, std_err = linregress(log_scales, log_vols)

        # Interpret scaling behavior
        if scaling_exponent > -0.4:
            interpretation = "rough (anti-persistent)"
        elif scaling_exponent < -0.6:
            interpretation = "smooth (persistent)"
        else:
            interpretation = "Brownian (random walk)"

        return {
            'time_scales': time_scales,
            'volatilities': volatilities,
            'scaling_exponent': scaling_exponent,
            'r_squared': r_value ** 2,
            'interpretation': interpretation
        }

    def _aggregate_returns(self, returns, scale):
        """
        Aggregate returns to a larger time scale.

        Parameters:
        -----------
        returns : array-like
            Returns time series
        scale : int
            Time scale to aggregate to

        Returns:
        --------
        array-like
            Aggregated returns
        """
        # Truncate series to multiple of scale
        trunc_len = (len(returns) // scale) * scale
        trunc_returns = returns[:trunc_len]

        # Reshape and sum
        return np.sum(trunc_returns.reshape(-1, scale), axis=1)

    def visualize_fractal_analysis(self, time_series, title="Fractal Analysis", show_plot=True):
        """
        Create visualizations of fractal analysis results.

        Parameters:
        -----------
        time_series : array-like
            Time series data
        title : str
            Title for the visualization
        show_plot : bool
            Whether to display the plot or return the figure

        Returns:
        --------
        matplotlib.figure.Figure or None
            Figure object if show_plot is False, None otherwise
        """
        ts = np.asarray(time_series)

        # Remove NaN values
        ts = ts[~np.isnan(ts)]

        # Calculate various fractal metrics
        hurst, hurst_metrics = self.calculate_hurst_exponent(ts)
        fd, fd_metrics = self.calculate_fractal_dimension(ts)
        wavelet_results = self.wavelet_analysis(ts)
        self_similarity = self.detect_self_similarity(ts)

        # Create a figure with multiple subplots
        fig, axs = plt.subplots(2, 2, figsize=(15, 10))

        # Plot 1: Original time series
        axs[0, 0].plot(ts)
        axs[0, 0].set_title(f"Original Time Series\nHurst={hurst:.3f}, FD={fd:.3f}")
        axs[0, 0].grid(True, alpha=0.3)

        # Plot 2: Rescaled range analysis
        max_lag = min(100, len(ts) // 10)
        lags = range(10, max_lag)
        rs_values = [self._calculate_rs(np.diff(np.log(ts)), lag) for lag in lags]

        # Filter out any NaN values
        valid_indices = ~np.isnan(rs_values)
        lags_filtered = np.array(lags)[valid_indices]
        rs_filtered = np.array(rs_values)[valid_indices]

        if len(lags_filtered) > 0:
            axs[0, 1].loglog(lags_filtered, rs_filtered, 'bo-', alpha=0.7)

            # Add trendline
            log_lags = np.log10(lags_filtered)
            log_rs = np.log10(rs_filtered)
            slope, intercept, _, _, _ = linregress(log_lags, log_rs)

            trendline_x = np.logspace(np.log10(min(lags_filtered)), np.log10(max(lags_filtered)), 100)
            trendline_y = 10 ** (intercept + slope * np.log10(trendline_x))

            axs[0, 1].loglog(trendline_x, trendline_y, 'r-', label=f'Slope={slope:.3f}')

        axs[0, 1].set_title(f"Rescaled Range Analysis\nHurst={hurst:.3f} ({hurst_metrics['interpretation']})")
        axs[0, 1].set_xlabel("Lag")
        axs[0, 1].set_ylabel("Rescaled Range (R/S)")
        axs[0, 1].grid(True, alpha=0.3)
        axs[0, 1].legend()

        # Plot 3: Wavelet analysis
        if 'normalized_energies' in wavelet_results:
            energies = wavelet_results['normalized_energies']
            scales = [f"D{i + 1}" for i in range(len(energies))]

            axs[1, 0].bar(scales, energies, alpha=0.7)
            axs[1, 0].set_title(f"Wavelet Energy Distribution\nEntropy={wavelet_results['wavelet_entropy']:.3f}")
            axs[1, 0].set_xlabel("Wavelet Scale")
            axs[1, 0].set_ylabel("Normalized Energy")
            axs[1, 0].grid(True, alpha=0.3)

        # Plot 4: Self-similarity analysis
        if 'window_analyses' in self_similarity:
            for analysis in self_similarity['window_analyses']:
                window_size = analysis['window_size']

                # Calculate autocorrelation
                autocorr = self._autocorrelation(ts, window_size)

                # Plot autocorrelation
                axs[1, 1].plot(autocorr, label=f'Window={window_size}', alpha=0.7)

            axs[1, 1].set_title(
                f"Self-Similarity Analysis\nOverall={self_similarity['overall_self_similarity']:.3f}")
            axs[1, 1].set_xlabel("Lag")
            axs[1, 1].set_ylabel("Autocorrelation")
            axs[1, 1].grid(True, alpha=0.3)
            axs[1, 1].legend()

        plt.tight_layout()
        plt.suptitle(title, fontsize=16)
        plt.subplots_adjust(top=0.9)

        # Display or return the figure
        if show_plot:
            plt.show()
            return None
        else:
            return fig

    def run_analysis(self, time_series, name="Time Series"):
        """
        Run a complete fractal analysis on the time series.

        Parameters:
        -----------
        time_series : array-like
            Time series data
        name : str
            Name of the time series for display

        Returns:
        --------
        dict
            Complete analysis results
        """
        ts = np.asarray(time_series)

        # Remove NaN values
        ts = ts[~np.isnan(ts)]

        if len(ts) < 100:
            logger.warning("Time series too short for fractal analysis")
            return {'error': 'Time series too short for fractal analysis'}

        # Run all analyses
        logger.info(f"Running fractal analysis on {name} ({len(ts)} data points)")

        try:
            hurst, hurst_metrics = self.calculate_hurst_exponent(ts)
            fd, fd_metrics = self.calculate_fractal_dimension(ts)
            wavelet_results = self.wavelet_analysis(ts)
            self_similarity = self.detect_self_similarity(ts)

            # Simplified multifractal analysis if the time series is long enough
            if len(ts) >= 500:
                multifractal = self.analyze_multifractal_spectrum(ts)
            else:
                multifractal = {'note': 'Time series too short for reliable multifractal analysis'}

            # Create a comprehensive report
            report = {
                'name': name,
                'length': len(ts),
                'hurst_exponent': hurst,
                'hurst_interpretation': hurst_metrics['interpretation'],
                'fractal_dimension': fd,
                'complexity': fd_metrics['complexity'],
                'wavelet_entropy': wavelet_results['wavelet_entropy'],
                'dominant_scale': wavelet_results['dominant_scale'],
                'self_similarity': self_similarity['overall_self_similarity'],
                'has_self_similarity': self_similarity['has_self_similarity'],
                'is_multifractal': multifractal.get('is_multifractal', False),
                'spectrum_width': multifractal.get('spectrum_width', 0),
                'raw_results': {
                    'hurst': hurst_metrics,
                    'fractal_dimension': fd_metrics,
                    'wavelet': wavelet_results,
                    'self_similarity': self_similarity,
                    'multifractal': multifractal
                }
            }

            logger.info(f"Fractal analysis completed for {name}: Hurst={hurst:.3f}, FD={fd:.3f}")
            return report

        except Exception as e:
            logger.error(f"Error in fractal analysis: {e}")
            return {'error': str(e)}

# Factory function to get a fractal analyzer
def get_fractal_analyzer(config=None):
    """
    Get a fractal analyzer instance.

    Parameters:
    -----------
    config : dict, optional
        Configuration parameters

    Returns:
    --------
    FractalAnalyzer
        Fractal analyzer instance
    """
    return FractalAnalyzer(config)