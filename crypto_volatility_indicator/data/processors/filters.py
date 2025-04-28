"""
Module for filtering and transforming data.
Provides specialized filters for time series data.
"""
import os
import sys
import pandas as pd
import numpy as np
from scipy import signal
import logging
from datetime import datetime, timedelta
import pywt
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, project_root)
from crypto_volatility_indicator.utils.logger import get_data_logger
# Set up logger
logger = get_data_logger()


class DataFilter:
    """
    Filter for time series data.

    This class provides various filtering techniques for noise reduction
    and signal processing in time series financial data.
    """

    def __init__(self):
        """Initialize the data filter."""
        logger.info("DataFilter initialized")

    def moving_average_filter(self, data, window=5, center=False):
        """
        Apply a simple moving average filter.

        Parameters:
        -----------
        data : array-like or pd.Series
            Input time series data
        window : int
            Window size for moving average
        center : bool
            If True, the window is centered around the current point

        Returns:
        --------
        array-like or pd.Series
            Filtered data
        """
        if isinstance(data, pd.Series):
            return data.rolling(window=window, center=center).mean()
        else:
            # Convert to pandas Series for rolling window
            series = pd.Series(data)
            filtered = series.rolling(window=window, center=center).mean()
            return filtered.values

    def exponential_moving_average_filter(self, data, span=5, adjust=False):
        """
        Apply an exponential moving average filter.

        Parameters:
        -----------
        data : array-like or pd.Series
            Input time series data
        span : int
            Window size for EMA calculation
        adjust : bool
            If True, use adjusted calculation

        Returns:
        --------
        array-like or pd.Series
            Filtered data
        """
        if isinstance(data, pd.Series):
            return data.ewm(span=span, adjust=adjust).mean()
        else:
            # Convert to pandas Series for EWM
            series = pd.Series(data)
            filtered = series.ewm(span=span, adjust=adjust).mean()
            return filtered.values

    def low_pass_filter(self, data, cutoff=0.1, fs=1.0, order=5):
        """
        Apply a low-pass Butterworth filter.

        Parameters:
        -----------
        data : array-like
            Input time series data
        cutoff : float
            Cutoff frequency (normalized to Nyquist frequency)
        fs : float
            Sampling frequency
        order : int
            Filter order

        Returns:
        --------
        array-like
            Filtered data
        """
        nyquist = 0.5 * fs
        normal_cutoff = cutoff / nyquist

        # Design the filter
        b, a = signal.butter(order, normal_cutoff, btype='low', analog=False)

        # Apply the filter
        if isinstance(data, pd.Series):
            filtered = signal.filtfilt(b, a, data.values)
            return pd.Series(filtered, index=data.index)
        else:
            filtered = signal.filtfilt(b, a, data)
            return filtered

    def high_pass_filter(self, data, cutoff=0.1, fs=1.0, order=5):
        """
        Apply a high-pass Butterworth filter.

        Parameters:
        -----------
        data : array-like
            Input time series data
        cutoff : float
            Cutoff frequency (normalized to Nyquist frequency)
        fs : float
            Sampling frequency
        order : int
            Filter order

        Returns:
        --------
        array-like
            Filtered data
        """
        nyquist = 0.5 * fs
        normal_cutoff = cutoff / nyquist

        # Design the filter
        b, a = signal.butter(order, normal_cutoff, btype='high', analog=False)

        # Apply the filter
        if isinstance(data, pd.Series):
            filtered = signal.filtfilt(b, a, data.values)
            return pd.Series(filtered, index=data.index)
        else:
            filtered = signal.filtfilt(b, a, data)
            return filtered

    def band_pass_filter(self, data, lowcut=0.05, highcut=0.2, fs=1.0, order=5):
        """
        Apply a band-pass Butterworth filter.

        Parameters:
        -----------
        data : array-like
            Input time series data
        lowcut : float
            Lower cutoff frequency
        highcut : float
            Upper cutoff frequency
        fs : float
            Sampling frequency
        order : int
            Filter order

        Returns:
        --------
        array-like
            Filtered data
        """
        nyquist = 0.5 * fs
        low = lowcut / nyquist
        high = highcut / nyquist

        # Design the filter
        b, a = signal.butter(order, [low, high], btype='band', analog=False)

        # Apply the filter
        if isinstance(data, pd.Series):
            filtered = signal.filtfilt(b, a, data.values)
            return pd.Series(filtered, index=data.index)
        else:
            filtered = signal.filtfilt(b, a, data)
            return filtered

    def kalman_filter(self, data, process_variance=1e-5, measurement_variance=1e-2):
        """
        Apply a simple Kalman filter.

        Parameters:
        -----------
        data : array-like
            Input time series data
        process_variance : float
            Process variance
        measurement_variance : float
            Measurement variance

        Returns:
        --------
        array-like
            Filtered data
        """
        # Convert to numpy array if needed
        if isinstance(data, pd.Series):
            values = data.values
            is_series = True
            index = data.index
        else:
            values = np.array(data)
            is_series = False

        n = len(values)

        # Initialize
        filtered = np.zeros(n)
        prediction = values[0]
        prediction_variance = 1.0

        for i in range(n):
            # Prediction update
            prediction_variance += process_variance

            # Measurement update
            kalman_gain = prediction_variance / (prediction_variance + measurement_variance)
            prediction = prediction + kalman_gain * (values[i] - prediction)
            prediction_variance = (1 - kalman_gain) * prediction_variance

            filtered[i] = prediction

        if is_series:
            return pd.Series(filtered, index=index)
        else:
            return filtered

    def savitzky_golay_filter(self, data, window=5, poly_order=2):
        """
        Apply a Savitzky-Golay filter for smoothing.

        Parameters:
        -----------
        data : array-like
            Input time series data
        window : int
            Window size (must be odd)
        poly_order : int
            Polynomial order

        Returns:
        --------
        array-like
            Filtered data
        """
        # Ensure window is odd
        if window % 2 == 0:
            window += 1

        if isinstance(data, pd.Series):
            filtered = signal.savgol_filter(data.values, window, poly_order)
            return pd.Series(filtered, index=data.index)
        else:
            filtered = signal.savgol_filter(data, window, poly_order)
            return filtered

    def median_filter(self, data, window=5):
        """
        Apply a median filter to remove outliers.

        Parameters:
        -----------
        data : array-like
            Input time series data
        window : int
            Window size

        Returns:
        --------
        array-like
            Filtered data
        """
        if isinstance(data, pd.Series):
            return data.rolling(window=window, center=True).median()
        else:
            # Convert to pandas Series for rolling window
            series = pd.Series(data)
            filtered = series.rolling(window=window, center=True).median()
            return filtered.values

    def wavelet_filter(self, data, wavelet='db4', level=1, mode='soft'):
        """
        Apply wavelet denoising.

        Parameters:
        -----------
        data : array-like
            Input time series data
        wavelet : str
            Wavelet type
        level : int
            Decomposition level
        mode : str
            Thresholding mode ('soft' or 'hard')

        Returns:
        --------
        array-like
            Filtered data
        """
        # Convert to numpy array if needed
        if isinstance(data, pd.Series):
            values = data.values
            is_series = True
            index = data.index
        else:
            values = np.array(data)
            is_series = False

        # Wavelet decomposition
        coeffs = pywt.wavedec(values, wavelet, level=level)

        # Threshold calculation
        sigma = np.median(np.abs(coeffs[-1])) / 0.6745
        threshold = sigma * np.sqrt(2 * np.log(len(values)))

        # Apply thresholding
        new_coeffs = list(coeffs)

        for i in range(1, len(new_coeffs)):
            if mode == 'soft':
                new_coeffs[i] = pywt.threshold(new_coeffs[i], threshold, mode='soft')
            else:
                new_coeffs[i] = pywt.threshold(new_coeffs[i], threshold, mode='hard')

        # Reconstruct signal
        filtered = pywt.waverec(new_coeffs, wavelet)

        # Make sure the filtered signal has the same length as the input
        filtered = filtered[:len(values)]

        if is_series:
            return pd.Series(filtered, index=index)
        else:
            return filtered

    def hampel_filter(self, data, window=5, threshold=3):
        """
        Apply a Hampel filter to remove outliers.

        Parameters:
        -----------
        data : array-like
            Input time series data
        window : int
            Window size
        threshold : float
            Threshold in terms of standard deviations

        Returns:
        --------
        array-like
            Filtered data
        """
        # Convert to numpy array if needed
        if isinstance(data, pd.Series):
            values = data.values
            is_series = True
            index = data.index
        else:
            values = np.array(data)
            is_series = False

        n = len(values)
        filtered = np.copy(values)

        k = int(window / 2)

        for i in range(n):
            # Define window boundaries
            start = max(0, i - k)
            end = min(n, i + k + 1)

            # Get window values
            window_vals = values[start:end]

            # Calculate median and MAD
            window_median = np.median(window_vals)
            window_mad = np.median(np.abs(window_vals - window_median))

            # Adjust for normally distributed data
            if window_mad == 0:
                window_mad = np.std(window_vals)
                if window_mad == 0:
                    continue

            # Check for outlier
            if np.abs(values[i] - window_median) > threshold * window_mad:
                filtered[i] = window_median

        if is_series:
            return pd.Series(filtered, index=index)
        else:
            return filtered

    def hodrick_prescott_filter(self, data, lambda_param=1600):
        """
        Apply a Hodrick-Prescott filter to separate trend and cyclical components.

        Parameters:
        -----------
        data : array-like
            Input time series data
        lambda_param : float
            Smoothing parameter

        Returns:
        --------
        tuple
            (trend, cyclical) components
        """
        # Convert to numpy array if needed
        if isinstance(data, pd.Series):
            values = data.values
            is_series = True
            index = data.index
        else:
            values = np.array(data)
            is_series = False

        # Apply the HP filter
        trend, cyclical = signal.hpfilter(values, lambda_param)

        if is_series:
            return pd.Series(trend, index=index), pd.Series(cyclical, index=index)
        else:
            return trend, cyclical

    def fourier_filter(self, data, cutoff_freq=0.1, keep_high=False):
        """
        Apply a Fourier transform filter.

        Parameters:
        -----------
        data : array-like
            Input time series data
        cutoff_freq : float
            Cutoff frequency as a fraction of the Nyquist frequency
        keep_high : bool
            If True, keep high frequencies (act as high-pass filter)
            If False, keep low frequencies (act as low-pass filter)

        Returns:
        --------
        array-like
            Filtered data
        """
        # Convert to numpy array if needed
        if isinstance(data, pd.Series):
            values = data.values
            is_series = True
            index = data.index
        else:
            values = np.array(data)
            is_series = False

        # Calculate the Fourier transform
        fft = np.fft.fft(values)

        # Frequency domain
        n = len(values)
        freq = np.fft.fftfreq(n)

        # Create a mask for frequencies
        if keep_high:
            mask = np.abs(freq) >= cutoff_freq
        else:
            mask = np.abs(freq) <= cutoff_freq

        # Apply the mask
        fft_filtered = fft * mask

        # Inverse Fourier transform
        filtered = np.real(np.fft.ifft(fft_filtered))

        if is_series:
            return pd.Series(filtered, index=index)
        else:
            return filtered

    def filter_ohlcv_data(self, df, filter_type='savitzky_golay', **kwargs):
        """
        Apply filtering to OHLCV data.

        Parameters:
        -----------
        df : pd.DataFrame
            DataFrame with OHLCV data
        filter_type : str
            Type of filter to apply
        **kwargs : dict
            Additional arguments for the specific filter

        Returns:
        --------
        pd.DataFrame
            Filtered DataFrame
        """
        if df is None or df.empty:
            logger.warning("Empty DataFrame provided to filter_ohlcv_data")
            return df

        # Make a copy to avoid modifying the original
        df_filtered = df.copy()

        # Check for required columns
        price_cols = ['open', 'high', 'low', 'close']
        missing_cols = [col for col in price_cols if col not in df_filtered.columns]

        if missing_cols:
            logger.warning(f"Missing price columns for filtering: {missing_cols}")
            # Filter only available columns
            price_cols = [col for col in price_cols if col in df_filtered.columns]

        if not price_cols:
            logger.warning("No price columns available for filtering")
            return df_filtered

        try:
            # Apply the specified filter to each price column
            for col in price_cols:
                if filter_type == 'moving_average':
                    df_filtered[f"{col}_filtered"] = self.moving_average_filter(df_filtered[col], **kwargs)
                elif filter_type == 'exponential_moving_average':
                    df_filtered[f"{col}_filtered"] = self.exponential_moving_average_filter(df_filtered[col],
                                                                                            **kwargs)
                elif filter_type == 'low_pass':
                    df_filtered[f"{col}_filtered"] = self.low_pass_filter(df_filtered[col], **kwargs)
                elif filter_type == 'high_pass':
                    df_filtered[f"{col}_filtered"] = self.high_pass_filter(df_filtered[col], **kwargs)
                elif filter_type == 'band_pass':
                    df_filtered[f"{col}_filtered"] = self.band_pass_filter(df_filtered[col], **kwargs)
                elif filter_type == 'kalman':
                    df_filtered[f"{col}_filtered"] = self.kalman_filter(df_filtered[col], **kwargs)
                elif filter_type == 'savitzky_golay':
                    df_filtered[f"{col}_filtered"] = self.savitzky_golay_filter(df_filtered[col], **kwargs)
                elif filter_type == 'median':
                    df_filtered[f"{col}_filtered"] = self.median_filter(df_filtered[col], **kwargs)
                elif filter_type == 'wavelet':
                    df_filtered[f"{col}_filtered"] = self.wavelet_filter(df_filtered[col], **kwargs)
                elif filter_type == 'hampel':
                    df_filtered[f"{col}_filtered"] = self.hampel_filter(df_filtered[col], **kwargs)
                elif filter_type == 'fourier':
                    df_filtered[f"{col}_filtered"] = self.fourier_filter(df_filtered[col], **kwargs)
                elif filter_type == 'hodrick_prescott':
                    trend, cyclical = self.hodrick_prescott_filter(df_filtered[col], **kwargs)
                    df_filtered[f"{col}_trend"] = trend
                    df_filtered[f"{col}_cyclical"] = cyclical
                else:
                    logger.warning(f"Unknown filter type: {filter_type}")
                    return df_filtered

            logger.info(f"Applied {filter_type} filter to {len(price_cols)} price columns")
            return df_filtered

        except Exception as e:
            logger.error(f"Error applying filter to OHLCV data: {e}")
            return df_filtered

    def filter_volatility(self, df, volatility_col='realized_volatility', filter_type='savitzky_golay', **kwargs):
        """
        Apply filtering to volatility data.

        Parameters:
        -----------
        df : pd.DataFrame
            DataFrame with volatility data
        volatility_col : str
            Column name for volatility data
        filter_type : str
            Type of filter to apply
        **kwargs : dict
            Additional arguments for the specific filter

        Returns:
        --------
        pd.DataFrame
            Filtered DataFrame
        """
        if df is None or df.empty:
            logger.warning("Empty DataFrame provided to filter_volatility")
            return df

        # Make a copy to avoid modifying the original
        df_filtered = df.copy()

        # Check for required column
        if volatility_col not in df_filtered.columns:
            logger.warning(f"Volatility column '{volatility_col}' not found in DataFrame")
            return df_filtered

        try:
            # Apply the specified filter to the volatility column
            if filter_type == 'moving_average':
                df_filtered[f"{volatility_col}_filtered"] = self.moving_average_filter(df_filtered[volatility_col],
                                                                                       **kwargs)
            elif filter_type == 'exponential_moving_average':
                df_filtered[f"{volatility_col}_filtered"] = self.exponential_moving_average_filter(
                    df_filtered[volatility_col], **kwargs)
            elif filter_type == 'low_pass':
                df_filtered[f"{volatility_col}_filtered"] = self.low_pass_filter(df_filtered[volatility_col],
                                                                                 **kwargs)
            elif filter_type == 'high_pass':
                df_filtered[f"{volatility_col}_filtered"] = self.high_pass_filter(df_filtered[volatility_col],
                                                                                  **kwargs)
            elif filter_type == 'band_pass':
                df_filtered[f"{volatility_col}_filtered"] = self.band_pass_filter(df_filtered[volatility_col],
                                                                                  **kwargs)
            elif filter_type == 'kalman':
                df_filtered[f"{volatility_col}_filtered"] = self.kalman_filter(df_filtered[volatility_col],
                                                                               **kwargs)
            elif filter_type == 'savitzky_golay':
                df_filtered[f"{volatility_col}_filtered"] = self.savitzky_golay_filter(df_filtered[volatility_col],
                                                                                       **kwargs)
            elif filter_type == 'median':
                df_filtered[f"{volatility_col}_filtered"] = self.median_filter(df_filtered[volatility_col],
                                                                               **kwargs)
            elif filter_type == 'wavelet':
                df_filtered[f"{volatility_col}_filtered"] = self.wavelet_filter(df_filtered[volatility_col],
                                                                                **kwargs)
            elif filter_type == 'hampel':
                df_filtered[f"{volatility_col}_filtered"] = self.hampel_filter(df_filtered[volatility_col],
                                                                               **kwargs)
            elif filter_type == 'fourier':
                df_filtered[f"{volatility_col}_filtered"] = self.fourier_filter(df_filtered[volatility_col],
                                                                                **kwargs)
            elif filter_type == 'hodrick_prescott':
                trend, cyclical = self.hodrick_prescott_filter(df_filtered[volatility_col], **kwargs)
                df_filtered[f"{volatility_col}_trend"] = trend
                df_filtered[f"{volatility_col}_cyclical"] = cyclical
            else:
                logger.warning(f"Unknown filter type: {filter_type}")
                return df_filtered

            logger.info(f"Applied {filter_type} filter to volatility column")
            return df_filtered

        except Exception as e:
            logger.error(f"Error applying filter to volatility data: {e}")
            return df_filtered

    def decompose_time_series(self, data, method='stl', period=None):
        """
        Decompose time series into trend, seasonal, and residual components.

        Parameters:
        -----------
        data : pd.Series
            Input time series data
        method : str
            Decomposition method ('stl' or 'seasonal_decompose')
        period : int, optional
            Period for seasonal component

        Returns:
        --------
        tuple
            (trend, seasonal, residual) components
        """
        if not isinstance(data, pd.Series):
            logger.warning("decompose_time_series requires a pandas Series")
            return None, None, None

        try:
            from statsmodels.tsa.seasonal import seasonal_decompose, STL

            # Ensure the series has a datetime index
            if not isinstance(data.index, pd.DatetimeIndex):
                logger.warning("Time series does not have a datetime index")
                # Try to convert to daily frequency
                if period is None:
                    period = 7  # Default to weekly seasonality
            else:
                # Infer period from frequency if not provided
                if period is None:
                    if data.index.freq == 'D':
                        period = 7  # Weekly seasonality
                    elif data.index.freq == 'M':
                        period = 12  # Annual seasonality
                    elif data.index.freq == 'Q':
                        period = 4  # Annual seasonality
                    elif data.index.freq == 'B':
                        period = 5  # Weekly business days
                    elif data.index.freq == 'H':
                        period = 24  # Daily seasonality
                    else:
                        # Default to 7 if frequency is not recognized
                        period = 7

            # Apply the decomposition
            if method == 'stl':
                stl = STL(data, period=period)
                result = stl.fit()

                trend = result.trend
                seasonal = result.seasonal
                residual = result.resid

            elif method == 'seasonal_decompose':
                result = seasonal_decompose(data, period=period)

                trend = result.trend
                seasonal = result.seasonal
                residual = result.resid

            else:
                logger.warning(f"Unknown decomposition method: {method}")
                return None, None, None

            logger.info(f"Decomposed time series using {method}")
            return trend, seasonal, residual

        except Exception as e:
            logger.error(f"Error decomposing time series: {e}")
            return None, None, None

# Factory function to get a data filter
def get_data_filter():
    """
    Get a data filter instance.

    Returns:
    --------
    DataFilter
        Data filter instance
    """
    return DataFilter()