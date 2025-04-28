"""
Module for collecting blockchain metrics.
Handles connection to blockchain data providers and fetches on-chain data.
"""
import os
import sys
import requests
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import logging
import time
import json
from pathlib import Path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, project_root)
from crypto_volatility_indicator.utils.config import config
from crypto_volatility_indicator.utils.logger import get_data_logger
from crypto_volatility_indicator.utils.helpers import (
    ensure_directory,
    save_dataframe,
    load_dataframe,
    create_cache_key
)

# Set up logger
logger = get_data_logger()


class BlockchainDataCollector:
    """
    Collector for blockchain metrics data.

    This class handles connections to blockchain data providers 
    and fetches on-chain metrics for analysis.
    """

    def __init__(self, api_keys=None):
        """
        Initialize the blockchain data collector.

        Parameters:
        -----------
        api_keys : dict, optional
            Dictionary of API keys for different blockchain data providers
            If None, uses the values from config
        """
        # Load API keys from config if not provided
        self.api_keys = api_keys or {
            'glassnode': config.get('blockchain.api_keys.glassnode', ''),
            'coinmetrics': config.get('blockchain.api_keys.coinmetrics', ''),
            'blockchair': config.get('blockchain.api_keys.blockchair', '')
        }

        # Supported assets
        self.supported_assets = ['BTC', 'ETH', 'SOL', 'BNB']

        # Cache directory
        self.cache_dir = Path('data/cache/blockchain')
        ensure_directory(self.cache_dir)

        # Data storage
        self.data = {}

        logger.info("BlockchainDataCollector initialized")

    def fetch_glassnode_metric(self, asset, metric, since=None, until=None, resolution='24h'):
        """
        Fetch a metric from Glassnode API.

        Parameters:
        -----------
        asset : str
            Asset symbol (e.g., 'BTC', 'ETH')
        metric : str
            Metric name (refer to Glassnode API docs)
        since : datetime, optional
            Start time for data fetching
        until : datetime, optional
            End time for data fetching
        resolution : str, optional
            Data resolution ('10m', '1h', '24h', etc.)

        Returns:
        --------
        pd.DataFrame
            DataFrame with the requested metric
        """
        if not self.api_keys.get('glassnode'):
            logger.warning("No Glassnode API key configured")
            return None

        # Base URL for Glassnode API
        base_url = "https://api.glassnode.com/v1/metrics"

        # Convert dates to UNIX timestamps if provided
        params = {
            'a': asset,
            'api_key': self.api_keys['glassnode'],
            'i': resolution
        }

        if since:
            params['s'] = int(since.timestamp())

        if until:
            params['u'] = int(until.timestamp())

        # Construct URL
        url = f"{base_url}/{metric}"

        try:
            logger.info(f"Fetching Glassnode metric: {metric} for {asset}")

            # Make API request
            response = requests.get(url, params=params)

            if response.status_code != 200:
                logger.error(f"Failed to fetch Glassnode metric: {response.status_code} - {response.text}")
                return None

            # Parse response
            data = response.json()

            if not data:
                logger.warning(f"No data returned for {metric} ({asset})")
                return None

            # Convert to DataFrame
            df = pd.DataFrame(data)

            # Convert timestamp to datetime
            df['timestamp'] = pd.to_datetime(df['t'], unit='s')
            df.set_index('timestamp', inplace=True)

            # Drop original timestamp column
            if 't' in df.columns:
                df.drop('t', axis=1, inplace=True)

            # Rename value column to metric name
            if 'v' in df.columns:
                df.rename(columns={'v': metric}, inplace=True)

            logger.info(f"Fetched {len(df)} datapoints for {metric} ({asset})")
            return df

        except Exception as e:
            logger.error(f"Error fetching Glassnode metric: {e}")
            return None

    def fetch_coinmetrics_metric(self, asset, metric, since=None, until=None):
        """
        Fetch a metric from CoinMetrics API.

        Parameters:
        -----------
        asset : str
            Asset symbol (e.g., 'BTC', 'ETH')
        metric : str
            Metric name (refer to CoinMetrics API docs)
        since : datetime, optional
            Start time for data fetching
        until : datetime, optional
            End time for data fetching

        Returns:
        --------
        pd.DataFrame
            DataFrame with the requested metric
        """
        if not self.api_keys.get('coinmetrics'):
            logger.warning("No CoinMetrics API key configured")
            return None

        # Base URL for CoinMetrics API
        base_url = "https://api.coinmetrics.io/v4"

        # Format dates
        params = {
            'api_key': self.api_keys['coinmetrics']
        }

        if since:
            params['start_time'] = since.strftime('%Y-%m-%dT%H:%M:%SZ')

        if until:
            params['end_time'] = until.strftime('%Y-%m-%dT%H:%M:%SZ')

        # Map asset symbol to CoinMetrics format if needed
        asset_id = asset.lower()

        # Construct URL
        url = f"{base_url}/timeseries/asset-metrics"
        params['assets'] = asset_id
        params['metrics'] = metric

        try:
            logger.info(f"Fetching CoinMetrics metric: {metric} for {asset}")

            # Make API request
            response = requests.get(url, params=params)

            if response.status_code != 200:
                logger.error(f"Failed to fetch CoinMetrics metric: {response.status_code} - {response.text}")
                return None

            # Parse response
            data = response.json()

            if 'data' not in data or not data['data']:
                logger.warning(f"No data returned for {metric} ({asset})")
                return None

            # Convert to DataFrame
            df = pd.DataFrame(data['data'])

            # Convert time to datetime
            df['timestamp'] = pd.to_datetime(df['time'])
            df.set_index('timestamp', inplace=True)

            # Drop original time column
            if 'time' in df.columns:
                df.drop('time', axis=1, inplace=True)

            # Rename metric column
            if metric in df.columns:
                df.rename(columns={metric: metric.lower()}, inplace=True)

            logger.info(f"Fetched {len(df)} datapoints for {metric} ({asset})")
            return df

        except Exception as e:
            logger.error(f"Error fetching CoinMetrics metric: {e}")
            return None

    def fetch_blockchair_stats(self, blockchain):
        """
        Fetch blockchain stats from Blockchair API.

        Parameters:
        -----------
        blockchain : str
            Blockchain name (e.g., 'bitcoin', 'ethereum')

        Returns:
        --------
        dict
            Dictionary with blockchain stats
        """
        if not self.api_keys.get('blockchair'):
            logger.warning("No Blockchair API key configured")
            return None

        # Base URL for Blockchair API
        base_url = "https://api.blockchair.com"

        # Construct URL
        url = f"{base_url}/{blockchain}/stats"

        params = {
            'key': self.api_keys['blockchair']
        }

        try:
            logger.info(f"Fetching Blockchair stats for {blockchain}")

            # Make API request
            response = requests.get(url, params=params)

            if response.status_code != 200:
                logger.error(f"Failed to fetch Blockchair stats: {response.status_code} - {response.text}")
                return None

            # Parse response
            data = response.json()

            if 'data' not in data:
                logger.warning(f"No data returned for {blockchain}")
                return None

            logger.info(f"Fetched Blockchair stats for {blockchain}")
            return data['data']

        except Exception as e:
            logger.error(f"Error fetching Blockchair stats: {e}")
            return None

    def fetch_network_metrics(self, asset, since=None, until=None):
        """
        Fetch a set of network metrics for an asset.

        Parameters:
        -----------
        asset : str
            Asset symbol (e.g., 'BTC', 'ETH')
        since : datetime, optional
            Start time for data fetching
        until : datetime, optional
            End time for data fetching

        Returns:
        --------
        dict
            Dictionary with DataFrames for different metrics
        """
        if asset not in self.supported_assets:
            logger.warning(f"Asset not supported: {asset}")
            return None

        # Default date range if not specified
        if not since:
            since = datetime.now() - timedelta(days=30)

        if not until:
            until = datetime.now()

        metrics = {}

        # Map asset to blockchain name for Blockchair
        blockchain_map = {
            'BTC': 'bitcoin',
            'ETH': 'ethereum',
            'SOL': None,  # Not supported by Blockchair
            'BNB': None  # Not supported by Blockchair
        }

        # Fetch metrics from Glassnode
        glassnode_metrics_map = {
            'BTC': [
                'transactions_count',
                'difficulty',
                'hashrate',
                'sopr',
                'active_addresses'
            ],
            'ETH': [
                'transactions_count',
                'gas_used',
                'active_addresses',
                'eth2_staking_total_volume'
            ],
            'SOL': [
                'transactions_count',
                'active_addresses'
            ],
            'BNB': [
                'transactions_count',
                'active_addresses'
            ]
        }

        for metric in glassnode_metrics_map.get(asset, []):
            df = self.fetch_glassnode_metric(asset, metric, since, until)
            if df is not None:
                metrics[metric] = df

        # Fetch metrics from CoinMetrics
        coinmetrics_metrics_map = {
            'BTC': [
                'TxCnt',
                'HashRate',
                'AdrActCnt',
                'FeeTotNtv',
                'RevNtv'
            ],
            'ETH': [
                'TxCnt',
                'GasUsed',
                'AdrActCnt',
                'FeeTotNtv',
                'RevNtv'
            ],
            'SOL': [
                'TxCnt',
                'AdrActCnt',
                'FeeTotNtv'
            ],
            'BNB': [
                'TxCnt',
                'AdrActCnt',
                'FeeTotNtv'
            ]
        }

        for metric in coinmetrics_metrics_map.get(asset, []):
            df = self.fetch_coinmetrics_metric(asset, metric, since, until)
            if df is not None:
                # Use lowercase metric name for consistency
                metrics[metric.lower()] = df

        # Fetch Blockchair stats if available
        blockchain = blockchain_map.get(asset)
        if blockchain:
            stats = self.fetch_blockchair_stats(blockchain)
            if stats:
                # Convert to DataFrame with current timestamp
                stats_df = pd.DataFrame([stats], index=[pd.Timestamp.now()])
                metrics['blockchair_stats'] = stats_df

        if not metrics:
            logger.warning(f"No metrics fetched for {asset}")
            return None

        logger.info(f"Fetched {len(metrics)} network metrics for {asset}")
        return metrics

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
        df_vwkama['volume_z'] = (df_vwkama[volume_col] - df_vwkama[volume_col].rolling(window=20).mean()) / df_vwkama[
            volume_col].rolling(window=20).std()
        df_vwkama['volume_z'] = df_vwkama['volume_z'].fillna(0)

        # Calculate volume factor (higher volume = faster adaptation)
        df_vwkama['volume_factor'] = np.clip(1.0 + 0.5 * df_vwkama['volume_z'], 0.5, 2.0)

        # Calculate price change
        df_vwkama['price_change'] = abs(df_vwkama[price_col] - df_vwkama[price_col].shift(er_period))

        # Calculate volatility (sum of absolute price changes)
        df_vwkama['volatility'] = abs(df_vwkama[price_col].diff()).rolling(window=er_period).sum()

        # Calculate efficiency ratio
        df_vwkama['efficiency_ratio'] = df_vwkama['price_change'] / df_vwkama['volatility']
        df_vwkama['efficiency_ratio'] = df_vwkama['efficiency_ratio'].replace([np.inf, -np.inf], np.nan).fillna(0)

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

        logger.info(f"Completed meso-volatility analysis with {metrics['volatility_regimes']['regime_count']} regimes")
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