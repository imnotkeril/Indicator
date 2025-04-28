"""
Module for detecting market regimes.
Identifies different market regimes based on price, volatility, and volume data.
"""
import os
import sys
import numpy as np
import pandas as pd
from datetime import datetime
import logging
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from scipy.stats import linregress
from scipy.signal import argrelextrema

from crypto_volatility_indicator.utils.logger import get_logger
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, project_root)
# Set up logger
logger = get_logger(__name__)


class RegimeDetector:
    """
    Detector for market regimes.

    This class identifies different market regimes (accumulation, trend,
    extreme volatility, consolidation) based on price, volatility,
    and other market data.
    """

    def __init__(self, n_regimes=4, window_size=20):
        """
        Initialize the regime detector.

        Parameters:
        -----------
        n_regimes : int
            Number of regimes to detect
        window_size : int
            Window size for feature calculation
        """
        self.n_regimes = n_regimes
        self.window_size = window_size
        self.regime_model = None
        self.scaler = StandardScaler()
        self.regime_names = {
            0: 'accumulation',  # Low volatility, sideways
            1: 'trend',  # Directional with moderate volatility
            2: 'extreme_volatility',  # High volatility
            3: 'consolidation'  # Decreasing volatility after a move
        }
        if n_regimes == 3:
            self.regime_names = {
                0: 'low_volatility',
                1: 'trend',
                2: 'high_volatility'
            }

        logger.info(f"RegimeDetector initialized with {n_regimes} regimes")

    def calculate_features(self, df, price_col='close', vol_col=None, return_col=None):
        """
        Calculate features for regime detection.

        Parameters:
        -----------
        df : pd.DataFrame
            DataFrame with price, volatility, and return data
        price_col : str
            Column name for price data
        vol_col : str, optional
            Column name for volatility data
        return_col : str, optional
            Column name for return data

        Returns:
        --------
        pd.DataFrame
            DataFrame with features for regime detection
        """
        if df is None or df.empty:
            logger.warning("Empty DataFrame provided to calculate_features")
            return df

        # Make a copy to avoid modifying the original
        df_features = df.copy()

        # Check if required columns exist
        if price_col not in df_features.columns:
            logger.warning(f"Price column '{price_col}' not found in DataFrame")
            return df_features

        # Calculate log returns if not provided
        if return_col is None:
            df_features['log_return'] = np.log(df_features[price_col] / df_features[price_col].shift(1))
            return_col = 'log_return'
        elif return_col not in df_features.columns:
            logger.warning(f"Return column '{return_col}' not found in DataFrame")
            return df_features

        # Calculate volatility if not provided
        if vol_col is None or vol_col not in df_features.columns:
            df_features['volatility'] = df_features[return_col].rolling(window=self.window_size).std()
            vol_col = 'volatility'

        try:
            # Feature 1: Volatility normalized by its recent history
            df_features['vol_normalized'] = df_features[vol_col] / df_features[vol_col].rolling(window=50).mean()

            # Feature 2: Directional strength (R-squared of linear regression on price)
            df_features['r_squared'] = 0.0

            for i in range(self.window_size, len(df_features)):
                y = df_features[price_col].iloc[i - self.window_size:i].values
                x = np.arange(self.window_size)
                _, _, r_value, _, _ = linregress(x, y)
                df_features.iloc[i, df_features.columns.get_loc('r_squared')] = r_value ** 2

            # Feature 3: Trend direction
            df_features['price_change'] = df_features[price_col].pct_change(self.window_size)

            # Feature 4: Recent volatility trend
            df_features['vol_change'] = df_features[vol_col].pct_change(self.window_size)

            # Feature 5: Price range relative to volatility
            df_features['range_percentage'] = (
                                                      df_features[price_col].rolling(window=self.window_size).max() -
                                                      df_features[price_col].rolling(window=self.window_size).min()
                                              ) / df_features[price_col].rolling(window=self.window_size).mean()

            # Feature 6: Count of local extrema (measure of choppiness)
            df_features['local_extrema_count'] = 0

            for i in range(self.window_size * 2, len(df_features)):
                # Get window of price data
                window = df_features[price_col].iloc[i - self.window_size:i].values

                # Find local maxima and minima
                max_idx = argrelextrema(window, np.greater, order=3)[0]
                min_idx = argrelextrema(window, np.less, order=3)[0]

                # Count extrema
                extrema_count = len(max_idx) + len(min_idx)
                df_features.iloc[i, df_features.columns.get_loc('local_extrema_count')] = extrema_count

            # Fill NaN values
            for col in ['vol_normalized', 'r_squared', 'price_change', 'vol_change', 'range_percentage',
                        'local_extrema_count']:
                df_features[col] = df_features[col].fillna(method='bfill').fillna(0)

            logger.info("Calculated features for regime detection")
            return df_features

        except Exception as e:
            logger.error(f"Error calculating features: {e}")
            return df_features

    def fit(self, df, features=None):
        """
        Fit the regime detection model.

        Parameters:
        -----------
        df : pd.DataFrame
            DataFrame with features
        features : list, optional
            List of feature columns to use

        Returns:
        --------
        bool
            True if successful, False otherwise
        """
        if df is None or df.empty:
            logger.warning("Empty DataFrame provided to fit")
            return False

        # Default features
        if features is None:
            features = ['vol_normalized', 'r_squared', 'price_change', 'vol_change', 'range_percentage',
                        'local_extrema_count']

        # Check if all features exist
        missing_features = [f for f in features if f not in df.columns]
        if missing_features:
            logger.warning(f"Missing features: {missing_features}")
            # Use only available features
            features = [f for f in features if f in df.columns]

        if not features:
            logger.warning("No features available for regime detection")
            return False

        try:
            # Prepare features
            X = df[features].dropna().values

            if len(X) < self.n_regimes * 2:
                logger.warning(f"Not enough data points ({len(X)}) for {self.n_regimes} regimes")
                return False

            # Scale features
            X_scaled = self.scaler.fit_transform(X)

            # Apply KMeans clustering
            kmeans = KMeans(n_clusters=self.n_regimes, random_state=42, n_init=10)
            labels = kmeans.fit_predict(X_scaled)

            # Store the model
            self.regime_model = kmeans

            # Analyze regimes
            self._analyze_regimes(df, features, labels)

            logger.info(f"Fitted regime detection model with {len(X)} data points")
            return True

        except Exception as e:
            logger.error(f"Error fitting regime model: {e}")
            return False

    def _analyze_regimes(self, df, features, labels):
        """
        Analyze the detected regimes to assign meaningful labels.

        Parameters:
        -----------
        df : pd.DataFrame
            DataFrame with features
        features : list
            List of feature columns used
        labels : array-like
            Cluster labels from KMeans
        """
        # Create a DataFrame with features and labels
        features_df = df[features].copy()
        features_df['cluster'] = pd.Series(labels, index=features_df.index)

        # Calculate mean feature values for each cluster
        cluster_means = features_df.groupby('cluster').mean()

        # Define characteristics of each regime
        if 'vol_normalized' in features and 'r_squared' in features:
            # Sort clusters by volatility
            volatility_order = cluster_means['vol_normalized'].sort_values().index

            if self.n_regimes == 4:
                # Assign regimes based on volatility and trend strength
                regime_mapping = {}

                # Lowest volatility -> Accumulation
                regime_mapping[volatility_order[0]] = 0

                # Highest volatility -> Extreme Volatility
                regime_mapping[volatility_order[-1]] = 2

                # For the middle two, check trend strength
                mid_clusters = [volatility_order[1], volatility_order[2]]
                r_squared_values = cluster_means.loc[mid_clusters, 'r_squared']

                # Higher R² -> Trend, Lower R² -> Consolidation
                if r_squared_values.iloc[0] > r_squared_values.iloc[1]:
                    regime_mapping[mid_clusters[0]] = 1  # Trend
                    regime_mapping[mid_clusters[1]] = 3  # Consolidation
                else:
                    regime_mapping[mid_clusters[0]] = 3  # Consolidation
                    regime_mapping[mid_clusters[1]] = 1  # Trend

                self.regime_mapping = regime_mapping

            elif self.n_regimes == 3:
                # Simpler mapping for 3 regimes
                regime_mapping = {
                    volatility_order[0]: 0,  # Low Volatility
                    volatility_order[1]: 1,  # Trend
                    volatility_order[2]: 2,  # High Volatility
                }
                self.regime_mapping = regime_mapping

            else:
                # Default: just map by volatility
                self.regime_mapping = {i: i for i in range(self.n_regimes)}

    def predict(self, df, features=None):
        """
        Predict regimes for new data.

        Parameters:
        -----------
        df : pd.DataFrame
            DataFrame with features
        features : list, optional
            List of feature columns to use

        Returns:
        --------
        pd.DataFrame
            DataFrame with regime predictions
        """
        if df is None or df.empty:
            logger.warning("Empty DataFrame provided to predict")
            return df

        if self.regime_model is None:
            logger.warning("Model not fitted. Call fit() first.")
            return df

        # Default features
        if features is None:
            features = ['vol_normalized', 'r_squared', 'price_change', 'vol_change', 'range_percentage',
                        'local_extrema_count']

        # Check if all features exist
        missing_features = [f for f in features if f not in df.columns]
        if missing_features:
            logger.warning(f"Missing features: {missing_features}")
            # Use only available features
            features = [f for f in features if f in df.columns]

        if not features:
            logger.warning("No features available for regime prediction")
            return df

        try:
            # Make a copy to avoid modifying the original
            df_pred = df.copy()

            # Prepare features
            X = df_pred[features].fillna(0).values

            # Scale features
            X_scaled = self.scaler.transform(X)

            # Predict clusters
            clusters = self.regime_model.predict(X_scaled)

            # Map clusters to regimes
            if hasattr(self, 'regime_mapping'):
                regimes = np.array([self.regime_mapping[c] for c in clusters])
            else:
                regimes = clusters

            # Add predictions to DataFrame
            df_pred['regime'] = regimes

            # Add regime names
            df_pred['regime_name'] = df_pred['regime'].map(self.regime_names)

            logger.info(f"Predicted regimes for {len(df_pred)} data points")
            return df_pred

        except Exception as e:
            logger.error(f"Error predicting regimes: {e}")
            return df

    def fit_predict(self, df, price_col='close', vol_col=None, return_col=None, features=None):
        """
        Calculate features, fit the model, and predict regimes.

        Parameters:
        -----------
        df : pd.DataFrame
            DataFrame with price, volatility, and return data
        price_col : str
            Column name for price data
        vol_col : str, optional
            Column name for volatility data
        return_col : str, optional
            Column name for return data
        features : list, optional
            List of feature columns to use

        Returns:
        --------
        pd.DataFrame
            DataFrame with regime predictions
        """
        # Calculate features
        df_features = self.calculate_features(df, price_col, vol_col, return_col)

        # Fit the model
        success = self.fit(df_features, features)

        if not success:
            logger.warning("Failed to fit regime model")
            return df_features

        # Predict regimes
        df_pred = self.predict(df_features, features)

        return df_pred

    def analyze_transitions(self, df, regime_col='regime'):
        """
        Analyze transitions between regimes.

        Parameters:
        -----------
        df : pd.DataFrame
            DataFrame with regime predictions
        regime_col : str
            Column name for regime labels

        Returns:
        --------
        dict
            Transition analysis
        """
        if df is None or df.empty:
            logger.warning("Empty DataFrame provided to analyze_transitions")
            return {}

        if regime_col not in df.columns:
            logger.warning(f"Regime column '{regime_col}' not found in DataFrame")
            return {}

        try:
            # Count regime occurrences
            regime_counts = df[regime_col].value_counts().to_dict()

            # Calculate regime durations
            regime_durations = {}

            for regime in range(self.n_regimes):
                # Create mask for the regime
                mask = df[regime_col] == regime

                if not mask.any():
                    continue

                # Find runs of the same regime
                runs = np.diff(np.where(np.concatenate(([mask.iloc[0]],
                                                        mask.iloc[:-1] != mask.iloc[1:],
                                                        [True])))[0])

                # Filter runs for the specific regime
                regime_runs = [run for i, run in enumerate(runs) if (i % 2 == 0) == mask.iloc[0]]

                if regime_runs:
                    regime_durations[regime] = {
                        'mean': np.mean(regime_runs),
                        'median': np.median(regime_runs),
                        'max': np.max(regime_runs),
                        'min': np.min(regime_runs)
                    }

            # Calculate transition probabilities
            transitions = {}

            for i in range(self.n_regimes):
                transitions[i] = {}

                # Count transitions from regime i to all other regimes
                mask_i = df[regime_col] == i

                if not mask_i.any():
                    continue

                indices = np.where(mask_i)[0]

                for idx in indices:
                    if idx + 1 < len(df):
                        next_regime = df[regime_col].iloc[idx + 1]

                        if next_regime not in transitions[i]:
                            transitions[i][next_regime] = 0

                        transitions[i][next_regime] += 1

                # Convert counts to probabilities
                total = sum(transitions[i].values())

                if total > 0:
                    transitions[i] = {k: v / total for k, v in transitions[i].items()}

            # Create transition matrix
            transition_matrix = np.zeros((self.n_regimes, self.n_regimes))

            for i in range(self.n_regimes):
                for j in range(self.n_regimes):
                    transition_matrix[i, j] = transitions.get(i, {}).get(j, 0)

            # Calculate regime persistence (probability of staying in the same regime)
            persistence = {i: transitions.get(i, {}).get(i, 0) for i in range(self.n_regimes)}

            # Named metrics
            named_counts = {self.regime_names[k]: v for k, v in regime_counts.items()}
            named_durations = {self.regime_names[k]: v for k, v in regime_durations.items()}

            # Named transition matrix
            named_transitions = {}
            for i in range(self.n_regimes):
                named_transitions[self.regime_names[i]] = {
                    self.regime_names[j]: transition_matrix[i, j]
                    for j in range(self.n_regimes)
                }

            analysis = {
                'regime_counts': regime_counts,
                'named_counts': named_counts,
                'regime_durations': regime_durations,
                'named_durations': named_durations,
                'transition_matrix': transition_matrix.tolist(),
                'named_transitions': named_transitions,
                'persistence': persistence,
                'current_regime': int(df[regime_col].iloc[-1]) if len(df) > 0 else None,
                'current_regime_name': self.regime_names[df[regime_col].iloc[-1]] if len(df) > 0 else None
            }

            logger.info(f"Analyzed regime transitions for {self.n_regimes} regimes")
            return analysis

        except Exception as e:
            logger.error(f"Error analyzing regime transitions: {e}")
            return {}

    def calculate_regime_statistics(self, df, regime_col='regime', price_col='close', vol_col=None,
                                    return_col=None):
        """
        Calculate statistics for each regime.

        Parameters:
        -----------
        df : pd.DataFrame
            DataFrame with regime predictions
        regime_col : str
            Column name for regime labels
        price_col : str
            Column name for price data
        vol_col : str, optional
            Column name for volatility data
        return_col : str, optional
            Column name for return data

        Returns:
        --------
        dict
            Statistics for each regime
        """
        if df is None or df.empty:
            logger.warning("Empty DataFrame provided to calculate_regime_statistics")
            return {}

        if regime_col not in df.columns:
            logger.warning(f"Regime column '{regime_col}' not found in DataFrame")
            return {}

        if price_col not in df.columns:
            logger.warning(f"Price column '{price_col}' not found in DataFrame")
            return {}

        # Determine return column
        if return_col is None:
            if 'log_return' in df.columns:
                return_col = 'log_return'
            else:
                df['return'] = df[price_col].pct_change()
                return_col = 'return'
        elif return_col not in df.columns:
            logger.warning(f"Return column '{return_col}' not found in DataFrame")
            return {}

        # Determine volatility column
        if vol_col is None:
            if 'volatility' in df.columns:
                vol_col = 'volatility'
            else:
                df['volatility'] = df[return_col].rolling(window=20).std()
                vol_col = 'volatility'
        elif vol_col not in df.columns:
            logger.warning(f"Volatility column '{vol_col}' not found in DataFrame")
            return {}

        try:
            # Group by regime
            grouped = df.groupby(regime_col)

            # Calculate statistics for each regime
            stats = {}

            for regime, group in grouped:
                if len(group) == 0:
                    continue

                # Calculate return statistics
                returns = group[return_col].dropna()

                # Calculate price movement
                first_price = group[price_col].iloc[0]
                last_price = group[price_col].iloc[-1]
                price_change = (last_price / first_price - 1) * 100

                # Calculate volatility statistics
                volatility = group[vol_col].dropna()

                # Calculate success metrics
                if len(returns) > 0:
                    win_rate = (returns > 0).mean()
                    profit_factor = abs(returns[returns > 0].sum() / returns[returns < 0].sum()) if returns[
                                                                                                        returns < 0].sum() != 0 else float(
                        'inf')
                    sharpe_ratio = returns.mean() / returns.std() if returns.std() > 0 else 0
                else:
                    win_rate = None
                    profit_factor = None
                    sharpe_ratio = None

                stats[regime] = {
                    'count': len(group),
                    'duration': {
                        'mean': None,  # Will be filled by analyze_transitions
                        'median': None,
                        'max': None,
                        'min': None
                    },
                    'returns': {
                        'mean': float(returns.mean()) if len(returns) > 0 else None,
                        'std': float(returns.std()) if len(returns) > 0 else None,
                        'min': float(returns.min()) if len(returns) > 0 else None,
                        'max': float(returns.max()) if len(returns) > 0 else None,
                        'cumulative': float(np.exp(returns.sum()) - 1) if len(returns) > 0 else None
                    },
                    'price': {
                        'first': float(first_price),
                        'last': float(last_price),
                        'change_pct': float(price_change)
                    },
                    'volatility': {
                        'mean': float(volatility.mean()) if len(volatility) > 0 else None,
                        'std': float(volatility.std()) if len(volatility) > 0 else None,
                        'min': float(volatility.min()) if len(volatility) > 0 else None,
                        'max': float(volatility.max()) if len(volatility) > 0 else None
                    },
                    'performance': {
                        'win_rate': float(win_rate) if win_rate is not None else None,
                        'profit_factor': float(profit_factor) if profit_factor is not None else None,
                        'sharpe_ratio': float(sharpe_ratio) if sharpe_ratio is not None else None
                    }
                }

            # Add named statistics
            named_stats = {self.regime_names[k]: v for k, v in stats.items()}

            # Add summary statistics
            summary = {
                'regime_count': len(stats),
                'most_common_regime': max(stats.keys(), key=lambda k: stats[k]['count']) if stats else None,
                'most_common_regime_name': self.regime_names[
                    max(stats.keys(), key=lambda k: stats[k]['count'])] if stats else None,
                'highest_volatility_regime': max(stats.keys(), key=lambda k: stats[k]['volatility'][
                                                                                 'mean'] or 0) if stats else None,
                'highest_volatility_regime_name': self.regime_names[
                    max(stats.keys(), key=lambda k: stats[k]['volatility']['mean'] or 0)] if stats else None,
                'highest_return_regime': max(stats.keys(), key=lambda k: stats[k]['returns'][
                                                                             'cumulative'] or 0) if stats else None,
                'highest_return_regime_name': self.regime_names[
                    max(stats.keys(), key=lambda k: stats[k]['returns']['cumulative'] or 0)] if stats else None
            }

            result = {
                'regime_statistics': stats,
                'named_statistics': named_stats,
                'summary': summary
            }

            logger.info(f"Calculated statistics for {len(stats)} regimes")
            return result

        except Exception as e:
            logger.error(f"Error calculating regime statistics: {e}")
            return {}

    def visualize_regimes(self, df, regime_col='regime', price_col='close', vol_col=None, show_plot=True):
        """
        Visualize regimes with price and volatility.

        Parameters:
        -----------
        df : pd.DataFrame
            DataFrame with regime predictions
        regime_col : str
            Column name for regime labels
        price_col : str
            Column name for price data
        vol_col : str, optional
            Column name for volatility data
        show_plot : bool
            Whether to display the plot or return the figure

        Returns:
        --------
        matplotlib.figure.Figure or None
            Figure object if show_plot is False, None otherwise
        """
        if df is None or df.empty:
            logger.warning("Empty DataFrame provided to visualize_regimes")
            return None

        if regime_col not in df.columns:
            logger.warning(f"Regime column '{regime_col}' not found in DataFrame")
            return None

        if price_col not in df.columns:
            logger.warning(f"Price column '{price_col}' not found in DataFrame")
            return None

        # Determine volatility column
        if vol_col is None:
            if 'volatility' in df.columns:
                vol_col = 'volatility'
            else:
                return_col = 'log_return' if 'log_return' in df.columns else 'return'
                if return_col in df.columns:
                    df['volatility'] = df[return_col].rolling(window=20).std()
                    vol_col = 'volatility'

        try:
            import matplotlib.pyplot as plt
            import matplotlib.patches as mpatches
            from matplotlib.colors import LinearSegmentedColormap

            # Create figure
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 10), sharex=True,
                                           gridspec_kw={'height_ratios': [3, 1]})

            # Plot price
            ax1.plot(df.index, df[price_col], color='black', linewidth=1.5, label='Price')
            ax1.set_title('Price with Market Regimes')
            ax1.set_ylabel('Price')
            ax1.legend(loc='upper left')
            ax1.grid(True, alpha=0.3)

            # Plot volatility if available
            if vol_col in df.columns:
                ax3 = ax1.twinx()
                ax3.plot(df.index, df[vol_col], color='blue', alpha=0.5, linewidth=1, label='Volatility')
                ax3.set_ylabel('Volatility')
                ax3.legend(loc='upper right')

            # Plot regimes
            unique_regimes = df[regime_col].unique()

            # Define colors for different regimes
            colors = ['green', 'blue', 'red', 'orange']

            # Ensure we have enough colors
            if len(unique_regimes) > len(colors):
                colors = plt.cm.tab10.colors[:len(unique_regimes)]

            # Create regime patches for legend
            regime_patches = []

            # Plot each regime
            for regime in unique_regimes:
                regime_mask = df[regime_col] == regime
                regime_periods = []
                start_idx = None

                # Find contiguous periods of the same regime
                for i in range(len(df)):
                    if regime_mask.iloc[i]:
                        if start_idx is None:
                            start_idx = i
                    elif start_idx is not None:
                        regime_periods.append((start_idx, i - 1))
                        start_idx = None

                # Add the last period if it extends to the end
                if start_idx is not None:
                    regime_periods.append((start_idx, len(df) - 1))

                # Shade each period
                for start, end in regime_periods:
                    ax1.axvspan(df.index[start], df.index[end], alpha=0.2, color=colors[regime % len(colors)])

                # Create patch for legend
                regime_patches.append(mpatches.Patch(
                    color=colors[regime % len(colors)],
                    alpha=0.2,
                    label=self.regime_names.get(regime, f'Regime {regime}')
                ))

            # Add regime legend
            ax1.legend(handles=regime_patches, loc='upper center', bbox_to_anchor=(0.5, -0.05),
                       ncol=len(unique_regimes))

            # Plot regime timeline
            c_dict = {regime: colors[regime % len(colors)] for regime in unique_regimes}
            colors_list = [c_dict[regime] for regime in df[regime_col]]

            # Create a colormap from the colors
            cmap = LinearSegmentedColormap.from_list('regime_cmap', colors_list, N=len(colors_list))

            # Plot regimes as a colored bar
            regime_bar = ax2.imshow([df[regime_col].values], aspect='auto', cmap=cmap,
                                    extent=[0, len(df), 0, 1])

            # Configure the regime bar axis
            ax2.set_yticks([])
            ax2.set_title('Market Regimes')

            # Set x-axis ticks
            ax2.set_xticks(range(0, len(df), len(df) // 10))
            ax2.set_xticklabels([df.index[i].strftime('%Y-%m-%d') for i in range(0, len(df), len(df) // 10)])

            plt.tight_layout()

            if show_plot:
                plt.show()
                return None
            else:
                return fig

        except Exception as e:
            logger.error(f"Error visualizing regimes: {e}")
            return None

    def run_analysis(self, df, price_col='close', vol_col=None, return_col=None):
        """
        Run a complete regime analysis.

        Parameters:
        -----------
        df : pd.DataFrame
            DataFrame with price, volatility, and return data
        price_col : str
            Column name for price data
        vol_col : str, optional
            Column name for volatility data
        return_col : str, optional
            Column name for return data

        Returns:
        --------
        tuple
            (DataFrame with regime predictions, dict with analysis results)
        """
        # Fit and predict regimes
        df_regimes = self.fit_predict(df, price_col, vol_col, return_col)

        # Analyze transitions
        transition_analysis = self.analyze_transitions(df_regimes)

        # Calculate statistics
        statistics = self.calculate_regime_statistics(df_regimes, price_col=price_col, vol_col=vol_col,
                                                      return_col=return_col)

        # Visualize regimes
        self.visualize_regimes(df_regimes, price_col=price_col, vol_col=vol_col)

        # Combine results
        analysis = {
            'transitions': transition_analysis,
            'statistics': statistics,
            'current_regime': transition_analysis.get('current_regime'),
            'current_regime_name': transition_analysis.get('current_regime_name')
        }

        logger.info(f"Completed regime analysis with {self.n_regimes} regimes")
        return df_regimes, analysis

    # Factory function to get a regime detector
    def get_regime_detector(n_regimes=4, window_size=20):
        """
        Get a configured regime detector.

        Parameters:
        -----------
        n_regimes : int
            Number of regimes to detect
        window_size : int
            Window size for feature calculation

        Returns:
        --------
        RegimeDetector
            Configured detector instance
        """
        return RegimeDetector(n_regimes=n_regimes, window_size=window_size)