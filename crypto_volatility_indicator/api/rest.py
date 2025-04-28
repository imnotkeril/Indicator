"""
REST API implementation for the volatility indicator.
"""
import os
import sys
from flask import Flask, jsonify, request, abort
from flask_cors import CORS
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import json
import threading
import time
import logging
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, project_root)
from crypto_volatility_indicator.utils.logger import get_logger

logger = get_logger(__name__)


class VolatilityAPI:
    """
    REST API for the volatility indicator.

    Provides endpoints for retrieving volatility data, regime information,
    predictions, and trading signals.
    """

    def __init__(self, volatility_indicator, config=None):
        """
        Initialize the API.

        Parameters:
        -----------
        volatility_indicator : object
            The volatility indicator instance
        config : dict, optional
            Configuration parameters
        """
        self.volatility_indicator = volatility_indicator
        self.config = config or {}

        # Configure API settings
        self.host = self.config.get('host', '0.0.0.0')
        self.port = self.config.get('port', 5000)
        self.debug = self.config.get('debug', False)
        self.cors_origins = self.config.get('cors_origins', '*')

        # API endpoints config
        self.api_prefix = self.config.get('api_prefix', '/api/v1')

        # Initialize Flask app
        self.app = Flask(__name__)

        # Enable CORS
        CORS(self.app, resources={f"{self.api_prefix}/*": {"origins": self.cors_origins}})

        # Data cache
        self.data_cache = {}
        self.last_update = {}
        self.cache_ttl = self.config.get('cache_ttl', 60)  # seconds

        # Register routes
        self._register_routes()

        # Background update thread
        self.update_thread_active = False
        self.update_thread = None

    def _register_routes(self):
        """Register API routes."""

        # Health check endpoint
        @self.app.route(f"{self.api_prefix}/health", methods=['GET'])
        def health_check():
            return jsonify({
                "status": "ok",
                "timestamp": datetime.now().isoformat(),
                "version": self.config.get('version', '1.0.0')
            })

        # Get available assets
        @self.app.route(f"{self.api_prefix}/assets", methods=['GET'])
        def get_assets():
            try:
                assets = self.volatility_indicator.get_monitored_assets()
                return jsonify({
                    "status": "success",
                    "data": {
                        "assets": assets
                    },
                    "timestamp": datetime.now().isoformat()
                })
            except Exception as e:
                logger.error(f"Error getting assets: {str(e)}")
                return jsonify({
                    "status": "error",
                    "message": str(e),
                    "timestamp": datetime.now().isoformat()
                }), 500

        # Get price data
        @self.app.route(f"{self.api_prefix}/prices/<asset>", methods=['GET'])
        def get_prices(asset):
            try:
                # Parse query parameters
                start_date = request.args.get('start_date', None)
                end_date = request.args.get('end_date', None)

                if start_date:
                    start_date = datetime.fromisoformat(start_date.replace('Z', '+00:00'))

                if end_date:
                    end_date = datetime.fromisoformat(end_date.replace('Z', '+00:00'))

                # Get data (from cache if available and fresh)
                data = self._get_data(
                    asset=asset,
                    data_type='price_data',
                    start_date=start_date,
                    end_date=end_date
                )

                if data is None:
                    return jsonify({
                        "status": "error",
                        "message": f"No price data available for {asset}",
                        "timestamp": datetime.now().isoformat()
                    }), 404

                # Convert to JSON-friendly format
                json_data = data.reset_index().to_dict(orient='records')

                return jsonify({
                    "status": "success",
                    "data": {
                        "asset": asset,
                        "prices": json_data
                    },
                    "timestamp": datetime.now().isoformat()
                })

            except Exception as e:
                logger.error(f"Error getting price data: {str(e)}")
                return jsonify({
                    "status": "error",
                    "message": str(e),
                    "timestamp": datetime.now().isoformat()
                }), 500

        # Get volatility data
        @self.app.route(f"{self.api_prefix}/volatility/<asset>", methods=['GET'])
        def get_volatility(asset):
            try:
                # Parse query parameters
                start_date = request.args.get('start_date', None)
                end_date = request.args.get('end_date', None)

                if start_date:
                    start_date = datetime.fromisoformat(start_date.replace('Z', '+00:00'))

                if end_date:
                    end_date = datetime.fromisoformat(end_date.replace('Z', '+00:00'))

                # Get data (from cache if available and fresh)
                data = self._get_data(
                    asset=asset,
                    data_type='vol_data',
                    start_date=start_date,
                    end_date=end_date
                )

                if data is None:
                    return jsonify({
                        "status": "error",
                        "message": f"No volatility data available for {asset}",
                        "timestamp": datetime.now().isoformat()
                    }), 404

                # Convert to JSON-friendly format
                json_data = data.reset_index().to_dict(orient='records')

                return jsonify({
                    "status": "success",
                    "data": {
                        "asset": asset,
                        "volatility": json_data
                    },
                    "timestamp": datetime.now().isoformat()
                })

            except Exception as e:
                logger.error(f"Error getting volatility data: {str(e)}")
                return jsonify({
                    "status": "error",
                    "message": str(e),
                    "timestamp": datetime.now().isoformat()
                }), 500

        # Get regime data
        @self.app.route(f"{self.api_prefix}/regimes/<asset>", methods=['GET'])
        def get_regimes(asset):
            try:
                # Parse query parameters
                start_date = request.args.get('start_date', None)
                end_date = request.args.get('end_date', None)

                if start_date:
                    start_date = datetime.fromisoformat(start_date.replace('Z', '+00:00'))

                if end_date:
                    end_date = datetime.fromisoformat(end_date.replace('Z', '+00:00'))

                # Get data (from cache if available and fresh)
                data = self._get_data(
                    asset=asset,
                    data_type='regime_data',
                    start_date=start_date,
                    end_date=end_date
                )

                if data is None:
                    return jsonify({
                        "status": "error",
                        "message": f"No regime data available for {asset}",
                        "timestamp": datetime.now().isoformat()
                    }), 404

                # Convert to JSON-friendly format
                json_data = data.reset_index().to_dict(orient='records')

                return jsonify({
                    "status": "success",
                    "data": {
                        "asset": asset,
                        "regimes": json_data
                    },
                    "timestamp": datetime.now().isoformat()
                })

            except Exception as e:
                logger.error(f"Error getting regime data: {str(e)}")
                return jsonify({
                    "status": "error",
                    "message": str(e),
                    "timestamp": datetime.now().isoformat()
                }), 500

        # Get predictions
        @self.app.route(f"{self.api_prefix}/predictions/<asset>", methods=['GET'])
        def get_predictions(asset):
            try:
                # Parse query parameters
                start_date = request.args.get('start_date', None)
                end_date = request.args.get('end_date', None)

                if start_date:
                    start_date = datetime.fromisoformat(start_date.replace('Z', '+00:00'))

                if end_date:
                    end_date = datetime.fromisoformat(end_date.replace('Z', '+00:00'))

                # Get data (from cache if available and fresh)
                data = self._get_data(
                    asset=asset,
                    data_type='prediction_data',
                    start_date=start_date,
                    end_date=end_date
                )

                if data is None:
                    return jsonify({
                        "status": "error",
                        "message": f"No prediction data available for {asset}",
                        "timestamp": datetime.now().isoformat()
                    }), 404

                # Convert to JSON-friendly format
                if isinstance(data, pd.DataFrame):
                    json_data = data.reset_index().to_dict(orient='records')
                else:  # Series
                    json_data = data.reset_index().to_frame('predicted_vol').to_dict(orient='records')

                return jsonify({
                    "status": "success",
                    "data": {
                        "asset": asset,
                        "predictions": json_data
                    },
                    "timestamp": datetime.now().isoformat()
                })

            except Exception as e:
                logger.error(f"Error getting prediction data: {str(e)}")
                return jsonify({
                    "status": "error",
                    "message": str(e),
                    "timestamp": datetime.now().isoformat()
                }), 500

        # Get signals
        @self.app.route(f"{self.api_prefix}/signals/<asset>", methods=['GET'])
        def get_signals(asset):
            try:
                # Parse query parameters
                start_date = request.args.get('start_date', None)
                end_date = request.args.get('end_date', None)

                if start_date:
                    start_date = datetime.fromisoformat(start_date.replace('Z', '+00:00'))

                if end_date:
                    end_date = datetime.fromisoformat(end_date.replace('Z', '+00:00'))

                # Get data (from cache if available and fresh)
                data = self._get_data(
                    asset=asset,
                    data_type='signal_data',
                    start_date=start_date,
                    end_date=end_date
                )

                if data is None:
                    return jsonify({
                        "status": "error",
                        "message": f"No signal data available for {asset}",
                        "timestamp": datetime.now().isoformat()
                    }), 404

                # Convert to JSON-friendly format
                json_data = data.reset_index().to_dict(orient='records')

                return jsonify({
                    "status": "success",
                    "data": {
                        "asset": asset,
                        "signals": json_data
                    },
                    "timestamp": datetime.now().isoformat()
                })

            except Exception as e:
                logger.error(f"Error getting signal data: {str(e)}")
                return jsonify({
                    "status": "error",
                    "message": str(e),
                    "timestamp": datetime.now().isoformat()
                }), 500

        # Get stop loss and take profit levels
        @self.app.route(f"{self.api_prefix}/risk/<asset>", methods=['GET'])
        def get_risk_levels(asset):
            try:
                # Parse query parameters
                position_type = request.args.get('position_type', 'long')

                # Get latest price data
                price_data = self._get_data(
                    asset=asset,
                    data_type='price_data',
                    limit=1
                )

                # Get latest volatility data
                vol_data = self._get_data(
                    asset=asset,
                    data_type='vol_data',
                    limit=1
                )

                if price_data is None or vol_data is None:
                    return jsonify({
                        "status": "error",
                        "message": f"Insufficient data available for {asset}",
                        "timestamp": datetime.now().isoformat()
                    }), 404

                # Calculate stop loss and take profit levels
                stop_loss = self.volatility_indicator.calculate_stop_loss(
                    price_data=price_data,
                    volatility_data=vol_data,
                    position_type=position_type
                )

                take_profit = self.volatility_indicator.calculate_take_profit(
                    price_data=price_data,
                    volatility_data=vol_data,
                    position_type=position_type
                )

                # Get current price
                current_price = price_data['close'].iloc[-1]

                # Get stop loss and take profit prices
                stop_loss_price = stop_loss.iloc[-1]
                take_profit_price = take_profit.iloc[-1]

                # Calculate distances as percentages
                if position_type == 'long':
                    stop_loss_pct = (current_price - stop_loss_price) / current_price * 100
                    take_profit_pct = (take_profit_price - current_price) / current_price * 100
                else:  # short position
                    stop_loss_pct = (stop_loss_price - current_price) / current_price * 100
                    take_profit_pct = (current_price - take_profit_price) / current_price * 100

                return jsonify({
                    "status": "success",
                    "data": {
                        "asset": asset,
                        "position_type": position_type,
                        "current_price": current_price,
                        "stop_loss": {
                            "price": stop_loss_price,
                            "distance_pct": stop_loss_pct
                        },
                        "take_profit": {
                            "price": take_profit_price,
                            "distance_pct": take_profit_pct
                        },
                        "risk_reward_ratio": take_profit_pct / stop_loss_pct if stop_loss_pct > 0 else None
                    },
                    "timestamp": datetime.now().isoformat()
                })

            except Exception as e:
                logger.error(f"Error getting risk levels: {str(e)}")
                return jsonify({
                    "status": "error",
                    "message": str(e),
                    "timestamp": datetime.now().isoformat()
                }), 500

        # Get composite data (all in one)
        @self.app.route(f"{self.api_prefix}/composite/<asset>", methods=['GET'])
        def get_composite_data(asset):
            try:
                # Parse query parameters
                start_date = request.args.get('start_date', None)
                end_date = request.args.get('end_date', None)

                if start_date:
                    start_date = datetime.fromisoformat(start_date.replace('Z', '+00:00'))

                if end_date:
                    end_date = datetime.fromisoformat(end_date.replace('Z', '+00:00'))

                # Get all data types
                price_data = self._get_data(
                    asset=asset,
                    data_type='price_data',
                    start_date=start_date,
                    end_date=end_date
                )

                vol_data = self._get_data(
                    asset=asset,
                    data_type='vol_data',
                    start_date=start_date,
                    end_date=end_date
                )

                regime_data = self._get_data(
                    asset=asset,
                    data_type='regime_data',
                    start_date=start_date,
                    end_date=end_date
                )

                signal_data = self._get_data(
                    asset=asset,
                    data_type='signal_data',
                    start_date=start_date,
                    end_date=end_date
                )

                prediction_data = self._get_data(
                    asset=asset,
                    data_type='prediction_data',
                    start_date=start_date,
                    end_date=end_date
                )

                # Check if we have at least price and volatility data
                if price_data is None or vol_data is None:
                    return jsonify({
                        "status": "error",
                        "message": f"Insufficient data available for {asset}",
                        "timestamp": datetime.now().isoformat()
                    }), 404

                # Create a composite DataFrame
                # Start with price data
                composite = price_data.copy()

                # Add volatility data
                for col in vol_data.columns:
                    composite[col] = vol_data[col]

                # Add regime data if available
                if regime_data is not None and 'regime' in regime_data.columns:
                    composite['regime'] = regime_data['regime']

                # Add signal data if available
                if signal_data is not None:
                    for col in signal_data.columns:
                        if col not in composite.columns:  # Avoid duplicating columns like 'close'
                            composite[col] = signal_data[col]

                # Add prediction data if available
                if prediction_data is not None:
                    if isinstance(prediction_data, pd.DataFrame):
                        pred_col = prediction_data.columns[0]
                        composite['predicted_vol'] = prediction_data[pred_col]
                    else:  # Series
                        composite['predicted_vol'] = prediction_data

                # Convert to JSON-friendly format
                json_data = composite.reset_index().to_dict(orient='records')

                return jsonify({
                    "status": "success",
                    "data": {
                        "asset": asset,
                        "composite": json_data
                    },
                    "timestamp": datetime.now().isoformat()
                })

            except Exception as e:
                logger.error(f"Error getting composite data: {str(e)}")
                return jsonify({
                    "status": "error",
                    "message": str(e),
                    "timestamp": datetime.now().isoformat()
                }), 500

    def _get_data(self, asset, data_type, start_date=None, end_date=None, limit=None):
        """
        Get data from cache or from the indicator.

        Parameters:
        -----------
        asset : str
            Asset symbol
        data_type : str
            Type of data to retrieve ('price_data', 'vol_data', etc.)
        start_date : datetime, optional
            Start date for filtering data
        end_date : datetime, optional
            End date for filtering data
        limit : int, optional
            Limit to the last N data points

        Returns:
        --------
        pd.DataFrame or pd.Series or None
            The requested data, or None if not available
        """
        # Check if we have cached data that's still fresh
        cache_key = f"{asset}_{data_type}"

        if cache_key in self.data_cache and cache_key in self.last_update:
            cache_age = (datetime.now() - self.last_update[cache_key]).total_seconds()

            if cache_age < self.cache_ttl:
                # Use cached data
                data = self.data_cache[cache_key]
            else:
                # Cache is stale, fetch fresh data
                data = self._fetch_data_from_indicator(asset, data_type)

                # Update cache
                if data is not None:
                    self.data_cache[cache_key] = data
                    self.last_update[cache_key] = datetime.now()
        else:
            # No cache entry, fetch fresh data
            data = self._fetch_data_from_indicator(asset, data_type)

            # Update cache
            if data is not None:
                self.data_cache[cache_key] = data
                self.last_update[cache_key] = datetime.now()

        # Handle case where we have no data
        if data is None:
            return None

        # Apply filters
        if limit is not None:
            data = data.iloc[-limit:]

        if start_date is not None:
            data = data[data.index >= start_date]

        if end_date is not None:
            data = data[data.index <= end_date]

        return data

    def _fetch_data_from_indicator(self, asset, data_type):
        """
        Fetch data directly from the indicator.

        Parameters:
        -----------
        asset : str
            Asset symbol
        data_type : str
            Type of data to retrieve

        Returns:
        --------
        pd.DataFrame or pd.Series or None
            The requested data, or None if not available
        """
        try:
            if data_type == 'price_data':
                return self.volatility_indicator.get_price_data(asset)
            elif data_type == 'vol_data':
                return self.volatility_indicator.get_volatility_data(asset)
            elif data_type == 'regime_data':
                return self.volatility_indicator.get_regime_data(asset)
            elif data_type == 'signal_data':
                return self.volatility_indicator.get_signal_data(asset)
            elif data_type == 'prediction_data':
                return self.volatility_indicator.get_predictions(asset)
            else:
                logger.error(f"Unknown data type: {data_type}")
                return None
        except Exception as e:
            logger.error(f"Error fetching {data_type} for {asset}: {str(e)}")
            return None

    def start_background_data_update(self, interval=60):
        """
        Start a background thread to periodically update the cached data.

        Parameters:
        -----------
        interval : int
            Update interval in seconds
        """
        if self.update_thread_active:
            logger.warning("Background update thread is already running")
            return

        self.update_thread_active = True
        self.update_thread = threading.Thread(
            target=self._background_update_loop,
            args=(interval,),
            daemon=True
        )
        self.update_thread.start()

        logger.info(f"Started background data update thread with interval {interval}s")

    def stop_background_data_update(self):
        """Stop the background data update thread."""
        if not self.update_thread_active:
            logger.warning("Background update thread is not running")
            return

        self.update_thread_active = False

        if self.update_thread and self.update_thread.is_alive():
            self.update_thread.join(timeout=5)

        logger.info("Stopped background data update thread")

    def _background_update_loop(self, interval):
        """
        Background loop to update cached data.

        Parameters:
        -----------
        interval : int
            Update interval in seconds
        """
        while self.update_thread_active:
            try:
                # Get all monitored assets
                assets = self.volatility_indicator.get_monitored_assets()

                for asset in assets:
                    # Update all data types for each asset
                    data_types = ['price_data', 'vol_data', 'regime_data', 'signal_data', 'prediction_data']

                    for data_type in data_types:
                        try:
                            data = self._fetch_data_from_indicator(asset, data_type)

                            if data is not None:
                                cache_key = f"{asset}_{data_type}"
                                self.data_cache[cache_key] = data
                                self.last_update[cache_key] = datetime.now()
                        except Exception as e:
                            logger.error(f"Error updating {data_type} for {asset}: {str(e)}")

                logger.debug(f"Updated cached data for {len(assets)} assets")

            except Exception as e:
                logger.error(f"Error in background update loop: {str(e)}")

            # Sleep before next update
            time.sleep(interval)

    def run(self):
        """Run the API server."""
        logger.info(f"Starting API server on {self.host}:{self.port}")

        # Start background data update thread
        self.start_background_data_update()

        # Run Flask app
        self.app.run(host=self.host, port=self.port, debug=self.debug, threaded=True)