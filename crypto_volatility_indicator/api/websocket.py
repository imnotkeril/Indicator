"""
WebSocket API implementation for the volatility indicator.
"""
import os
import sys
import asyncio
import websockets
import json
import pandas as pd
from datetime import datetime, timedelta
import logging
import threading
import time
import os
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, project_root)
from crypto_volatility_indicator.utils.logger import get_logger

logger = get_logger(__name__)


class VolatilityWebSocket:
    """
    WebSocket API for the volatility indicator.

    Provides real-time volatility data, regime changes, and trading signals
    via WebSocket connections.
    """

    def __init__(self, volatility_indicator, config=None):
        """
        Initialize the WebSocket API.

        Parameters:
        -----------
        volatility_indicator : object
            The volatility indicator instance
        config : dict, optional
            Configuration parameters
        """
        self.volatility_indicator = volatility_indicator
        self.config = config or {}

        # Configure WebSocket settings
        self.host = self.config.get('host', '0.0.0.0')
        self.port = self.config.get('port', 8765)

        # Active connections and their subscriptions
        self.connections = {}
        self.subscriptions = {}

        # Data cache
        self.data_cache = {}
        self.last_update = {}

        # Initialize server
        self.server = None
        self.server_task = None

        # Lock for thread safety
        self.lock = asyncio.Lock()

        # Thread for event loop
        self.thread = None
        self.loop = None
        self.running = False

    async def handle_connection(self, websocket, path):
        """
        Handle a WebSocket connection.

        Parameters:
        -----------
        websocket : websockets.WebSocketServerProtocol
            The WebSocket connection
        path : str
            The connection path
        """
        # Register new connection
        connection_id = id(websocket)
        async with self.lock:
            self.connections[connection_id] = websocket
            self.subscriptions[connection_id] = set()

        logger.info(f"New connection: {connection_id}")

        try:
            # Send welcome message
            await websocket.send(json.dumps({
                "type": "welcome",
                "message": "Connected to Volatility Indicator WebSocket API",
                "timestamp": datetime.now().isoformat()
            }))

            # Handle messages
            async for message in websocket:
                try:
                    # Parse message as JSON
                    data = json.loads(message)

                    # Handle different message types
                    if 'action' in data:
                        if data['action'] == 'subscribe':
                            await self.handle_subscribe(connection_id, data)
                        elif data['action'] == 'unsubscribe':
                            await self.handle_unsubscribe(connection_id, data)
                        elif data['action'] == 'ping':
                            await self.handle_ping(connection_id)
                        elif data['action'] == 'get_data':
                            await self.handle_get_data(connection_id, data)
                        else:
                            await websocket.send(json.dumps({
                                "type": "error",
                                "message": f"Unknown action: {data['action']}",
                                "timestamp": datetime.now().isoformat()
                            }))
                    else:
                        await websocket.send(json.dumps({
                            "type": "error",
                            "message": "Invalid message format, 'action' field is required",
                            "timestamp": datetime.now().isoformat()
                        }))

                except json.JSONDecodeError:
                    await websocket.send(json.dumps({
                        "type": "error",
                        "message": "Invalid JSON format",
                        "timestamp": datetime.now().isoformat()
                    }))
                except Exception as e:
                    logger.error(f"Error handling message: {str(e)}")
                    await websocket.send(json.dumps({
                        "type": "error",
                        "message": f"Error processing request: {str(e)}",
                        "timestamp": datetime.now().isoformat()
                    }))

        except websockets.exceptions.ConnectionClosed:
            logger.info(f"Connection closed: {connection_id}")
        except Exception as e:
            logger.error(f"Connection error: {str(e)}")
        finally:
            # Clean up connection
            async with self.lock:
                if connection_id in self.connections:
                    del self.connections[connection_id]
                if connection_id in self.subscriptions:
                    del self.subscriptions[connection_id]

            logger.info(f"Connection removed: {connection_id}")

    async def handle_subscribe(self, connection_id, data):
        """
        Handle a subscription request.

        Parameters:
        -----------
        connection_id : int
            The connection ID
        data : dict
            The subscription request data
        """
        if 'channel' not in data:
            await self.send_error(connection_id, "Missing 'channel' field in subscription request")
            return

        if 'asset' not in data:
            await self.send_error(connection_id, "Missing 'asset' field in subscription request")
            return

        channel = data['channel']
        asset = data['asset']

        # Validate channel
        valid_channels = ['price', 'volatility', 'regime', 'signals', 'predictions', 'all']
        if channel not in valid_channels:
            await self.send_error(connection_id, f"Invalid channel: {channel}")
            return

        # Validate asset
        if not self.volatility_indicator.is_valid_asset(asset):
            await self.send_error(connection_id, f"Invalid asset: {asset}")
            return

        # Add subscription
        subscription = f"{channel}:{asset}"
        async with self.lock:
            if connection_id in self.subscriptions:
                self.subscriptions[connection_id].add(subscription)

        # Send confirmation
        await self.connections[connection_id].send(json.dumps({
            "type": "subscription",
            "status": "success",
            "channel": channel,
            "asset": asset,
            "message": f"Subscribed to {channel} updates for {asset}",
            "timestamp": datetime.now().isoformat()
        }))

        logger.info(f"Connection {connection_id} subscribed to {subscription}")

        # Send initial data for this subscription
        await self.send_initial_data(connection_id, channel, asset)

    async def handle_unsubscribe(self, connection_id, data):
        """
        Handle an unsubscription request.

        Parameters:
        -----------
        connection_id : int
            The connection ID
        data : dict
            The unsubscription request data
        """
        if 'channel' not in data:
            await self.send_error(connection_id, "Missing 'channel' field in unsubscription request")
            return

        if 'asset' not in data:
            await self.send_error(connection_id, "Missing 'asset' field in unsubscription request")
            return

        channel = data['channel']
        asset = data['asset']

        # Remove subscription
        subscription = f"{channel}:{asset}"
        async with self.lock:
            if connection_id in self.subscriptions:
                if subscription in self.subscriptions[connection_id]:
                    self.subscriptions[connection_id].remove(subscription)

        # Send confirmation
        await self.connections[connection_id].send(json.dumps({
            "type": "unsubscription",
            "status": "success",
            "channel": channel,
            "asset": asset,
            "message": f"Unsubscribed from {channel} updates for {asset}",
            "timestamp": datetime.now().isoformat()
        }))

        logger.info(f"Connection {connection_id} unsubscribed from {subscription}")

    async def handle_ping(self, connection_id):
        """
        Handle a ping request.

        Parameters:
        -----------
        connection_id : int
            The connection ID
        """
        await self.connections[connection_id].send(json.dumps({
            "type": "pong",
            "timestamp": datetime.now().isoformat()
        }))

    async def handle_get_data(self, connection_id, data):
        """
        Handle a data request.

        Parameters:
        -----------
        connection_id : int
            The connection ID
        data : dict
            The data request data
        """
        if 'channel' not in data:
            await self.send_error(connection_id, "Missing 'channel' field in data request")
            return

        if 'asset' not in data:
            await self.send_error(connection_id, "Missing 'asset' field in data request")
            return

        channel = data['channel']
        asset = data['asset']

        # Validate channel
        valid_channels = ['price', 'volatility', 'regime', 'signals', 'predictions', 'all']
        if channel not in valid_channels:
            await self.send_error(connection_id, f"Invalid channel: {channel}")
            return

        # Validate asset
        if not self.volatility_indicator.is_valid_asset(asset):
            await self.send_error(connection_id, f"Invalid asset: {asset}")
            return

        # Parse optional parameters
        start_date = None
        end_date = None
        limit = None

        if 'start_date' in data:
            try:
                start_date = datetime.fromisoformat(data['start_date'].replace('Z', '+00:00'))
            except ValueError:
                await self.send_error(connection_id, f"Invalid start_date format: {data['start_date']}")
                return

        if 'end_date' in data:
            try:
                end_date = datetime.fromisoformat(data['end_date'].replace('Z', '+00:00'))
            except ValueError:
                await self.send_error(connection_id, f"Invalid end_date format: {data['end_date']}")
                return

        if 'limit' in data:
            try:
                limit = int(data['limit'])
                if limit <= 0:
                    raise ValueError("Limit must be positive")
            except ValueError:
                await self.send_error(connection_id, f"Invalid limit value: {data['limit']}")
                return

        # Get and send data
        await self.send_channel_data(connection_id, channel, asset, start_date, end_date, limit)

    async def send_error(self, connection_id, message):
        """
        Send an error message to a connection.

        Parameters:
        -----------
        connection_id : int
            The connection ID
        message : str
            The error message
        """
        if connection_id in self.connections:
            await self.connections[connection_id].send(json.dumps({
                "type": "error",
                "message": message,
                "timestamp": datetime.now().isoformat()
            }))

    async def send_initial_data(self, connection_id, channel, asset):
        """
        Send initial data for a subscription.

        Parameters:
        -----------
        connection_id : int
            The connection ID
        channel : str
            The channel name
        asset : str
            The asset symbol
        """
        # Get last 100 data points for the subscription
        await self.send_channel_data(connection_id, channel, asset, limit=100)

    async def send_channel_data(self, connection_id, channel, asset, start_date=None, end_date=None, limit=None):
        """
        Send data for a specific channel and asset.

        Parameters:
        -----------
        connection_id : int
            The connection ID
        channel : str
            The channel name
        asset : str
            The asset symbol
        start_date : datetime, optional
            Start date for filtering data
        end_date : datetime, optional
            End date for filtering data
        limit : int, optional
            Limit to the last N data points
        """
        try:
            # Get data based on channel
            if channel == 'price' or channel == 'all':
                data = await self.get_price_data(asset, start_date, end_date, limit)
                if data is not None:
                    await self.send_data_message(connection_id, 'price', asset, data)

            if channel == 'volatility' or channel == 'all':
                data = await self.get_volatility_data(asset, start_date, end_date, limit)
                if data is not None:
                    await self.send_data_message(connection_id, 'volatility', asset, data)

            if channel == 'regime' or channel == 'all':
                data = await self.get_regime_data(asset, start_date, end_date, limit)
                if data is not None:
                    await self.send_data_message(connection_id, 'regime', asset, data)

            if channel == 'signals' or channel == 'all':
                data = await self.get_signal_data(asset, start_date, end_date, limit)
                if data is not None:
                    await self.send_data_message(connection_id, 'signals', asset, data)

            if channel == 'predictions' or channel == 'all':
                data = await self.get_prediction_data(asset, start_date, end_date, limit)
                if data is not None:
                    await self.send_data_message(connection_id, 'predictions', asset, data)

        except Exception as e:
            logger.error(f"Error sending channel data: {str(e)}")
            await self.send_error(connection_id, f"Error retrieving data: {str(e)}")

    async def send_data_message(self, connection_id, channel, asset, data):
        """
        Send a data message to a connection.

        Parameters:
        -----------
        connection_id : int
            The connection ID
        channel : str
            The channel name
        asset : str
            The asset symbol
        data : list
            The data to send
        """
        if connection_id in self.connections:
            await self.connections[connection_id].send(json.dumps({
                "type": "data",
                "channel": channel,
                "asset": asset,
                "data": data,
                "timestamp": datetime.now().isoformat()
            }))

    async def get_price_data(self, asset, start_date=None, end_date=None, limit=None):
        """
        Get price data for an asset.

        Parameters:
        -----------
        asset : str
            Asset symbol
        start_date : datetime, optional
            Start date for filtering data
        end_date : datetime, optional
            End date for filtering data
        limit : int, optional
            Limit to the last N data points

        Returns:
        --------
        list
            List of price data records
        """
        try:
            # Get data synchronously
            data = await asyncio.to_thread(
                self.volatility_indicator.get_price_data,
                asset, start_date, end_date
            )

            if data is None or data.empty:
                logger.warning(f"No price data available for {asset}")
                return []

            # Apply limit if specified
            if limit is not None:
                data = data.iloc[-limit:]

            # Convert to JSON-friendly format
            result = []
            for timestamp, row in data.iterrows():
                record = {
                    'timestamp': timestamp.isoformat(),
                    'close': row['close']
                }

                # Add additional columns if available
                if 'open' in row:
                    record['open'] = row['open']
                if 'high' in row:
                    record['high'] = row['high']
                if 'low' in row:
                    record['low'] = row['low']
                if 'volume' in row:
                    record['volume'] = row['volume']

                result.append(record)

            return result

        except Exception as e:
            logger.error(f"Error getting price data for {asset}: {str(e)}")
            return None

    async def get_volatility_data(self, asset, start_date=None, end_date=None, limit=None):
        """
        Get volatility data for an asset.

        Parameters:
        -----------
        asset : str
            Asset symbol
        start_date : datetime, optional
            Start date for filtering data
        end_date : datetime, optional
            End date for filtering data
        limit : int, optional
            Limit to the last N data points

        Returns:
        --------
        list
            List of volatility data records
        """
        try:
            # Get data synchronously
            data = await asyncio.to_thread(
                self.volatility_indicator.get_volatility_data,
                asset, start_date, end_date
            )

            if data is None or data.empty:
                logger.warning(f"No volatility data available for {asset}")
                return []

            # Apply limit if specified
            if limit is not None:
                data = data.iloc[-limit:]

            # Convert to JSON-friendly format
            result = []
            for timestamp, row in data.iterrows():
                record = {
                    'timestamp': timestamp.isoformat()
                }

                # Add all volatility columns
                for col, value in row.items():
                    record[col] = value

                result.append(record)

            return result

        except Exception as e:
            logger.error(f"Error getting volatility data for {asset}: {str(e)}")
            return None

    async def get_regime_data(self, asset, start_date=None, end_date=None, limit=None):
        """
        Get regime data for an asset.

        Parameters:
        -----------
        asset : str
            Asset symbol
        start_date : datetime, optional
            Start date for filtering data
        end_date : datetime, optional
            End date for filtering data
        limit : int, optional
            Limit to the last N data points

        Returns:
        --------
        list
            List of regime data records
        """
        try:
            # Get data synchronously
            data = await asyncio.to_thread(
                self.volatility_indicator.get_regime_data,
                asset, start_date, end_date
            )

            if data is None or data.empty:
                logger.warning(f"No regime data available for {asset}")
                return []

            # Apply limit if specified
            if limit is not None:
                data = data.iloc[-limit:]

            # Convert to JSON-friendly format
            result = []
            for timestamp, row in data.iterrows():
                record = {
                    'timestamp': timestamp.isoformat()
                }

                # Add regime column
                if 'regime' in row:
                    record['regime'] = row['regime']

                # Add any additional columns
                for col, value in row.items():
                    if col != 'regime':
                        record[col] = value

                result.append(record)

            return result

        except Exception as e:
            logger.error(f"Error getting regime data for {asset}: {str(e)}")
            return None

    async def get_signal_data(self, asset, start_date=None, end_date=None, limit=None):
        """
        Get signal data for an asset.

        Parameters:
        -----------
        asset : str
            Asset symbol
        start_date : datetime, optional
            Start date for filtering data
        end_date : datetime, optional
            End date for filtering data
        limit : int, optional
            Limit to the last N data points

        Returns:
        --------
        list
            List of signal data records
        """
        try:
            # Get data synchronously
            data = await asyncio.to_thread(
                self.volatility_indicator.get_signal_data,
                asset, start_date, end_date
            )

            if data is None or data.empty:
                logger.warning(f"No signal data available for {asset}")
                return []

            # Apply limit if specified
            if limit is not None:
                data = data.iloc[-limit:]

            # Convert to JSON-friendly format
            result = []
            for timestamp, row in data.iterrows():
                record = {
                    'timestamp': timestamp.isoformat()
                }

                # Add all signal columns
                for col, value in row.items():
                    record[col] = value

                result.append(record)

            return result

        except Exception as e:
            logger.error(f"Error getting signal data for {asset}: {str(e)}")
            return None

    async def get_prediction_data(self, asset, start_date=None, end_date=None, limit=None):
        """
        Get prediction data for an asset.

        Parameters:
        -----------
        asset : str
            Asset symbol
        start_date : datetime, optional
            Start date for filtering data
        end_date : datetime, optional
            End date for filtering data
        limit : int, optional
            Limit to the last N data points

        Returns:
        --------
        list
            List of prediction data records
        """
        try:
            # Get data synchronously
            data = await asyncio.to_thread(
                self.volatility_indicator.get_predictions,
                asset, start_date, end_date
            )

            if data is None or data.empty:
                logger.warning(f"No prediction data available for {asset}")
                return []

            # Apply limit if specified
            if limit is not None:
                data = data.iloc[-limit:]

            # Convert to JSON-friendly format
            result = []

            # Handle both DataFrame and Series formats
            if isinstance(data, pd.DataFrame):
                for timestamp, row in data.iterrows():
                    record = {
                        'timestamp': timestamp.isoformat()
                    }

                    # Add all prediction columns
                    for col, value in row.items():
                        record[col] = value

                    result.append(record)
            else:  # Series
                for timestamp, value in data.items():
                    result.append({
                        'timestamp': timestamp.isoformat(),
                        'predicted_vol': value
                    })

            return result

        except Exception as e:
            logger.error(f"Error getting prediction data for {asset}: {str(e)}")
            return None

    async def broadcast_updates(self):
        """Broadcast data updates to all subscribed connections."""
        while True:
            try:
                # Get all assets
                assets = self.volatility_indicator.get_monitored_assets()

                for asset in assets:
                    # Get latest data
                    price_data = await self.get_price_data(asset, limit=1)
                    vol_data = await self.get_volatility_data(asset, limit=1)
                    regime_data = await self.get_regime_data(asset, limit=1)
                    signal_data = await self.get_signal_data(asset, limit=1)
                    prediction_data = await self.get_prediction_data(asset, limit=1)

                    # Send updates to subscribed connections
                    async with self.lock:
                        for connection_id, subscriptions in self.subscriptions.items():
                            # Check each subscription
                            if f"price:{asset}" in subscriptions or f"all:{asset}" in subscriptions:
                                if price_data is not None:
                                    await self.send_data_message(connection_id, 'price', asset, price_data)

                            if f"volatility:{asset}" in subscriptions or f"all:{asset}" in subscriptions:
                                if vol_data is not None:
                                    await self.send_data_message(connection_id, 'volatility', asset, vol_data)

                            if f"regime:{asset}" in subscriptions or f"all:{asset}" in subscriptions:
                                if regime_data is not None:
                                    await self.send_data_message(connection_id, 'regime', asset, regime_data)

                            if f"signals:{asset}" in subscriptions or f"all:{asset}" in subscriptions:
                                if signal_data is not None:
                                    await self.send_data_message(connection_id, 'signals', asset, signal_data)

                            if f"predictions:{asset}" in subscriptions or f"all:{asset}" in subscriptions:
                                if prediction_data is not None:
                                    await self.send_data_message(connection_id, 'predictions', asset, prediction_data)

            except Exception as e:
                logger.error(f"Error broadcasting updates: {str(e)}")

            # Wait before next broadcast
            await asyncio.sleep(1)

    def run_in_thread(self):
        """Run the WebSocket server in a separate thread."""
        if self.thread is not None and self.thread.is_alive():
            logger.warning("WebSocket server is already running")
            return

        self.thread = threading.Thread(target=self._run_eventloop)
        self.thread.daemon = True
        self.thread.start()

        logger.info(f"WebSocket server started on {self.host}:{self.port}")

    def _run_eventloop(self):
        """Run the asyncio event loop."""
        self.loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self.loop)

        self.running = True

        try:
            self.loop.run_until_complete(self._run_server())
        except Exception as e:
            logger.error(f"WebSocket server error: {str(e)}")
        finally:
            self.loop.close()
            self.running = False

    async def _run_server(self):
        """Run the WebSocket server."""
        self.server = await websockets.serve(
            self.handle_connection,
            self.host,
            self.port
        )

        # Start broadcast task
        self.server_task = asyncio.create_task(self.broadcast_updates())

        # Keep the server running
        await self.server.wait_closed()

    def stop(self):
        """Stop the WebSocket server."""
        if not self.running:
            logger.warning("WebSocket server is not running")
            return

        if self.loop is not None and self.server is not None:
            async def stop_server():
                self.server.close()
                await self.server.wait_closed()
                if self.server_task is not None:
                    self.server_task.cancel()
                    try:
                        await self.server_task
                    except asyncio.CancelledError:
                        pass

            future = asyncio.run_coroutine_threadsafe(stop_server(), self.loop)
            future.result(timeout=5)

            self.running = False

            logger.info("WebSocket server stopped")

    def run(self):
        """Run the WebSocket server synchronously (blocking)."""
        logger.info(f"Starting WebSocket server on {self.host}:{self.port}")

        try:
            asyncio.run(self._run_server())
        except KeyboardInterrupt:
            logger.info("WebSocket server stopped by user")
        except Exception as e:
            logger.error(f"WebSocket server error: {str(e)}")
        finally:
            self.running = False