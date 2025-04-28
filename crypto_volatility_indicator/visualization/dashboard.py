"""
Dashboard implementation for visualizing volatility indicators.
"""
import os
import sys
import dash
from dash import dcc, html, callback
from dash.dependencies import Input, Output, State
import dash_bootstrap_components as dbc
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import threading
import time
import logging
import json
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, project_root)
from crypto_volatility_indicator.utils.logger import get_logger
from crypto_volatility_indicator.visualization.charts import VolatilityChartGenerator

logger = get_logger(__name__)

class VolatilityDashboard:
    """
    Dashboard for visualizing volatility indicators and analytics.
    """

    def __init__(self, volatility_indicator, config=None):
        """
        Initialize the dashboard.

        Parameters:
        -----------
        volatility_indicator : object
            The volatility indicator instance to visualize
        config : dict, optional
            Configuration parameters
        """
        self.volatility_indicator = volatility_indicator
        self.config = config or {}

        # Initialize chart generator
        self.chart_generator = VolatilityChartGenerator(self.config.get('charts', {}))

        # Configure dashboard settings
        self.title = self.config.get('title', 'Crypto Volatility Dashboard')
        self.theme = self.config.get('theme', 'darkly')
        self.port = self.config.get('port', 8050)
        self.host = self.config.get('host', '127.0.0.1')
        self.debug = self.config.get('debug', False)
        self.update_interval = self.config.get('update_interval', 60000)  # in milliseconds

        # Create Dash app
        self.app = dash.Dash(
            __name__,
            external_stylesheets=[dbc.themes.DARKLY if self.theme == 'darkly' else dbc.themes.BOOTSTRAP],
            title=self.title
        )

        # Configure layout
        self._configure_layout()

        # Configure callbacks
        self._configure_callbacks()

        # Data cache
        self.data_cache = {}
        self.last_update = {}

        # Background update thread
        self.update_thread_active = False
        self.update_thread = None

    def _configure_layout(self):
        """Configure the dashboard layout."""
        # Sidebar controls
        sidebar = html.Div(
            [
                html.H2("Controls", className="display-6 text-center mb-4"),
                html.Hr(),

                # Asset selection
                html.Div([
                    html.Label("Select Asset", className="form-label"),
                    dcc.Dropdown(
                        id="asset-dropdown",
                        options=[],
                        value=None,
                        className="mb-3"
                    )
                ]),

                # Time period selection
                html.Div([
                    html.Label("Time Period", className="form-label"),
                    dcc.RadioItems(
                        id="time-period",
                        options=[
                            {'label': '1 Day', 'value': '1d'},
                            {'label': '1 Week', 'value': '1w'},
                            {'label': '1 Month', 'value': '1m'},
                            {'label': '3 Months', 'value': '3m'},
                            {'label': 'All', 'value': 'all'}
                        ],
                        value='1w',
                        className="mb-3"
                    )
                ]),

                # Volatility component selection
                html.Div([
                    html.Label("Volatility Components", className="form-label"),
                    dcc.Checklist(
                        id="vol-components",
                        options=[
                            {'label': 'Micro Volatility', 'value': 'micro_vol'},
                            {'label': 'Meso Volatility', 'value': 'meso_vol'},
                            {'label': 'Macro Volatility', 'value': 'macro_vol'},
                            {'label': 'Composite Volatility', 'value': 'composite_vol'},
                            {'label': 'Predicted Volatility', 'value': 'predicted_vol'}
                        ],
                        value=['composite_vol', 'predicted_vol'],
                        className="mb-3"
                    )
                ]),

                # Signal overlay
                html.Div([
                    html.Label("Signal Overlay", className="form-label"),
                    dbc.Checklist(
                        id="signal-overlay",
                        options=[
                            {'label': 'Trading Signals', 'value': 'signals'},
                            {'label': 'Regime Changes', 'value': 'regimes'}
                        ],
                        value=['regimes'],
                        className="mb-3"
                    )
                ]),

                # Update button
                dbc.Button(
                    "Update Data",
                    id="update-button",
                    color="primary",
                    className="w-100 mb-4"
                ),

                # Auto-update toggle
                dbc.Switch(
                    id="auto-update-toggle",
                    label="Auto-update",
                    value=True,
                    className="mb-3"
                ),

                # Last update time
                html.Div(
                    id="last-update-text",
                    className="text-muted text-center small"
                )
            ],
            className="bg-light p-4 h-100 border-end"
        )

        # Main content area
        content = html.Div(
            [
                # Loading spinner for main chart
                dcc.Loading(
                    id="loading-main-chart",
                    type="circle",
                    children=[
                        # Main chart
                        dcc.Graph(
                            id="main-chart",
                            className="mb-4",
                            style={"height": "600px"}
                        )
                    ]
                ),

                # Stats cards row
                dbc.Row([
                    # Current volatility card
                    dbc.Col([
                        dbc.Card([
                            dbc.CardHeader("Current Volatility"),
                            dbc.CardBody([
                                html.H3(id="current-vol-value", className="card-title"),
                                html.P(id="current-vol-trend", className="card-text")
                            ])
                        ], className="h-100 bg-dark text-light")
                    ], width=3),

                    # Last signal card
                    dbc.Col([
                        dbc.Card([
                            dbc.CardHeader("Latest Signal"),
                            dbc.CardBody([
                                html.H3(id="latest-signal-value", className="card-title"),
                                html.P(id="signal-time", className="card-text")
                            ])
                        ], className="h-100 bg-dark text-light")
                    ], width=3)
                ], className="mb-4"),

                # Additional charts row
                dbc.Row([
                    # Regime distribution chart
                    dbc.Col([
                        dbc.Card([
                            dbc.CardHeader("Regime Distribution"),
                            dbc.CardBody([
                                dcc.Loading(
                                    id="loading-regime-chart",
                                    children=[
                                        dcc.Graph(
                                            id="regime-chart",
                                            config={'displayModeBar': False}
                                        )
                                    ]
                                )
                            ])
                        ], className="h-100 bg-dark text-light")
                    ], width=6),

                    # Volatility forecast chart
                    dbc.Col([
                        dbc.Card([
                            dbc.CardHeader("Volatility Forecast"),
                            dbc.CardBody([
                                dcc.Loading(
                                    id="loading-forecast-chart",
                                    children=[
                                        dcc.Graph(
                                            id="forecast-chart",
                                            config={'displayModeBar': False}
                                        )
                                    ]
                                )
                            ])
                        ], className="h-100 bg-dark text-light")
                    ], width=6)
                ]),

                # Hidden div for storing data
                html.Div(id="stored-data", style={"display": "none"}),

                # Interval for auto-updates
                dcc.Interval(
                    id="auto-update-interval",
                    interval=self.update_interval,
                    n_intervals=0,
                    disabled=False
                )
            ],
            className="p-4"
        )

        # Combine sidebar and content in a row
        self.app.layout = dbc.Container(
            [
                # Header
                html.H1(self.title, className="my-4 text-center"),

                # Main row with sidebar and content
                dbc.Row([
                    # Sidebar column
                    dbc.Col(sidebar, width=3, className="vh-100"),

                    # Content column
                    dbc.Col(content, width=9)
                ], className="g-0"),

                # Footer
                html.Footer(
                    f"© {datetime.now().year} Progressive Adaptive Volatility Indicator",
                    className="text-center text-muted my-4"
                )
            ],
            fluid=True,
            className="bg-dark text-light"
        )

    def _configure_callbacks(self):
        """Configure the dashboard callbacks."""
        # Callback to populate asset dropdown on page load
        @self.app.callback(
            Output("asset-dropdown", "options"),
            Input("asset-dropdown", "value")
        )
        def populate_asset_dropdown(value):
            try:
                assets = self.volatility_indicator.get_monitored_assets()
                return [{'label': asset, 'value': asset} for asset in assets]
            except Exception as e:
                logger.error(f"Error populating asset dropdown: {str(e)}")
                return []

        # Callback to handle auto-update toggling
        @self.app.callback(
            Output("auto-update-interval", "disabled"),
            Input("auto-update-toggle", "value")
        )
        def toggle_auto_update(value):
            return not value

        # Callback to update stored data
        @self.app.callback(
            Output("stored-data", "children"),
            Output("last-update-text", "children"),
            [
                Input("update-button", "n_clicks"),
                Input("auto-update-interval", "n_intervals"),
                Input("asset-dropdown", "value"),
                Input("time-period", "value")
            ],
            [
                State("stored-data", "children")
            ],
            prevent_initial_call=True
        )
        def update_data(n_clicks, n_intervals, asset, time_period, stored_data):
            if asset is None:
                # Try to get the first asset if none is selected
                try:
                    assets = self.volatility_indicator.get_monitored_assets()
                    if assets:
                        asset = assets[0]
                    else:
                        return dash.no_update, "No assets available"
                except Exception as e:
                    logger.error(f"Error getting assets: {str(e)}")
                    return dash.no_update, "Error fetching assets"

            try:
                # Get time range based on selected period
                end_date = datetime.now()
                if time_period == '1d':
                    start_date = end_date - timedelta(days=1)
                elif time_period == '1w':
                    start_date = end_date - timedelta(weeks=1)
                elif time_period == '1m':
                    start_date = end_date - timedelta(days=30)
                elif time_period == '3m':
                    start_date = end_date - timedelta(days=90)
                else:  # 'all'
                    start_date = None

                # Get data from indicator
                price_data = self.volatility_indicator.get_price_data(asset, start_date, end_date)
                vol_data = self.volatility_indicator.get_volatility_data(asset, start_date, end_date)
                regime_data = self.volatility_indicator.get_regime_data(asset, start_date, end_date)
                signal_data = self.volatility_indicator.get_signal_data(asset, start_date, end_date)
                prediction_data = self.volatility_indicator.get_predictions(asset, start_date, end_date)

                # Store data in JSON format
                data = {
                    'asset': asset,
                    'time_period': time_period,
                    'price_data': price_data.to_json(orient='split', date_format='iso'),
                    'vol_data': vol_data.to_json(orient='split', date_format='iso'),
                    'regime_data': regime_data.to_json(orient='split', date_format='iso') if regime_data is not None else None,
                    'signal_data': signal_data.to_json(orient='split', date_format='iso') if signal_data is not None else None,
                    'prediction_data': prediction_data.to_json(orient='split', date_format='iso') if prediction_data is not None else None,
                    'update_time': datetime.now().isoformat()
                }

                # Update last update time
                update_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

                return json.dumps(data), f"Last updated: {update_time}"

            except Exception as e:
                logger.error(f"Error updating data: {str(e)}")
                return dash.no_update, f"Error: {str(e)}"

        # Callback to update main chart
        @self.app.callback(
            Output("main-chart", "figure"),
            [
                Input("stored-data", "children"),
                Input("vol-components", "value"),
                Input("signal-overlay", "value")
            ]
        )
        def update_main_chart(stored_data_json, vol_components, signal_overlay):
            if not stored_data_json:
                # Return empty figure if no data
                return go.Figure()

            try:
                # Parse stored data
                stored_data = json.loads(stored_data_json)

                # Convert JSON data back to DataFrames
                price_data = pd.read_json(stored_data['price_data'], orient='split')
                vol_data = pd.read_json(stored_data['vol_data'], orient='split')

                regime_data = None
                if stored_data['regime_data']:
                    regime_data = pd.read_json(stored_data['regime_data'], orient='split')

                signal_data = None
                if 'signals' in signal_overlay and stored_data['signal_data']:
                    signal_data = pd.read_json(stored_data['signal_data'], orient='split')

                # Filter vol_data columns based on selected components
                selected_vol_data = vol_data[vol_components].copy() if all(col in vol_data.columns for col in vol_components) else vol_data

                # Add prediction data if selected
                if 'predicted_vol' in vol_components and stored_data['prediction_data']:
                    prediction_data = pd.read_json(stored_data['prediction_data'], orient='split')
                    # Ensure it's a Series for compatibility with chart generator
                    if isinstance(prediction_data, pd.DataFrame):
                        prediction_data = prediction_data.iloc[:, 0]
                else:
                    prediction_data = None

                # Create interactive chart
                fig = self.chart_generator.create_interactive_volatility_chart(
                    price_data=price_data,
                    volatility_data=selected_vol_data,
                    predicted_volatility=prediction_data,
                    regime_data=regime_data if 'regimes' in signal_overlay else None,
                    signals=signal_data
                )

                return fig

            except Exception as e:
                logger.error(f"Error updating main chart: {str(e)}")
                # Return empty figure on error
                return go.Figure()

        # Callback to update stats cards
        @self.app.callback(
            [
                Output("current-vol-value", "children"),
                Output("current-vol-trend", "children"),
                Output("predicted-vol-value", "children"),
                Output("predicted-vol-change", "children"),
                Output("market-regime-value", "children"),
                Output("regime-duration", "children"),
                Output("latest-signal-value", "children"),
                Output("signal-time", "children")
            ],
            Input("stored-data", "children")
        )
        def update_stats_cards(stored_data_json):
            if not stored_data_json:
                # Return empty values if no data
                return "N/A", "N/A", "N/A", "N/A", "N/A", "N/A", "N/A", "N/A"

            try:
                # Parse stored data
                stored_data = json.loads(stored_data_json)

                # Convert JSON data back to DataFrames
                vol_data = pd.read_json(stored_data['vol_data'], orient='split')

                # Current volatility
                if 'composite_vol' in vol_data.columns:
                    current_vol = vol_data['composite_vol'].iloc[-1]
                    # Calculate trend (percent change over last 24 hours)
                    if len(vol_data) > 1:
                        vol_24h_ago = vol_data['composite_vol'].iloc[-min(len(vol_data), 24)]
                        vol_change_pct = (current_vol - vol_24h_ago) / vol_24h_ago * 100
                        trend_text = f"{'↑' if vol_change_pct >= 0 else '↓'} {abs(vol_change_pct):.2f}% in 24h"
                        trend_style = "color: green" if vol_change_pct < 0 else "color: red"
                        vol_trend = html.Span(trend_text, style=trend_style)
                    else:
                        vol_trend = "Insufficient data for trend"
                else:
                    current_vol = vol_data.iloc[-1, 0]
                    vol_trend = "N/A"

                current_vol_formatted = f"{current_vol:.6f}"

                # Predicted volatility
                if stored_data['prediction_data']:
                    prediction_data = pd.read_json(stored_data['prediction_data'], orient='split')
                    if isinstance(prediction_data, pd.DataFrame):
                        predicted_vol = prediction_data.iloc[-1, 0]
                    else:
                        predicted_vol = prediction_data.iloc[-1]

                    # Calculate change from current
                    vol_pred_change_pct = (predicted_vol - current_vol) / current_vol * 100
                    pred_change_text = f"{'↑' if vol_pred_change_pct >= 0 else '↓'} {abs(vol_pred_change_pct):.2f}% from current"
                    pred_change_style = "color: red" if vol_pred_change_pct >= 0 else "color: green"
                    vol_pred_change = html.Span(pred_change_text, style=pred_change_style)

                    predicted_vol_formatted = f"{predicted_vol:.6f}"
                else:
                    predicted_vol_formatted = "N/A"
                    vol_pred_change = "No prediction available"

                # Market regime
                if stored_data['regime_data']:
                    regime_data = pd.read_json(stored_data['regime_data'], orient='split')
                    if 'regime' in regime_data.columns:
                        current_regime = regime_data['regime'].iloc[-1]

                        # Calculate duration of current regime
                        current_regime_data = regime_data[regime_data['regime'] == current_regime]
                        if not current_regime_data.empty:
                            # Find the start of the current continuous regime
                            regime_changes = regime_data['regime'].ne(regime_data['regime'].shift()).cumsum()
                            current_regime_group = regime_changes.iloc[-1]
                            regime_start = regime_data[regime_changes == current_regime_group].index[0]

                            # Calculate duration
                            duration = regime_data.index[-1] - regime_start
                            days = duration.days
                            hours = duration.seconds // 3600

                            if days > 0:
                                duration_text = f"{days} days {hours} hours"
                            else:
                                duration_text = f"{hours} hours"
                        else:
                            duration_text = "Unknown"
                    else:
                        current_regime = "N/A"
                        duration_text = "N/A"
                else:
                    current_regime = "N/A"
                    duration_text = "N/A"

                # Latest signal
                if stored_data['signal_data']:
                    signal_data = pd.read_json(stored_data['signal_data'], orient='split')

                    # Find the most recent signal
                    signal_columns = [col for col in signal_data.columns if col not in ['close', 'volume']]

                    latest_signal = "None"
                    signal_time_text = "N/A"

                    if signal_columns:
                        # Create a boolean mask for any signal
                        any_signal = signal_data[signal_columns].any(axis=1)

                        # Get rows with signals
                        signals_only = signal_data[any_signal]

                        if not signals_only.empty:
                            # Get the most recent signal
                            latest_signal_row = signals_only.iloc[-1]

                            # Find which signal(s) are active
                            active_signals = [col for col in signal_columns if latest_signal_row[col]]

                            if active_signals:
                                latest_signal = ", ".join(active_signals)
                                signal_time = signals_only.index[-1]
                                signal_time_text = signal_time.strftime("%Y-%m-%d %H:%M")
                else:
                    latest_signal = "No signals available"
                    signal_time_text = "N/A"

                return (
                    current_vol_formatted,
                    vol_trend,
                    predicted_vol_formatted,
                    vol_pred_change,
                    current_regime,
                    duration_text,
                    latest_signal,
                    signal_time_text
                )

            except Exception as e:
                logger.error(f"Error updating stats cards: {str(e)}")
                return "Error", str(e), "Error", str(e), "Error", str(e), "Error", str(e)

        # Callback to update regime distribution chart
        @self.app.callback(
            Output("regime-chart", "figure"),
            Input("stored-data", "children")
        )
        def update_regime_chart(stored_data_json):
            if not stored_data_json:
                # Return empty figure if no data
                return go.Figure()

            try:
                # Parse stored data
                stored_data = json.loads(stored_data_json)

                # Check if regime data is available
                if not stored_data['regime_data']:
                    # Return empty figure with message
                    fig = go.Figure()
                    fig.add_annotation(
                        text="No regime data available",
                        xref="paper", yref="paper",
                        x=0.5, y=0.5,
                        showarrow=False,
                        font=dict(size=16, color="white")
                    )
                    fig.update_layout(
                        template="plotly_dark",
                        paper_bgcolor="rgba(0,0,0,0)",
                        plot_bgcolor="rgba(0,0,0,0)"
                    )
                    return fig

                # Convert JSON data back to DataFrame
                regime_data = pd.read_json(stored_data['regime_data'], orient='split')

                if 'regime' not in regime_data.columns:
                    # Return empty figure with message
                    fig = go.Figure()
                    fig.add_annotation(
                        text="No regime column in data",
                        xref="paper", yref="paper",
                        x=0.5, y=0.5,
                        showarrow=False,
                        font=dict(size=16, color="white")
                    )
                    fig.update_layout(
                        template="plotly_dark",
                        paper_bgcolor="rgba(0,0,0,0)",
                        plot_bgcolor="rgba(0,0,0,0)"
                    )
                    return fig

                # Count occurrences of each regime
                regime_counts = regime_data['regime'].value_counts()

                # Calculate percentages
                total = regime_counts.sum()
                regime_pcts = (regime_counts / total * 100).round(1)

                # Create labels with percentages
                labels = [f"{regime} ({pct}%)" for regime, pct in zip(regime_counts.index, regime_pcts)]

                # Define colors for regimes
                regime_colors = {
                    'low_vol': '#4CAF50',  # Green
                    'normal_vol': '#FFEB3B',  # Yellow
                    'high_vol': '#FF9800',  # Orange
                    'extreme_vol': '#F44336'  # Red
                }

                # Get colors for each regime
                colors = [regime_colors.get(regime, f'hsl({hash(regime) % 360}, 70%, 50%)') for regime in regime_counts.index]

                # Create pie chart
                fig = go.Figure(data=[
                    go.Pie(
                        labels=labels,
                        values=regime_counts,
                        hole=0.4,
                        marker=dict(colors=colors)
                    )
                ])

                # Update layout
                fig.update_layout(
                    template="plotly_dark",
                    margin=dict(l=20, r=20, t=20, b=20),
                    paper_bgcolor="rgba(0,0,0,0)",
                    plot_bgcolor="rgba(0,0,0,0)",
                    legend=dict(
                        orientation="h",
                        yanchor="bottom",
                        y=-0.2,
                        xanchor="center",
                        x=0.5
                    )
                )

                return fig

            except Exception as e:
                logger.error(f"Error updating regime chart: {str(e)}")
                # Return empty figure on error
                fig = go.Figure()
                fig.add_annotation(
                    text=f"Error: {str(e)}",
                    xref="paper", yref="paper",
                    x=0.5, y=0.5,
                    showarrow=False,
                    font=dict(size=14, color="white")
                )
                fig.update_layout(
                    template="plotly_dark",
                    paper_bgcolor="rgba(0,0,0,0)",
                    plot_bgcolor="rgba(0,0,0,0)"
                )
                return fig

        # Callback to update forecast chart
        @self.app.callback(
            Output("forecast-chart", "figure"),
            Input("stored-data", "children")
        )
        def update_forecast_chart(stored_data_json):
            if not stored_data_json:
                # Return empty figure if no data
                return go.Figure()

            try:
                # Parse stored data
                stored_data = json.loads(stored_data_json)

                # Check if prediction data is available
                if not stored_data['prediction_data']:
                    # Return empty figure with message
                    fig = go.Figure()
                    fig.add_annotation(
                        text="No forecast data available",
                        xref="paper", yref="paper",
                        x=0.5, y=0.5,
                        showarrow=False,
                        font=dict(size=16, color="white")
                    )
                    fig.update_layout(
                        template="plotly_dark",
                        paper_bgcolor="rgba(0,0,0,0)",
                        plot_bgcolor="rgba(0,0,0,0)"
                    )
                    return fig

                # Convert JSON data back to DataFrames
                vol_data = pd.read_json(stored_data['vol_data'], orient='split')
                prediction_data = pd.read_json(stored_data['prediction_data'], orient='split')

                # Get actual volatility (last 30 data points)
                if 'composite_vol' in vol_data.columns:
                    actual_vol = vol_data['composite_vol'].iloc[-30:]
                else:
                    actual_vol = vol_data.iloc[-30:, 0]

                # Ensure prediction_data is a Series
                if isinstance(prediction_data, pd.DataFrame):
                    prediction_series = prediction_data.iloc[:, 0]
                else:
                    prediction_series = prediction_data

                # Create figure
                fig = go.Figure()

                # Add actual volatility trace
                fig.add_trace(
                    go.Scatter(
                        x=actual_vol.index,
                        y=actual_vol,
                        mode='lines',
                        name='Actual Volatility',
                        line=dict(color='#F44336', width=2)
                    )
                )

                # Add predicted volatility trace
                fig.add_trace(
                    go.Scatter(
                        x=prediction_series.index,
                        y=prediction_series,
                        mode='lines',
                        name='Predicted Volatility',
                        line=dict(color='#4CAF50', width=2, dash='dash')
                    )
                )

                # Add vertical line at current time
                current_time = vol_data.index[-1]
                fig.add_shape(
                    type="line",
                    x0=current_time,
                    y0=0,
                    x1=current_time,
                    y1=1,
                    yref="paper",
                    line=dict(
                        color="White",
                        width=2,
                        dash="dash",
                    )
                )

                fig.add_annotation(
                    x=current_time,
                    y=1,
                    yref="paper",
                    text="Now",
                    showarrow=False,
                    yshift=10
                )

                # Update layout
                fig.update_layout(
                    template="plotly_dark",
                    margin=dict(l=20, r=20, t=20, b=20),
                    paper_bgcolor="rgba(0,0,0,0)",
                    plot_bgcolor="rgba(0,0,0,0)",
                    xaxis=dict(
                        showgrid=True,
                        gridwidth=1,
                        gridcolor='rgba(255, 255, 255, 0.1)'
                    ),
                    yaxis=dict(
                        showgrid=True,
                        gridwidth=1,
                        gridcolor='rgba(255, 255, 255, 0.1)',
                        zeroline=False
                    ),
                    legend=dict(
                        orientation="h",
                        yanchor="bottom",
                        y=-0.2,
                        xanchor="center",
                        x=0.5
                    )
                )

                return fig

            except Exception as e:
                logger.error(f"Error updating forecast chart: {str(e)}")
                # Return empty figure on error
                fig = go.Figure()
                fig.add_annotation(
                    text=f"Error: {str(e)}",
                    xref="paper", yref="paper",
                    x=0.5, y=0.5,
                    showarrow=False,
                    font=dict(size=14, color="white")
                )
                fig.update_layout(
                    template="plotly_dark",
                    paper_bgcolor="rgba(0,0,0,0)",
                    plot_bgcolor="rgba(0,0,0,0)"
                )
                return fig

    def start_background_data_update(self, interval=300):
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
                    # Get latest data for each asset
                    self.data_cache[asset] = {
                        'price_data': self.volatility_indicator.get_price_data(asset),
                        'vol_data': self.volatility_indicator.get_volatility_data(asset),
                        'regime_data': self.volatility_indicator.get_regime_data(asset),
                        'signal_data': self.volatility_indicator.get_signal_data(asset),
                        'prediction_data': self.volatility_indicator.get_predictions(asset)
                    }

                    self.last_update[asset] = datetime.now()

                logger.debug(f"Updated cached data for {len(assets)} assets")

            except Exception as e:
                logger.error(f"Error in background update loop: {str(e)}")

            # Sleep before next update
            time.sleep(interval)

    def run(self):
        """Run the dashboard server."""
        logger.info(f"Starting dashboard server on {self.host}:{self.port}")
        self.app.run_server(host=self.host, port=self.port, debug=self.debug)

    def get_cached_data(self, asset):
        """
        Get cached data for an asset.

        Parameters:
        -----------
        asset : str
            Asset symbol

        Returns:
        --------
        dict
            Cached data for the asset
        """
        return self.data_cache.get(asset, None)

    def create_config_panel(self):
        """
        Create a configuration panel for the dashboard.

        Returns:
        --------
        html.Div
            Configuration panel
        """
        return html.Div([
            html.H3("Configuration", className="mb-3"),

            # Volatility weights
            html.Div([
                html.Label("Volatility Component Weights"),
                dbc.Row([
                    dbc.Col([
                        html.Label("Micro"),
                        dbc.Input(type="number", id="micro-weight", min=0, max=1, step=0.1, value=0.3)
                    ]),
                    dbc.Col([
                        html.Label("Meso"),
                        dbc.Input(type="number", id="meso-weight", min=0, max=1, step=0.1, value=0.5)
                    ]),
                    dbc.Col([
                        html.Label("Macro"),
                        dbc.Input(type="number", id="macro-weight", min=0, max=1, step=0.1, value=0.2)
                    ])
                ])
            ], className="mb-3"),

            # Thresholds
            html.Div([
                html.Label("Alert Thresholds"),
                dbc.Row([
                    dbc.Col([
                        html.Label("Breakout"),
                        dbc.Input(type="number", id="breakout-threshold", min=0, max=5, step=0.1, value=2.0)
                    ]),
                    dbc.Col([
                        html.Label("Contraction"),
                        dbc.Input(type="number", id="contraction-threshold", min=0, max=5, step=0.1, value=0.5)
                    ])
                ])
            ], className="mb-3"),

            # Apply button
            dbc.Button("Apply Settings", id="apply-settings", color="primary", className="w-100")
        ], className="mb-4 p-3 border rounded")

    def create_indicators_panel(self, asset):
        """
        Create a panel showing various indicators for an asset.

        Parameters:
        -----------
        asset : str
            Asset symbol

        Returns:
        --------
        html.Div
            Indicators panel
        """
        # Get latest data
        data = self.data_cache.get(asset, {})

        price_data = data.get('price_data', None)
        vol_data = data.get('vol_data', None)
        regime_data = data.get('regime_data', None)

        # Extract latest values
        current_price = "N/A"
        price_change = "N/A"

        if price_data is not None and not price_data.empty:
            current_price = f"{price_data['close'].iloc[-1]:.2f}"

            if len(price_data) > 1:
                prev_close = price_data['close'].iloc[-2]
                curr_close = price_data['close'].iloc[-1]
                price_change = f"{((curr_close / prev_close) - 1) * 100:.2f}%"

        # Volatility values
        micro_vol = "N/A"
        meso_vol = "N/A"
        macro_vol = "N/A"
        composite_vol = "N/A"

        if vol_data is not None and not vol_data.empty:
            micro_vol = f"{vol_data['micro_vol'].iloc[-1]:.6f}" if 'micro_vol' in vol_data.columns else "N/A"
            meso_vol = f"{vol_data['meso_vol'].iloc[-1]:.6f}" if 'meso_vol' in vol_data.columns else "N/A"
            macro_vol = f"{vol_data['macro_vol'].iloc[-1]:.6f}" if 'macro_vol' in vol_data.columns else "N/A"
            composite_vol = f"{vol_data['composite_vol'].iloc[-1]:.6f}" if 'composite_vol' in vol_data.columns else "N/A"

        # Regime value
        current_regime = "N/A"

        if regime_data is not None and not regime_data.empty and 'regime' in regime_data.columns:
            current_regime = regime_data['regime'].iloc[-1]

        return html.Div([
            html.H3(f"{asset} Indicators", className="mb-3"),

            dbc.Row([
                # Price card
                dbc.Col([
                    dbc.Card([
                        dbc.CardHeader("Price"),
                        dbc.CardBody([
                            html.H4(current_price, className="card-title"),
                            html.P(f"Change: {price_change}", className="card-text")
                        ])
                    ], className="h-100")
                ], width=3),

                # Composite volatility card
                dbc.Col([
                    dbc.Card([
                        dbc.CardHeader("Composite Volatility"),
                        dbc.CardBody([
                            html.H4(composite_vol, className="card-title")
                        ])
                    ], className="h-100")
                ], width=3),

                # Regime card
                dbc.Col([
                    dbc.Card([
                        dbc.CardHeader("Current Regime"),
                        dbc.CardBody([
                            html.H4(current_regime, className="card-title")
                        ])
                    ], className="h-100")
                ], width=3),

                # Time card
                dbc.Col([
                    dbc.Card([
                        dbc.CardHeader("Latest Update"),
                        dbc.CardBody([
                            html.H4(
                                datetime.now().strftime("%H:%M:%S"),
                                id="clock",
                                className="card-title"
                            )
                        ])
                    ], className="h-100")
                ], width=3)
            ], className="mb-3"),

            dbc.Row([
                # Component volatilities
                dbc.Col([
                    dbc.Card([
                        dbc.CardHeader("Volatility Components"),
                        dbc.CardBody([
                            html.Div([
                                html.Label("Micro Volatility:"),
                                html.Span(micro_vol, className="ms-2")
                            ], className="mb-2"),
                            html.Div([
                                html.Label("Meso Volatility:"),
                                html.Span(meso_vol, className="ms-2")
                            ], className="mb-2"),
                            html.Div([
                                html.Label("Macro Volatility:"),
                                html.Span(macro_vol, className="ms-2")
                            ])
                        ])
                    ], className="h-100")
                ], width=6),

                # Trading recommendations
                dbc.Col([
                    dbc.Card([
                        dbc.CardHeader("Trading Recommendations"),
                        dbc.CardBody([
                            html.Div(id="trading-recommendations")
                        ])
                    ], className="h-100")
                ], width=6)
            ])
        ], className="mb-4 p-3 border rounded")

    def create_settings_modal(self):
        """
        Create a settings modal.

        Returns:
        --------
        dbc.Modal
            Settings modal
        """
        return dbc.Modal([
            dbc.ModalHeader("Dashboard Settings"),
            dbc.ModalBody([
                # Theme selection
                html.Div([
                    html.Label("Theme"),
                    dbc.Select(
                        id="theme-select",
                        options=[
                            {'label': 'Dark', 'value': 'darkly'},
                            {'label': 'Light', 'value': 'bootstrap'}
                        ],
                        value=self.theme
                    )
                ], className="mb-3"),

                # Update interval
                html.Div([
                    html.Label("Update Interval (seconds)"),
                    dbc.Input(
                        id="update-interval",
                        type="number",
                        min=1,
                        max=600,
                        step=1,
                        value=self.update_interval / 1000
                    )
                ], className="mb-3"),

                # Chart height
                html.Div([
                    html.Label("Chart Height (pixels)"),
                    dbc.Input(
                        id="chart-height",
                        type="number",
                        min=300,
                        max=1200,
                        step=50,
                        value=600
                    )
                ])
            ]),
            dbc.ModalFooter([
                dbc.Button("Close", id="close-settings", className="ms-auto")
            ])
        ], id="settings-modal")

    def create_alerts_panel(self, volatility_indicator):
        """
        Create a panel showing recent alerts.

        Parameters:
        -----------
        volatility_indicator : object
            Volatility indicator instance

        Returns:
        --------
        html.Div
            Alerts panel
        """
        # Check if AlertManager is available
        if not hasattr(volatility_indicator, 'signal_generator'):
            return html.Div("Alert system not initialized")

        signal_generator = volatility_indicator.signal_generator

        # Get recent alerts
        recent_alerts = []
        for asset in volatility_indicator.get_monitored_assets():
            # Check for volatility breakout signals
            signals = volatility_indicator.get_signal_data(asset)
            if signals is not None and not signals.empty:
                if 'volatility_breakout' in signals.columns:
                    breakout_points = signals[signals['volatility_breakout'] == 1].index
                    for point in breakout_points[-5:]:  # Show last 5 breakouts
                        recent_alerts.append({
                            'timestamp': point,
                            'asset': asset,
                            'type': 'Volatility Breakout',
                            'priority': 'high'
                        })

                if 'volatility_contraction' in signals.columns:
                    contraction_points = signals[signals['volatility_contraction'] == 1].index
                    for point in contraction_points[-5:]:  # Show last 5 contractions
                        recent_alerts.append({
                            'timestamp': point,
                            'asset': asset,
                            'type': 'Volatility Contraction',
                            'priority': 'medium'
                        })

        # Sort by timestamp (latest first)
        recent_alerts.sort(key=lambda x: x['timestamp'], reverse=True)

        # Create alert list
        alerts_list = []
        for alert in recent_alerts[:10]:  # Show up to 10 alerts
            # Format timestamp
            timestamp_str = alert['timestamp'].strftime("%Y-%m-%d %H:%M")

            # Determine alert color
            color = {
                'high': 'danger',
                'medium': 'warning',
                'low': 'info'
            }.get(alert['priority'], 'secondary')

            alerts_list.append(
                dbc.ListGroupItem(
                    [
                        html.Div(f"{alert['type']} - {alert['asset']}", className="fw-bold"),
                        html.Div(timestamp_str, className="text-muted small")
                    ],
                    color=color
                )
            )

        # Create panel
        return html.Div([
            html.H3("Recent Alerts", className="mb-3"),
            dbc.ListGroup(alerts_list if alerts_list else [
                dbc.ListGroupItem("No recent alerts")
            ])
        ], className="mb-4 p-3 border rounded")

    def update_clock(self, n_intervals):
        """
        Update the clock display.

        Parameters:
        -----------
        n_intervals : int
            Number of intervals elapsed

        Returns:
        --------
        str
            Current time
        """
        return datetime.now().strftime("%H:%M:%S")

    def update_trading_recommendations(self, asset):
        """
        Update trading recommendations based on current data.

        Parameters:
        -----------
        asset : str
            Asset symbol

        Returns:
        --------
        html.Div
            Trading recommendations
        """
        # Get latest data
        data = self.data_cache.get(asset, {})

        price_data = data.get('price_data', None)
        vol_data = data.get('vol_data', None)
        regime_data = data.get('regime_data', None)

        # Default recommendation
        if price_data is None or vol_data is None:
            return html.Div("Insufficient data for recommendations")

        # Get current regime
        current_regime = None
        if regime_data is not None and not regime_data.empty and 'regime' in regime_data.columns:
            current_regime = regime_data['regime'].iloc[-1]

        # Get current volatility
        current_vol = None
        if 'composite_vol' in vol_data.columns:
            current_vol = vol_data['composite_vol'].iloc[-1]

        # Generate recommendations
        recommendations = []

        # Position sizing recommendation
        position_size = "Medium"
        position_color = "primary"

        if current_regime == 'low_vol':
            position_size = "Large"
            position_color = "success"
        elif current_regime == 'high_vol':
            position_size = "Small"
            position_color = "warning"
        elif current_regime == 'extreme_vol':
            position_size = "Minimal"
            position_color = "danger"

        recommendations.append(html.Div([
            html.Span("Position Size: ", className="fw-bold"),
            dbc.Badge(position_size, color=position_color, className="ms-1")
        ], className="mb-2"))

        # Risk management recommendation
        if current_vol is not None:
            # Determine stop loss distance based on volatility
            stop_loss_pct = min(max(current_vol * 2.5, 0.01), 0.1) * 100

            recommendations.append(html.Div([
                html.Span("Suggested Stop Loss: ", className="fw-bold"),
                html.Span(f"{stop_loss_pct:.2f}% from entry")
            ], className="mb-2"))

            # Take profit recommendation
            take_profit_pct = stop_loss_pct * 2  # 2:1 reward-risk ratio

            recommendations.append(html.Div([
                html.Span("Suggested Take Profit: ", className="fw-bold"),
                html.Span(f"{take_profit_pct:.2f}% from entry")
            ], className="mb-2"))

        # Trading strategy recommendation
        strategy = "Neutral"
        strategy_color = "secondary"

        if current_regime == 'low_vol':
            strategy = "Breakout Strategy"
            strategy_color = "success"
        elif current_regime == 'normal_vol':
            strategy = "Trend Following"
            strategy_color = "primary"
        elif current_regime == 'high_vol':
            strategy = "Range Trading"
            strategy_color = "warning"
        elif current_regime == 'extreme_vol':
            strategy = "Reduced Exposure"
            strategy_color = "danger"

        recommendations.append(html.Div([
            html.Span("Recommended Strategy: ", className="fw-bold"),
            dbc.Badge(strategy, color=strategy_color, className="ms-1")
        ]))

        return html.Div(recommendations)