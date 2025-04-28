"""
Module for creating charts and visualizations for the volatility indicator.
"""
import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.gridspec as gridspec
from matplotlib.colors import LinearSegmentedColormap
import seaborn as sns
import plotly.graph_objects as go
import plotly.express as px

from plotly.subplots import make_subplots
import logging
from datetime import datetime
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, project_root)
from crypto_volatility_indicator.utils.logger import get_logger

logger = get_logger(__name__)


class VolatilityChartGenerator:
    """
    Generator for volatility charts and visualizations.
    """

    def __init__(self, config=None):
        """
        Initialize the chart generator.

        Parameters:
        -----------
        config : dict, optional
            Configuration parameters
        """
        self.config = config or {}

        # Set default style
        self.style = self.config.get('style', 'dark_background')
        plt.style.use(self.style)

        # Default colors
        self.colors = self.config.get('colors', {
            'price': '#2196F3',  # Blue
            'volatility': '#F44336',  # Red
            'predicted_volatility': '#4CAF50',  # Green
            'micro_vol': '#FF9800',  # Orange
            'meso_vol': '#9C27B0',  # Purple
            'macro_vol': '#607D8B',  # Blue Gray
            'regime_1': '#4CAF50',  # Green
            'regime_2': '#FFEB3B',  # Yellow
            'regime_3': '#FF9800',  # Orange
            'regime_4': '#F44336',  # Red
            'background': '#212121',  # Dark Gray
            'grid': '#424242',  # Medium Gray
            'text': '#FFFFFF'  # White
        })

        # Chart dimensions
        self.width = self.config.get('width', 14)
        self.height = self.config.get('height', 8)
        self.dpi = self.config.get('dpi', 100)

        # Chart output directory
        self.output_dir = self.config.get('output_dir', 'charts')
        os.makedirs(self.output_dir, exist_ok=True)

    def plot_volatility_indicator(self, price_data, volatility_data,
                                  predicted_volatility=None, regime_data=None,
                                  signals=None, save_path=None, show=True,
                                  start_date=None, end_date=None):
        """
        Create a comprehensive volatility indicator chart.

        Parameters:
        -----------
        price_data : pd.DataFrame
            DataFrame with price data (must have 'close' column)
        volatility_data : pd.DataFrame
            DataFrame with volatility data
        predicted_volatility : pd.Series, optional
            Series with predicted volatility
        regime_data : pd.DataFrame, optional
            DataFrame with regime data
        signals : pd.DataFrame, optional
            DataFrame with trading signals
        save_path : str, optional
            Path to save the figure
        show : bool
            Whether to display the figure
        start_date, end_date : str or datetime, optional
            Date range to plot

        Returns:
        --------
        matplotlib.figure.Figure
            The created figure
        """
        # Filter data by date range if specified
        if start_date is not None or end_date is not None:
            price_data = self._filter_by_date(price_data, start_date, end_date)
            volatility_data = self._filter_by_date(volatility_data, start_date, end_date)
            if predicted_volatility is not None:
                predicted_volatility = self._filter_by_date(predicted_volatility, start_date, end_date)
            if regime_data is not None:
                regime_data = self._filter_by_date(regime_data, start_date, end_date)
            if signals is not None:
                signals = self._filter_by_date(signals, start_date, end_date)

        # Create figure and subplot grid
        fig = plt.figure(figsize=(self.width, self.height), dpi=self.dpi,
                         facecolor=self.colors['background'])

        # Determine grid layout based on available data
        n_rows = 2  # Price and volatility are always shown

        if regime_data is not None:
            n_rows += 1

        # Create GridSpec
        gs = gridspec.GridSpec(n_rows, 1, height_ratios=[2] + [1] * (n_rows - 1))

        # Price chart
        ax1 = fig.add_subplot(gs[0])
        self._plot_price_chart(ax1, price_data, signals)

        # Volatility chart
        ax2 = fig.add_subplot(gs[1], sharex=ax1)
        self._plot_volatility_chart(ax2, volatility_data, predicted_volatility)

        # Regime chart if available
        if regime_data is not None:
            ax3 = fig.add_subplot(gs[2], sharex=ax1)
            self._plot_regime_chart(ax3, regime_data)

        # Adjust layout
        plt.tight_layout()
        fig.subplots_adjust(hspace=0.2)

        # Save figure if path provided
        if save_path:
            plt.savefig(save_path, dpi=self.dpi, bbox_inches='tight')
            logger.info(f"Chart saved to {save_path}")

        # Show or close figure
        if show:
            plt.show()
        else:
            plt.close(fig)

        return fig

    def _filter_by_date(self, data, start_date=None, end_date=None):
        """
        Filter DataFrame or Series by date range.

        Parameters:
        -----------
        data : pd.DataFrame or pd.Series
            Data to filter
        start_date, end_date : str or datetime, optional
            Date range to filter by

        Returns:
        --------
        pd.DataFrame or pd.Series
            Filtered data
        """
        if data is None:
            return None

        if start_date is not None:
            data = data[data.index >= pd.to_datetime(start_date)]

        if end_date is not None:
            data = data[data.index <= pd.to_datetime(end_date)]

        return data

    def _plot_price_chart(self, ax, price_data, signals=None):
        """
        Plot price chart with signals if available.

        Parameters:
        -----------
        ax : matplotlib.axes.Axes
            Axes to plot on
        price_data : pd.DataFrame
            DataFrame with price data
        signals : pd.DataFrame, optional
            DataFrame with trading signals
        """
        # Plot price
        ax.plot(price_data.index, price_data['close'],
                color=self.colors['price'], linewidth=1.5, label='Price')

        # Add signals if available
        if signals is not None:
            if 'volatility_breakout' in signals.columns:
                breakout_points = signals[signals['volatility_breakout'] == 1].index
                if len(breakout_points) > 0:
                    breakout_prices = price_data.loc[breakout_points, 'close']
                    ax.scatter(breakout_points, breakout_prices,
                               color='yellow', s=100, marker='^',
                               label='Volatility Breakout')

            if 'volatility_contraction' in signals.columns:
                contraction_points = signals[signals['volatility_contraction'] == 1].index
                if len(contraction_points) > 0:
                    contraction_prices = price_data.loc[contraction_points, 'close']
                    ax.scatter(contraction_points, contraction_prices,
                               color='cyan', s=100, marker='v',
                               label='Volatility Contraction')

            if 'regime_change' in signals.columns:
                regime_change_points = signals[signals['regime_change'] == 1].index
                if len(regime_change_points) > 0:
                    regime_change_prices = price_data.loc[regime_change_points, 'close']
                    ax.scatter(regime_change_points, regime_change_prices,
                               color='white', s=100, marker='*',
                               label='Regime Change')

        # Set title and labels
        ax.set_title('Price Chart', color=self.colors['text'], fontsize=14)
        ax.set_ylabel('Price', color=self.colors['text'], fontsize=12)

        # Style the chart
        ax.grid(True, alpha=0.3, color=self.colors['grid'])
        ax.tick_params(colors=self.colors['text'])
        ax.legend(loc='upper left', facecolor=self.colors['background'],
                  edgecolor=self.colors['grid'], labelcolor=self.colors['text'])

        # Format x-axis
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')

    def _plot_volatility_chart(self, ax, volatility_data, predicted_volatility=None):
        """
        Plot volatility chart with predictions if available.

        Parameters:
        -----------
        ax : matplotlib.axes.Axes
            Axes to plot on
        volatility_data : pd.DataFrame
            DataFrame with volatility data
        predicted_volatility : pd.Series, optional
            Series with predicted volatility
        """
        # Plot actual volatility
        if 'composite_vol' in volatility_data.columns:
            # Plot composite volatility
            ax.plot(volatility_data.index, volatility_data['composite_vol'],
                    color=self.colors['volatility'], linewidth=1.5,
                    label='Composite Volatility')

            # Plot component volatilities with lower alpha
            if 'micro_vol' in volatility_data.columns:
                ax.plot(volatility_data.index, volatility_data['micro_vol'],
                        color=self.colors['micro_vol'], linewidth=1, alpha=0.6,
                        label='Micro Volatility')

            if 'meso_vol' in volatility_data.columns:
                ax.plot(volatility_data.index, volatility_data['meso_vol'],
                        color=self.colors['meso_vol'], linewidth=1, alpha=0.6,
                        label='Meso Volatility')

            if 'macro_vol' in volatility_data.columns:
                ax.plot(volatility_data.index, volatility_data['macro_vol'],
                        color=self.colors['macro_vol'], linewidth=1, alpha=0.6,
                        label='Macro Volatility')
        else:
            # If no composite_vol, plot the first column
            col = volatility_data.columns[0]
            ax.plot(volatility_data.index, volatility_data[col],
                    color=self.colors['volatility'], linewidth=1.5,
                    label=col)

        # Plot predicted volatility if available
        if predicted_volatility is not None:
            ax.plot(predicted_volatility.index, predicted_volatility,
                    color=self.colors['predicted_volatility'], linewidth=1.5,
                    linestyle='--', label='Predicted Volatility')

        # Set title and labels
        ax.set_title('Volatility Indicators', color=self.colors['text'], fontsize=14)
        ax.set_ylabel('Volatility', color=self.colors['text'], fontsize=12)

        # Style the chart
        ax.grid(True, alpha=0.3, color=self.colors['grid'])
        ax.tick_params(colors=self.colors['text'])
        ax.legend(loc='upper left', facecolor=self.colors['background'],
                  edgecolor=self.colors['grid'], labelcolor=self.colors['text'])

    def _plot_regime_chart(self, ax, regime_data):
        """
        Plot market regime chart.

        Parameters:
        -----------
        ax : matplotlib.axes.Axes
            Axes to plot on
        regime_data : pd.DataFrame
            DataFrame with regime data
        """
        if 'regime' not in regime_data.columns:
            logger.warning("No 'regime' column found in regime_data")
            return

        # Convert regime labels to numeric values for plotting
        regimes = regime_data['regime'].unique()
        regime_map = {regime: i for i, regime in enumerate(sorted(regimes))}
        numeric_regimes = regime_data['regime'].map(regime_map)

        # Create colors for regimes
        n_regimes = len(regimes)
        regime_colors = [self.colors.get(f'regime_{i + 1}', f'C{i}') for i in range(n_regimes)]

        # Plot as a step function
        ax.step(regime_data.index, numeric_regimes, where='post',
                color=self.colors['text'], linewidth=1.5)

        # Fill areas with appropriate colors
        for i, regime in enumerate(sorted(regimes)):
            mask = regime_data['regime'] == regime
            ax.fill_between(regime_data.index, i, i + 1, where=mask,
                            color=regime_colors[i], alpha=0.7, step='post')

        # Set y-ticks to regime names
        ax.set_yticks(np.arange(n_regimes) + 0.5)
        ax.set_yticklabels(sorted(regimes), fontsize=10, color=self.colors['text'])

        # Set title and labels
        ax.set_title('Market Regime', color=self.colors['text'], fontsize=14)

        # Style the chart
        ax.grid(True, alpha=0.3, color=self.colors['grid'])
        ax.tick_params(colors=self.colors['text'])

    def create_interactive_volatility_chart(self, price_data, volatility_data,
                                            predicted_volatility=None, regime_data=None,
                                            signals=None, save_path=None):
        """
        Create an interactive Plotly chart for volatility analysis.

        Parameters:
        -----------
        price_data : pd.DataFrame
            DataFrame with price data (must have 'close' column)
        volatility_data : pd.DataFrame
            DataFrame with volatility data
        predicted_volatility : pd.Series, optional
            Series with predicted volatility
        regime_data : pd.DataFrame, optional
            DataFrame with regime data
        signals : pd.DataFrame, optional
            DataFrame with trading signals
        save_path : str, optional
            Path to save the figure as HTML

        Returns:
        --------
        plotly.graph_objects.Figure
            Interactive Plotly figure
        """
        # Determine number of rows based on available data
        n_rows = 2  # Price and volatility are always shown
        if regime_data is not None:
            n_rows += 1

        # Create subplots
        fig = make_subplots(
            rows=n_rows,
            cols=1,
            shared_xaxes=True,
            vertical_spacing=0.05,
            subplot_titles=("Price", "Volatility", "Market Regime")[:n_rows],
            row_heights=[0.5, 0.3] + ([0.2] if n_rows > 2 else [])
        )

        # Add price trace
        fig.add_trace(
            go.Scatter(
                x=price_data.index,
                y=price_data['close'],
                mode='lines',
                name='Price',
                line=dict(color='#2196F3', width=2)
            ),
            row=1, col=1
        )

        # Add signals if available
        if signals is not None:
            # Volatility breakout signals
            if 'volatility_breakout' in signals.columns:
                breakout_points = signals[signals['volatility_breakout'] == 1].index
                if len(breakout_points) > 0:
                    breakout_prices = price_data.loc[breakout_points, 'close']
                    fig.add_trace(
                        go.Scatter(
                            x=breakout_points,
                            y=breakout_prices,
                            mode='markers',
                            name='Volatility Breakout',
                            marker=dict(
                                symbol='triangle-up',
                                size=12,
                                color='yellow'
                            )
                        ),
                        row=1, col=1
                    )

            # Volatility contraction signals
            if 'volatility_contraction' in signals.columns:
                contraction_points = signals[signals['volatility_contraction'] == 1].index
                if len(contraction_points) > 0:
                    contraction_prices = price_data.loc[contraction_points, 'close']
                    fig.add_trace(
                        go.Scatter(
                            x=contraction_points,
                            y=contraction_prices,
                            mode='markers',
                            name='Volatility Contraction',
                            marker=dict(
                                symbol='triangle-down',
                                size=12,
                                color='cyan'
                            )
                        ),
                        row=1, col=1
                    )

        # Add volatility traces
        # Add main volatility indicator
        if 'composite_vol' in volatility_data.columns:
            fig.add_trace(
                go.Scatter(
                    x=volatility_data.index,
                    y=volatility_data['composite_vol'],
                    mode='lines',
                    name='Composite Volatility',
                    line=dict(color='#F44336', width=2)
                ),
                row=2, col=1
            )

            # Add component volatilities
            vol_components = {
                'micro_vol': {'color': '#FF9800', 'name': 'Micro Volatility'},
                'meso_vol': {'color': '#9C27B0', 'name': 'Meso Volatility'},
                'macro_vol': {'color': '#607D8B', 'name': 'Macro Volatility'}
            }

            for vol_name, props in vol_components.items():
                if vol_name in volatility_data.columns:
                    fig.add_trace(
                        go.Scatter(
                            x=volatility_data.index,
                            y=volatility_data[vol_name],
                            mode='lines',
                            name=props['name'],
                            line=dict(color=props['color'], width=1.5, dash='dot'),
                            opacity=0.7
                        ),
                        row=2, col=1
                    )
        else:
            # If no composite_vol, use the first column
            col = volatility_data.columns[0]
            fig.add_trace(
                go.Scatter(
                    x=volatility_data.index,
                    y=volatility_data[col],
                    mode='lines',
                    name=col,
                    line=dict(color='#F44336', width=2)
                ),
                row=2, col=1
            )

        # Add predicted volatility if available
        if predicted_volatility is not None:
            fig.add_trace(
                go.Scatter(
                    x=predicted_volatility.index,
                    y=predicted_volatility,
                    mode='lines',
                    name='Predicted Volatility',
                    line=dict(color='#4CAF50', width=2, dash='dash')
                ),
                row=2, col=1
            )

        # Add regime chart if available
        if regime_data is not None and 'regime' in regime_data.columns:
            # Convert regime labels to numeric values
            regimes = sorted(regime_data['regime'].unique())
            regime_map = {regime: i for i, regime in enumerate(regimes)}

            # Create a numeric series for regimes
            numeric_regimes = regime_data['regime'].map(regime_map)

            # Add regime chart
            fig.add_trace(
                go.Scatter(
                    x=regime_data.index,
                    y=numeric_regimes,
                    mode='lines',
                    name='Market Regime',
                    line=dict(color='#FFFFFF', width=0),
                    fill='tozeroy',
                    fillcolor='rgba(255, 255, 255, 0.1)'
                ),
                row=3, col=1
            )

            # Set y-axis properties for regime chart
            fig.update_yaxes(
                tickvals=list(range(len(regimes))),
                ticktext=regimes,
                row=3, col=1
            )

        # Update layout
        fig.update_layout(
            template='plotly_dark',
            height=800,
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1
            ),
            margin=dict(l=50, r=50, t=70, b=50)
        )

        # Update axes
        fig.update_xaxes(
            rangeslider_visible=False,
            showgrid=True,
            gridwidth=1,
            gridcolor='rgba(255, 255, 255, 0.1)'
        )

        fig.update_yaxes(
            showgrid=True,
            gridwidth=1,
            gridcolor='rgba(255, 255, 255, 0.1)',
            zeroline=False
        )

        # Save to HTML if path provided
        if save_path:
            fig.write_html(save_path)
            logger.info(f"Interactive chart saved to {save_path}")

        return fig

    def plot_volatility_correlation_matrix(self, volatility_data, save_path=None, show=True):
        """
        Create a correlation matrix heatmap for volatility components.

        Parameters:
        -----------
        volatility_data : pd.DataFrame
            DataFrame with volatility data
        save_path : str, optional
            Path to save the figure
        show : bool
            Whether to display the figure

        Returns:
        --------
        matplotlib.figure.Figure
            The created figure
        """
        # Calculate correlation matrix
        corr_matrix = volatility_data.corr()

        # Create figure
        plt.figure(figsize=(10, 8), dpi=self.dpi, facecolor=self.colors['background'])

        # Create custom colormap (from blue to red)
        cmap = LinearSegmentedColormap.from_list(
            'volatility_cmap', ['#4287f5', '#f54242']
        )

        # Plot heatmap
        ax = sns.heatmap(
            corr_matrix,
            annot=True,
            cmap=cmap,
            linewidths=0.5,
            fmt='.2f',
            square=True,
            cbar_kws={'shrink': 0.8}
        )

        # Set title and labels
        plt.title('Volatility Components Correlation Matrix',
                  color=self.colors['text'], fontsize=16, pad=20)

        # Style the chart
        plt.xticks(color=self.colors['text'])
        plt.yticks(color=self.colors['text'])

        # Save figure if path provided
        if save_path:
            plt.savefig(save_path, dpi=self.dpi, bbox_inches='tight')
            logger.info(f"Correlation matrix saved to {save_path}")

        # Show or close figure
        if show:
            plt.show()
        else:
            plt.close()

        return plt.gcf()

    def plot_regime_distribution(self, regime_data, time_period='all', save_path=None, show=True):
        """
        Create a pie chart showing the distribution of market regimes.

        Parameters:
        -----------
        regime_data : pd.DataFrame
            DataFrame with regime data (must have 'regime' column)
        time_period : str
            Time period to analyze ('all', '1m', '3m', '6m', '1y')
        save_path : str, optional
            Path to save the figure
        show : bool
            Whether to display the figure

        Returns:
        --------
        matplotlib.figure.Figure
            The created figure
        """
        if 'regime' not in regime_data.columns:
            raise ValueError("regime_data must have a 'regime' column")

        # Filter by time period
        if time_period != 'all':
            end_date = regime_data.index[-1]
            if time_period == '1m':
                start_date = end_date - pd.DateOffset(months=1)
            elif time_period == '3m':
                start_date = end_date - pd.DateOffset(months=3)
            elif time_period == '6m':
                start_date = end_date - pd.DateOffset(months=6)
            elif time_period == '1y':
                start_date = end_date - pd.DateOffset(years=1)
            else:
                raise ValueError(f"Invalid time_period: {time_period}")

            regime_data = regime_data[(regime_data.index >= start_date) &
                                      (regime_data.index <= end_date)]

        # Count regime occurrences
        regime_counts = regime_data['regime'].value_counts()
        regime_pct = regime_counts / regime_counts.sum() * 100

        # Create regime colors mapping
        regimes = regime_counts.index
        n_regimes = len(regimes)
        regime_colors = [self.colors.get(f'regime_{i + 1}', f'C{i}') for i in range(n_regimes)]

        # Create figure
        plt.figure(figsize=(10, 8), dpi=self.dpi, facecolor=self.colors['background'])

        # Create pie chart
        wedges, texts, autotexts = plt.pie(
            regime_counts,
            labels=regimes,
            autopct='%1.1f%%',
            startangle=90,
            colors=regime_colors,
            wedgeprops={'edgecolor': 'w', 'linewidth': 1, 'antialiased': True}
        )

        # Style the text
        for text in texts:
            text.set_color(self.colors['text'])
        for autotext in autotexts:
            autotext.set_color('white')
            autotext.set_fontsize(10)

        # Add title
        title = f'Market Regime Distribution ({time_period})' if time_period != 'all' else 'Market Regime Distribution'
        plt.title(title, color=self.colors['text'], fontsize=16, pad=20)

        # Add additional statistics as text
        plt.figtext(
            0.5, 0.02,
            f"Total periods: {len(regime_data)}\n"
            f"Most frequent: {regime_counts.index[0]} ({regime_pct.iloc[0]:.1f}%)\n"
            f"Least frequent: {regime_counts.index[-1]} ({regime_pct.iloc[-1]:.1f}%)",
            ha='center',
            color=self.colors['text'],
            fontsize=12
        )

        # Save figure if path provided
        if save_path:
            plt.savefig(save_path, dpi=self.dpi, bbox_inches='tight')
            logger.info(f"Regime distribution saved to {save_path}")

        # Show or close figure
        if show:
            plt.show()
        else:
            plt.close()

        return plt.gcf()

    def plot_volatility_forecast(self, actual_volatility, predicted_volatility,
                                 confidence_intervals=None, save_path=None, show=True):
        """
        Create a chart showing actual vs predicted volatility with confidence intervals.

        Parameters:
        -----------
        actual_volatility : pd.Series
            Series with actual volatility values
        predicted_volatility : pd.Series
            Series with predicted volatility values
        confidence_intervals : dict, optional
            Dictionary with confidence interval series (e.g., {'95%': (lower, upper)})
        save_path : str, optional
            Path to save the figure
        show : bool
            Whether to display the figure

        Returns:
        --------
        matplotlib.figure.Figure
            The created figure
        """
        # Create figure
        fig, ax = plt.subplots(figsize=(self.width, self.height), dpi=self.dpi,
                               facecolor=self.colors['background'])

        # Plot actual volatility
        ax.plot(actual_volatility.index, actual_volatility,
                color=self.colors['volatility'], linewidth=2,
                label='Actual Volatility')

        # Plot predicted volatility
        ax.plot(predicted_volatility.index, predicted_volatility,
                color=self.colors['predicted_volatility'], linewidth=2, linestyle='--',
                label='Predicted Volatility')

        # Plot confidence intervals if provided
        if confidence_intervals:
            for label, (lower, upper) in confidence_intervals.items():
                ax.fill_between(
                    lower.index, lower, upper,
                    alpha=0.2,
                    color=self.colors['predicted_volatility'],
                    label=f'{label} Confidence Interval'
                )

        # Add vertical line at last actual data point if forecasting into the future
        last_actual_date = actual_volatility.index[-1]
        if predicted_volatility.index[-1] > last_actual_date:
            ax.axvline(
                x=last_actual_date,
                color='white',
                linestyle=':',
                alpha=0.7,
                label='Forecast Start'
            )

        # Set title and labels
        ax.set_title('Volatility Forecast', color=self.colors['text'], fontsize=16)
        ax.set_xlabel('Date', color=self.colors['text'], fontsize=12)
        ax.set_ylabel('Volatility', color=self.colors['text'], fontsize=12)

        # Style the chart
        ax.grid(True, alpha=0.3, color=self.colors['grid'])
        ax.tick_params(colors=self.colors['text'])
        ax.legend(loc='upper left', facecolor=self.colors['background'],
                  edgecolor=self.colors['grid'], labelcolor=self.colors['text'])

        # Format x-axis
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')

        # Save figure if path provided
        if save_path:
            plt.savefig(save_path, dpi=self.dpi, bbox_inches='tight')
            logger.info(f"Volatility forecast saved to {save_path}")

        # Show or close figure
        if show:
            plt.tight_layout()
            plt.show()
        else:
            plt.close(fig)

        return fig

    def create_report(self, price_data, volatility_data, predicted_volatility=None,
                      regime_data=None, signals=None, report_dir=None):
        """
        Create a comprehensive volatility analysis report with multiple charts.

        Parameters:
        -----------
        price_data : pd.DataFrame
            DataFrame with price data
        volatility_data : pd.DataFrame
            DataFrame with volatility data
        predicted_volatility : pd.Series, optional
            Series with predicted volatility values
        regime_data : pd.DataFrame, optional
            DataFrame with regime data
        signals : pd.DataFrame, optional
            DataFrame with trading signals
        report_dir : str, optional
            Directory to save the report

        Returns:
        --------
        dict
            Dictionary with paths to all generated charts
        """
        # Create report directory if provided
        if report_dir:
            os.makedirs(report_dir, exist_ok=True)
        else:
            report_dir = self.output_dir

        # Generate timestamp for filenames
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

        # Dictionary to store chart paths
        chart_paths = {}

        # 1. Main volatility indicator chart
        main_chart_path = os.path.join(report_dir, f'volatility_chart_{timestamp}.png')
        self.plot_volatility_indicator(
            price_data, volatility_data, predicted_volatility, regime_data, signals,
            save_path=main_chart_path, show=False
        )
        chart_paths['main_chart'] = main_chart_path

        # 2. Interactive chart
        interactive_chart_path = os.path.join(report_dir, f'interactive_chart_{timestamp}.html')
        self.create_interactive_volatility_chart(
            price_data, volatility_data, predicted_volatility, regime_data, signals,
            save_path=interactive_chart_path
        )
        chart_paths['interactive_chart'] = interactive_chart_path

        # 3. Correlation matrix
        if volatility_data.shape[1] > 1:
            corr_matrix_path = os.path.join(report_dir, f'correlation_matrix_{timestamp}.png')
            self.plot_volatility_correlation_matrix(
                volatility_data, save_path=corr_matrix_path, show=False
            )
            chart_paths['correlation_matrix'] = corr_matrix_path

        # 4. Regime distribution
        if regime_data is not None and 'regime' in regime_data.columns:
            regime_dist_path = os.path.join(report_dir, f'regime_distribution_{timestamp}.png')
            self.plot_regime_distribution(
                regime_data, time_period='all', save_path=regime_dist_path, show=False
            )
            chart_paths['regime_distribution'] = regime_dist_path

        # 5. Volatility forecast
        if predicted_volatility is not None:
            forecast_path = os.path.join(report_dir, f'volatility_forecast_{timestamp}.png')
            # For the forecast chart, use the last portion of actual data plus predictions
            lookback = min(len(price_data) // 4, 30)  # Use last 25% or 30 days, whichever is smaller
            last_actual = volatility_data.iloc[-lookback:][
                'composite_vol'] if 'composite_vol' in volatility_data.columns else volatility_data.iloc[
                                                                                    -lookback:].iloc[:, 0]

            self.plot_volatility_forecast(
                last_actual, predicted_volatility,
                save_path=forecast_path, show=False
            )
            chart_paths['forecast'] = forecast_path

        logger.info(f"Generated {len(chart_paths)} charts for the report in {report_dir}")
        return chart_paths