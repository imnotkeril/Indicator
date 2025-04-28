"""
Main entry point for the Progressive Adaptive Volatility Indicator.
"""
import os
import sys
import argparse
import yaml
import logging
import signal
import threading
import time
from datetime import datetime, timedelta

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, project_root)
from crypto_volatility_indicator.utils.logger import get_logger, setup_logger
from crypto_volatility_indicator.engine.indicators.composite import ProgressiveAdaptiveVolatilityIndicator
from crypto_volatility_indicator.visualization.dashboard import VolatilityDashboard
from crypto_volatility_indicator.visualization.alerts import AlertManager
from crypto_volatility_indicator.utils.config import load_config
from crypto_volatility_indicator.api.rest import VolatilityAPI
from crypto_volatility_indicator.api.websocket import VolatilityWebSocket

logger = get_logger(__name__)


def signal_handler(sig, frame):
    """Handle termination signals."""
    logger.info("Received signal to terminate")
    sys.exit(0)


def run_dashboard(volatility_indicator, config):
    """Run the dashboard."""
    dashboard = VolatilityDashboard(volatility_indicator, config.get('dashboard', {}))
    dashboard.run()


def run_rest_api(volatility_indicator, config):
    """Run the REST API."""
    api = VolatilityAPI(volatility_indicator, config.get('api', {}).get('rest', {}))
    api.run()


def run_websocket_api(volatility_indicator, config):
    """Run the WebSocket API."""
    websocket = VolatilityWebSocket(volatility_indicator, config.get('api', {}).get('websocket', {}))
    websocket.run()


def run_alert_manager(volatility_indicator, config):
    """Run the alert manager."""
    alert_manager = AlertManager(config.get('alerts', {}))
    alert_manager.start_monitoring(volatility_indicator)

    # Keep thread running
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        alert_manager.stop_monitoring()


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Progressive Adaptive Volatility Indicator')

    parser.add_argument(
        '-c', '--config',
        type=str,
        default='config.yaml',
        help='Path to configuration file'
    )

    parser.add_argument(
        '-m', '--mode',
        type=str,
        choices=['indicator', 'dashboard', 'rest-api', 'websocket-api', 'alerts', 'all'],
        default='indicator',
        help='Mode to run'
    )

    parser.add_argument(
        '-a', '--assets',
        type=str,
        nargs='+',
        help='Assets to monitor (overrides config)'
    )

    parser.add_argument(
        '-v', '--verbose',
        action='store_true',
        help='Enable verbose logging'
    )

    parser.add_argument(
        '-o', '--output',
        type=str,
        help='Output directory for logs and charts'
    )

    parser.add_argument(
        '--test',
        action='store_true',
        help='Run in test mode with backtest data'
    )

    parser.add_argument(
        '--test-period',
        type=str,
        default='1M',
        help='Test period (1D, 1W, 1M, 3M, 6M, 1Y)'
    )

    return parser.parse_args()


def setup_logging(config, args):
    """Setup logging."""
    log_level = logging.DEBUG if args.verbose else logging.INFO
    log_dir = args.output or config.get('logging', {}).get('directory', 'logs')

    os.makedirs(log_dir, exist_ok=True)

    log_file = os.path.join(log_dir, f"volatility_indicator_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")

    setup_logger({
        'logging': {
            'level': logging.getLevelName(log_level),
            'file': os.path.basename(log_file),
            'console': True
        }
    })


def get_test_data_period(period_str):
    """Convert period string to start and end dates."""
    end_date = datetime.now()

    if period_str == '1D':
        start_date = end_date - timedelta(days=1)
    elif period_str == '1W':
        start_date = end_date - timedelta(weeks=1)
    elif period_str == '1M':
        start_date = end_date - timedelta(days=30)
    elif period_str == '3M':
        start_date = end_date - timedelta(days=90)
    elif period_str == '6M':
        start_date = end_date - timedelta(days=180)
    elif period_str == '1Y':
        start_date = end_date - timedelta(days=365)
    else:
        # Default to 1 month
        start_date = end_date - timedelta(days=30)

    return start_date, end_date


def main():
    """Main entry point."""
    # Parse arguments
    args = parse_args()

    # Load configuration
    config = load_config(args.config)

    # Setup logging
    setup_logging(config, args)

    # Log startup information
    logger.info(f"Starting Progressive Adaptive Volatility Indicator in {args.mode} mode")
    logger.info(f"Configuration file: {args.config}")

    # Register signal handlers
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    # Override config with command line arguments
    if args.assets:
        config['assets'] = args.assets

    # Initialize indicator
    volatility_indicator = ProgressiveAdaptiveVolatilityIndicator(config)

    # Initialize assets
    if args.test:
        logger.info(f"Running in test mode with period {args.test_period}")
        start_date, end_date = get_test_data_period(args.test_period)
        volatility_indicator.initialize_with_test_data(start_date, end_date)
    else:
        volatility_indicator.initialize_assets()

    # Run in specified mode
    if args.mode == 'indicator':
        # Just run the indicator in the current thread
        try:
            logger.info("Running indicator in standalone mode")
            while True:
                volatility_indicator.update()
                time.sleep(config.get('update_interval', 60))
        except KeyboardInterrupt:
            logger.info("Indicator stopped by user")

    elif args.mode == 'dashboard':
        # Run the dashboard
        run_dashboard(volatility_indicator, config)

    elif args.mode == 'rest-api':
        # Run the REST API
        run_rest_api(volatility_indicator, config)

    elif args.mode == 'websocket-api':
        # Run the WebSocket API
        run_websocket_api(volatility_indicator, config)

    elif args.mode == 'alerts':
        # Run the alert manager
        run_alert_manager(volatility_indicator, config)

    elif args.mode == 'all':
        # Run everything in separate threads
        logger.info("Running all components")

        # Start indicator update thread
        indicator_thread = threading.Thread(
            target=lambda: volatility_indicator.start_background_update(
                config.get('update_interval', 60)
            )
        )
        indicator_thread.daemon = True
        indicator_thread.start()

        # Start dashboard in a separate thread
        dashboard_config = config.get('dashboard', {})
        if dashboard_config.get('enabled', True):
            dashboard_thread = threading.Thread(
                target=run_dashboard,
                args=(volatility_indicator, config)
            )
            dashboard_thread.daemon = True
            dashboard_thread.start()

        # Start REST API in a separate thread
        rest_api_config = config.get('api', {}).get('rest', {})
        if rest_api_config.get('enabled', True):
            rest_api_thread = threading.Thread(
                target=run_rest_api,
                args=(volatility_indicator, config)
            )
            rest_api_thread.daemon = True
            rest_api_thread.start()

        # Start WebSocket API in a separate thread
        websocket_config = config.get('api', {}).get('websocket', {})
        if websocket_config.get('enabled', True):
            websocket_thread = threading.Thread(
                target=run_websocket_api,
                args=(volatility_indicator, config)
            )
            websocket_thread.daemon = True
            websocket_thread.start()

        # Start Alert Manager in a separate thread
        alerts_config = config.get('alerts', {})
        if alerts_config.get('enabled', True):
            alerts_thread = threading.Thread(
                target=run_alert_manager,
                args=(volatility_indicator, config)
            )
            alerts_thread.daemon = True
            alerts_thread.start()

        # Keep main thread running
        try:
            logger.info("All components started successfully")

            # Print URLs for services
            dashboard_url = f"http://{dashboard_config.get('host', '127.0.0.1')}:{dashboard_config.get('port', 8050)}"
            rest_api_url = f"http://{rest_api_config.get('host', '0.0.0.0')}:{rest_api_config.get('port', 5000)}"
            websocket_url = f"ws://{websocket_config.get('host', '0.0.0.0')}:{websocket_config.get('port', 8765)}"

            logger.info(f"Dashboard URL: {dashboard_url}")
            logger.info(f"REST API URL: {rest_api_url}")
            logger.info(f"WebSocket URL: {websocket_url}")

            # Wait indefinitely
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            logger.info("Stopping all components")

            # Stop background threads
            volatility_indicator.stop_background_update()

            if alerts_config.get('enabled', True):
                # Get AlertManager instance from the thread's target function scope
                alert_manager = AlertManager(config.get('alerts', {}))
                alert_manager.stop_monitoring()

            logger.info("All components stopped")


if __name__ == "__main__":
    main()