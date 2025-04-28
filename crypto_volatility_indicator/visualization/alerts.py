"""
Module for generating and sending alerts based on volatility indicators.
"""

import os
import sys
import numpy as np
import pandas as pd
import json
import smtplib
import requests
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
import logging
from datetime import datetime
import time
import threading
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, project_root)
from crypto_volatility_indicator.utils.logger import get_logger

logger = get_logger(__name__)


class AlertType:
    """Alert type constants."""
    VOLATILITY_BREAKOUT = "volatility_breakout"
    VOLATILITY_CONTRACTION = "volatility_contraction"
    REGIME_CHANGE = "regime_change"
    PREDICTION_ALERT = "prediction_alert"
    SIGNAL_ALERT = "signal_alert"
    CUSTOM_ALERT = "custom_alert"


class AlertPriority:
    """Alert priority constants."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class Alert:
    """Class representing an alert."""

    def __init__(self, alert_type, message, asset, timestamp=None, priority=None,
                 data=None, threshold=None, actual_value=None):
        """
        Initialize an alert.

        Parameters:
        -----------
        alert_type : str
            Type of the alert (e.g., 'volatility_breakout')
        message : str
            Alert message
        asset : str
            Asset symbol (e.g., 'BTC/USD')
        timestamp : datetime, optional
            Alert timestamp (defaults to current time)
        priority : str, optional
            Alert priority (defaults to 'medium')
        data : dict, optional
            Additional data for the alert
        threshold : float, optional
            Threshold value that triggered the alert
        actual_value : float, optional
            Actual value that crossed the threshold
        """
        self.alert_type = alert_type
        self.message = message
        self.asset = asset
        self.timestamp = timestamp or datetime.now()
        self.priority = priority or AlertPriority.MEDIUM
        self.data = data or {}
        self.threshold = threshold
        self.actual_value = actual_value
        self.id = f"{self.alert_type}_{self.asset}_{int(time.time() * 1000)}"

    def to_dict(self):
        """Convert alert to dictionary."""
        return {
            'id': self.id,
            'type': self.alert_type,
            'message': self.message,
            'asset': self.asset,
            'timestamp': self.timestamp.isoformat(),
            'priority': self.priority,
            'data': self.data,
            'threshold': self.threshold,
            'actual_value': self.actual_value
        }

    def to_json(self):
        """Convert alert to JSON string."""
        return json.dumps(self.to_dict(), default=str)

    @classmethod
    def from_dict(cls, data):
        """Create an alert from a dictionary."""
        # Convert timestamp string to datetime
        if 'timestamp' in data and isinstance(data['timestamp'], str):
            data['timestamp'] = datetime.fromisoformat(data['timestamp'].replace('Z', '+00:00'))

        return cls(
            alert_type=data.get('type'),
            message=data.get('message'),
            asset=data.get('asset'),
            timestamp=data.get('timestamp'),
            priority=data.get('priority'),
            data=data.get('data'),
            threshold=data.get('threshold'),
            actual_value=data.get('actual_value')
        )

    @classmethod
    def from_json(cls, json_str):
        """Create an alert from a JSON string."""
        data = json.loads(json_str)
        return cls.from_dict(data)


class AlertNotifier:
    """Class for sending alert notifications through various channels."""

    def __init__(self, config=None):
        """
        Initialize the AlertNotifier.

        Parameters:
        -----------
        config : dict, optional
            Configuration parameters for notification channels
        """
        self.config = config or {}

        # Initialize notification channels
        self.email_config = self.config.get('email', {})
        self.webhook_config = self.config.get('webhook', {})
        self.telegram_config = self.config.get('telegram', {})

        # Set up logged alerts storage
        self.alerts_log_file = self.config.get('alerts_log_file', 'alerts.log')

    def send_email(self, alert):
        """
        Send an alert via email.

        Parameters:
        -----------
        alert : Alert
            The alert to send

        Returns:
        --------
        bool
            True if email was sent successfully, False otherwise
        """
        if not self.email_config.get('enabled', False):
            logger.debug("Email notifications not enabled")
            return False

        try:
            # Get email configuration
            smtp_server = self.email_config.get('smtp_server')
            smtp_port = self.email_config.get('smtp_port', 587)
            username = self.email_config.get('username')
            password = self.email_config.get('password')

            sender = self.email_config.get('sender')
            recipients = self.email_config.get('recipients', [])

            if not all([smtp_server, username, password, sender, recipients]):
                logger.error("Email configuration incomplete")
                return False

            # Create message
            msg = MIMEMultipart()
            msg['From'] = sender
            msg['To'] = ', '.join(recipients)
            msg['Subject'] = f"[{alert.priority.upper()}] {alert.alert_type} Alert: {alert.asset}"

            # Create email body with HTML formatting
            body = f"""
            <html>
            <head>
                <style>
                    body {{ font-family: Arial, sans-serif; }}
                    .alert {{ padding: 15px; border-radius: 5px; margin-bottom: 20px; }}
                    .low {{ background-color: #e8f5e9; border-left: 5px solid #4caf50; }}
                    .medium {{ background-color: #fff8e1; border-left: 5px solid #ffc107; }}
                    .high {{ background-color: #fff3e0; border-left: 5px solid #ff9800; }}
                    .critical {{ background-color: #ffebee; border-left: 5px solid #f44336; }}
                    .details {{ margin-top: 10px; }}
                    table {{ border-collapse: collapse; width: 100%; }}
                    th, td {{ padding: 8px; text-align: left; border-bottom: 1px solid #ddd; }}
                </style>
            </head>
            <body>
                <div class="alert {alert.priority}">
                    <h2>{alert.message}</h2>
                    <p>Asset: <strong>{alert.asset}</strong></p>
                    <p>Time: {alert.timestamp.strftime('%Y-%m-%d %H:%M:%S')}</p>

                    <div class="details">
                        <h3>Alert Details:</h3>
                        <table>
                            <tr>
                                <th>Type</th>
                                <td>{alert.alert_type}</td>
                            </tr>
                            <tr>
                                <th>Priority</th>
                                <td>{alert.priority}</td>
                            </tr>
            """

            # Add threshold and actual value if available
            if alert.threshold is not None:
                body += f"""
                            <tr>
                                <th>Threshold</th>
                                <td>{alert.threshold}</td>
                            </tr>
                """

            if alert.actual_value is not None:
                body += f"""
                            <tr>
                                <th>Actual Value</th>
                                <td>{alert.actual_value}</td>
                            </tr>
                """

            # Add any additional data
            for key, value in alert.data.items():
                body += f"""
                            <tr>
                                <th>{key}</th>
                                <td>{value}</td>
                            </tr>
                """

            # Close HTML
            body += """
                        </table>
                    </div>
                </div>
            </body>
            </html>
            """

            msg.attach(MIMEText(body, 'html'))

            # Connect to SMTP server and send email
            with smtplib.SMTP(smtp_server, smtp_port) as server:
                server.starttls()
                server.login(username, password)
                server.send_message(msg)

            logger.info(f"Email alert sent: {alert.alert_type} for {alert.asset}")
            return True

        except Exception as e:
            logger.error(f"Failed to send email alert: {str(e)}")
            return False

    def send_webhook(self, alert):
        """
        Send an alert via webhook.

        Parameters:
        -----------
        alert : Alert
            The alert to send

        Returns:
        --------
        bool
            True if webhook request was successful, False otherwise
        """
        if not self.webhook_config.get('enabled', False):
            logger.debug("Webhook notifications not enabled")
            return False

        try:
            # Get webhook configuration
            webhook_url = self.webhook_config.get('url')
            headers = self.webhook_config.get('headers', {})

            if not webhook_url:
                logger.error("Webhook URL not configured")
                return False

            # Prepare payload
            payload = alert.to_dict()

            # Add webhook specific fields if needed
            if 'additional_fields' in self.webhook_config:
                payload.update(self.webhook_config['additional_fields'])

            # Send request
            response = requests.post(
                webhook_url,
                json=payload,
                headers=headers
            )

            # Check response
            if response.status_code >= 200 and response.status_code < 300:
                logger.info(f"Webhook alert sent: {alert.alert_type} for {alert.asset}")
                return True
            else:
                logger.error(f"Webhook request failed with status code {response.status_code}: {response.text}")
                return False

        except Exception as e:
            logger.error(f"Failed to send webhook alert: {str(e)}")
            return False

    def send_telegram(self, alert):
        """
        Send an alert via Telegram.

        Parameters:
        -----------
        alert : Alert
            The alert to send

        Returns:
        --------
        bool
            True if Telegram message was sent successfully, False otherwise
        """
        if not self.telegram_config.get('enabled', False):
            logger.debug("Telegram notifications not enabled")
            return False

        try:
            # Get Telegram configuration
            bot_token = self.telegram_config.get('bot_token')
            chat_id = self.telegram_config.get('chat_id')

            if not all([bot_token, chat_id]):
                logger.error("Telegram configuration incomplete")
                return False

            # Create message
            priority_emoji = {
                AlertPriority.LOW: "🟢",
                AlertPriority.MEDIUM: "🟡",
                AlertPriority.HIGH: "🟠",
                AlertPriority.CRITICAL: "🔴"
            }.get(alert.priority, "⚪")

            message = f"{priority_emoji} *{alert.alert_type.upper()}*\n\n"
            message += f"*{alert.message}*\n\n"
            message += f"Asset: `{alert.asset}`\n"
            message += f"Time: `{alert.timestamp.strftime('%Y-%m-%d %H:%M:%S')}`\n"

            # Add threshold and actual value if available
            if alert.threshold is not None:
                message += f"Threshold: `{alert.threshold}`\n"

            if alert.actual_value is not None:
                message += f"Actual Value: `{alert.actual_value}`\n"

            # Add any additional data
            if alert.data:
                message += "\nAdditional Info:\n"
                for key, value in alert.data.items():
                    message += f"{key}: `{value}`\n"

            # Send message
            url = f"https://api.telegram.org/bot{bot_token}/sendMessage"
            payload = {
                'chat_id': chat_id,
                'text': message,
                'parse_mode': 'Markdown'
            }

            response = requests.post(url, json=payload)

            # Check response
            if response.status_code == 200:
                logger.info(f"Telegram alert sent: {alert.alert_type} for {alert.asset}")
                return True
            else:
                logger.error(f"Telegram request failed with status code {response.status_code}: {response.text}")
                return False

        except Exception as e:
            logger.error(f"Failed to send Telegram alert: {str(e)}")
            return False

    def log_alert(self, alert):
        """
        Log an alert to file.

        Parameters:
        -----------
        alert : Alert
            The alert to log

        Returns:
        --------
        bool
            True if alert was logged successfully, False otherwise
        """
        try:
            # Create directory for log file if it doesn't exist
            log_dir = os.path.dirname(self.alerts_log_file)
            if log_dir and not os.path.exists(log_dir):
                os.makedirs(log_dir)

            # Append alert to log file
            with open(self.alerts_log_file, 'a') as f:
                f.write(alert.to_json() + '\n')

            logger.debug(f"Alert logged to {self.alerts_log_file}: {alert.alert_type} for {alert.asset}")
            return True

        except Exception as e:
            logger.error(f"Failed to log alert: {str(e)}")
            return False

    def send_alert(self, alert):
        """
        Send an alert through all configured channels.

        Parameters:
        -----------
        alert : Alert
            The alert to send

        Returns:
        --------
        dict
            Dictionary with status for each channel
        """
        results = {}

        # Log alert
        results['log'] = self.log_alert(alert)

        # Send via email if configured
        if self.email_config.get('enabled', False):
            # For low priority, only send if explicitly configured
            if alert.priority == AlertPriority.LOW and not self.email_config.get('send_low_priority', False):
                results['email'] = 'skipped'
            else:
                results['email'] = self.send_email(alert)

        # Send via webhook if configured
        if self.webhook_config.get('enabled', False):
            # For low priority, only send if explicitly configured
            if alert.priority == AlertPriority.LOW and not self.webhook_config.get('send_low_priority', False):
                results['webhook'] = 'skipped'
            else:
                results['webhook'] = self.send_webhook(alert)

        # Send via Telegram if configured
        if self.telegram_config.get('enabled', False):
            # For low priority, only send if explicitly configured
            if alert.priority == AlertPriority.LOW and not self.telegram_config.get('send_low_priority', False):
                results['telegram'] = 'skipped'
            else:
                results['telegram'] = self.send_telegram(alert)

        return results

class AlertManager:
    """
    Class for managing volatility alerts and notifications.

    This class monitors volatility data and generates alerts
    based on configurable conditions.
    """

    def __init__(self, config=None):
        """
        Initialize the AlertManager.

        Parameters:
        -----------
        config : dict, optional
            Configuration parameters
        """
        self.config = config or {}

        # Initialize alert notifier
        self.notifier = AlertNotifier(self.config.get('notifications', {}))

        # Initialize alert thresholds
        self.thresholds = self.config.get('thresholds', {
            'volatility_breakout': 2.0,
            'volatility_contraction': 0.5,
            'prediction_threshold': 1.5
        })

        # Initialize alert storage
        self.alerts = []
        self.max_stored_alerts = self.config.get('max_stored_alerts', 1000)

        # Initialize alert suppression
        self.suppression_time = self.config.get('suppression_time', {
            AlertType.VOLATILITY_BREAKOUT: 3600,  # 1 hour
            AlertType.VOLATILITY_CONTRACTION: 3600,  # 1 hour
            AlertType.REGIME_CHANGE: 86400,  # 24 hours
            AlertType.PREDICTION_ALERT: 3600,  # 1 hour
            AlertType.SIGNAL_ALERT: 1800,  # 30 minutes
            AlertType.CUSTOM_ALERT: 300  # 5 minutes
        })

        # Store last alert times to implement suppression
        self.last_alert_times = {}

        # Initialize alert threads
        self.monitoring_active = False
        self.monitor_thread = None

    def check_volatility_breakout(self, asset, current_vol, historical_vols):
        """
        Check for volatility breakout.

        Parameters:
        -----------
        asset : str
            Asset symbol
        current_vol : float
            Current volatility value
        historical_vols : pd.Series or list
            Historical volatility values

        Returns:
        --------
        Alert or None
            Alert if breakout detected, None otherwise
        """
        if len(historical_vols) < 10:
            logger.warning(f"Not enough historical data for {asset} to detect breakout")
            return None

        # Convert to numpy array if needed
        if isinstance(historical_vols, pd.Series):
            historical_vols = historical_vols.values

        # Calculate mean and standard deviation
        mean_vol = np.mean(historical_vols)
        std_vol = np.std(historical_vols)

        # Get threshold
        threshold = mean_vol + self.thresholds['volatility_breakout'] * std_vol

        # Check for breakout
        if current_vol > threshold:
            # Calculate how many standard deviations away
            std_devs = (current_vol - mean_vol) / std_vol

            # Create alert
            alert = Alert(
                alert_type=AlertType.VOLATILITY_BREAKOUT,
                message=f"Volatility breakout detected for {asset}",
                asset=asset,
                priority=AlertPriority.HIGH if std_devs > 3 else AlertPriority.MEDIUM,
                data={
                    'std_deviations': round(std_devs, 2),
                    'mean_volatility': round(mean_vol, 6),
                    'historical_std': round(std_vol, 6)
                },
                threshold=round(threshold, 6),
                actual_value=round(current_vol, 6)
            )

            return alert

        return None

    def check_volatility_contraction(self, asset, current_vol, historical_vols):
        """
        Check for volatility contraction.

        Parameters:
        -----------
        asset : str
            Asset symbol
        current_vol : float
            Current volatility value
        historical_vols : pd.Series or list
            Historical volatility values

        Returns:
        --------
        Alert or None
            Alert if contraction detected, None otherwise
        """
        if len(historical_vols) < 10:
            logger.warning(f"Not enough historical data for {asset} to detect contraction")
            return None

        # Convert to numpy array if needed
        if isinstance(historical_vols, pd.Series):
            historical_vols = historical_vols.values

        # Calculate mean and standard deviation
        mean_vol = np.mean(historical_vols)
        std_vol = np.std(historical_vols)

        # Get threshold
        threshold = mean_vol - self.thresholds['volatility_contraction'] * std_vol

        # Check for contraction
        if current_vol < threshold:
            # Calculate how many standard deviations away
            std_devs = (mean_vol - current_vol) / std_vol

            # Create alert
            alert = Alert(
                alert_type=AlertType.VOLATILITY_CONTRACTION,
                message=f"Volatility contraction detected for {asset}",
                asset=asset,
                priority=AlertPriority.MEDIUM,
                data={
                    'std_deviations': round(std_devs, 2),
                    'mean_volatility': round(mean_vol, 6),
                    'historical_std': round(std_vol, 6)
                },
                threshold=round(threshold, 6),
                actual_value=round(current_vol, 6)
            )

            return alert

        return None

    def check_regime_change(self, asset, current_regime, previous_regime):
        """
        Check for market regime change.

        Parameters:
        -----------
        asset : str
            Asset symbol
        current_regime : str
            Current market regime
        previous_regime : str
            Previous market regime

        Returns:
        --------
        Alert or None
            Alert if regime change detected, None otherwise
        """
        if current_regime == previous_regime:
            return None

        # Determine priority based on the new regime
        priority_map = {
            'low_vol': AlertPriority.LOW,
            'normal_vol': AlertPriority.LOW,
            'high_vol': AlertPriority.MEDIUM,
            'extreme_vol': AlertPriority.HIGH
        }

        priority = priority_map.get(current_regime, AlertPriority.MEDIUM)

        # Create alert
        alert = Alert(
            alert_type=AlertType.REGIME_CHANGE,
            message=f"Market regime change for {asset}: {previous_regime} -> {current_regime}",
            asset=asset,
            priority=priority,
            data={
                'previous_regime': previous_regime,
                'new_regime': current_regime
            }
        )

        return alert

    def check_prediction_alert(self, asset, predicted_vol, current_vol):
        """
        Check for significant predicted volatility change.

        Parameters:
        -----------
        asset : str
            Asset symbol
        predicted_vol : float
            Predicted volatility value
        current_vol : float
            Current volatility value

        Returns:
        --------
        Alert or None
            Alert if significant change predicted, None otherwise
        """
        # Calculate percent change
        percent_change = (predicted_vol - current_vol) / current_vol * 100

        # Get threshold
        threshold = self.thresholds['prediction_threshold']

        # Only alert on significant changes
        if abs(percent_change) < threshold:
            return None

        # Determine message and priority based on direction
        if percent_change > 0:
            message = f"Predicted volatility increase for {asset}"
            priority = AlertPriority.MEDIUM if percent_change > 2 * threshold else AlertPriority.LOW
        else:
            message = f"Predicted volatility decrease for {asset}"
            priority = AlertPriority.LOW

        # Create alert
        alert = Alert(
            alert_type=AlertType.PREDICTION_ALERT,
            message=message,
            asset=asset,
            priority=priority,
            data={
                'percent_change': round(percent_change, 2),
                'current_volatility': round(current_vol, 6)
            },
            threshold=threshold,
            actual_value=round(predicted_vol, 6)
        )

        return alert

    def create_signal_alert(self, asset, signal_type, signal_data):
        """
        Create an alert from a trading signal.

        Parameters:
        -----------
        asset : str
            Asset symbol
        signal_type : str
            Type of trading signal
        signal_data : dict
            Signal data

        Returns:
        --------
        Alert
            The created alert
        """
        # Map signal types to priorities
        priority_map = {
            'entry_long': AlertPriority.HIGH,
            'entry_short': AlertPriority.HIGH,
            'exit_long': AlertPriority.MEDIUM,
            'exit_short': AlertPriority.MEDIUM,
            'reduce_position': AlertPriority.LOW,
            'increase_position': AlertPriority.LOW,
            'volatility_breakout': AlertPriority.MEDIUM,
            'volatility_contraction': AlertPriority.LOW,
            'regime_change': AlertPriority.MEDIUM
        }

        priority = priority_map.get(signal_type, AlertPriority.MEDIUM)

        # Create message based on signal type
        message = f"Trading signal: {signal_type} for {asset}"

        # Create alert
        alert = Alert(
            alert_type=AlertType.SIGNAL_ALERT,
            message=message,
            asset=asset,
            priority=priority,
            data=signal_data
        )

        return alert

    def create_custom_alert(self, asset, message, data=None, priority=None):
        """
        Create a custom alert.

        Parameters:
        -----------
        asset : str
            Asset symbol
        message : str
            Alert message
        data : dict, optional
            Additional data
        priority : str, optional
            Alert priority

        Returns:
        --------
        Alert
            The created alert
        """
        return Alert(
            alert_type=AlertType.CUSTOM_ALERT,
            message=message,
            asset=asset,
            priority=priority or AlertPriority.MEDIUM,
            data=data or {}
        )

    def should_suppress_alert(self, alert):
        """
        Check if an alert should be suppressed based on time since last alert.

        Parameters:
        -----------
        alert : Alert
            The alert to check

        Returns:
        --------
        bool
            True if alert should be suppressed, False otherwise
        """
        # Create key for this type of alert and asset
        key = f"{alert.alert_type}_{alert.asset}"

        # Get last alert time
        last_time = self.last_alert_times.get(key)

        if last_time is None:
            # No previous alert, don't suppress
            return False

        # Get suppression time for this alert type
        suppression_seconds = self.suppression_time.get(
            alert.alert_type,
            self.suppression_time.get(AlertType.CUSTOM_ALERT, 300)
        )

        # Check if enough time has passed
        now = datetime.now().timestamp()
        time_since_last = now - last_time

        return time_since_last < suppression_seconds

    def process_alert(self, alert):
        """
        Process an alert: store, send notifications, and update last alert time.

        Parameters:
        -----------
        alert : Alert
            The alert to process

        Returns:
        --------
        dict
            Status of notification delivery
        """
        # Check for suppression
        if self.should_suppress_alert(alert):
            logger.debug(f"Alert suppressed: {alert.alert_type} for {alert.asset}")
            return {'status': 'suppressed'}

        # Store alert
        self.alerts.append(alert)

        # Trim alerts if needed
        if len(self.alerts) > self.max_stored_alerts:
            self.alerts = self.alerts[-self.max_stored_alerts:]

        # Update last alert time
        key = f"{alert.alert_type}_{alert.asset}"
        self.last_alert_times[key] = datetime.now().timestamp()

        # Send notifications
        results = self.notifier.send_alert(alert)
        results['status'] = 'sent'

        return results

    def start_monitoring(self, volatility_indicator, interval=60):
        """
        Start a background thread to monitor volatility and generate alerts.

        Parameters:
        -----------
        volatility_indicator : object
            Volatility indicator to monitor
        interval : int
            Monitoring interval in seconds

        Returns:
        --------
        bool
            True if monitoring started, False otherwise
        """
        if self.monitoring_active:
            logger.warning("Monitoring is already active")
            return False

        # Set flag
        self.monitoring_active = True

        # Start monitoring thread
        self.monitor_thread = threading.Thread(
            target=self._monitoring_loop,
            args=(volatility_indicator, interval),
            daemon=True
        )
        self.monitor_thread.start()

        logger.info(f"Started alert monitoring with interval {interval}s")
        return True

    def stop_monitoring(self):
        """
        Stop the monitoring thread.

        Returns:
        --------
        bool
            True if monitoring stopped, False otherwise
        """
        if not self.monitoring_active:
            logger.warning("Monitoring is not active")
            return False

        # Set flag to stop
        self.monitoring_active = False

        # Wait for thread to finish
        if self.monitor_thread and self.monitor_thread.is_alive():
            self.monitor_thread.join(timeout=5)

        logger.info("Stopped alert monitoring")
        return True

    def _monitoring_loop(self, volatility_indicator, interval):
        """
        Background monitoring loop.

        Parameters:
        -----------
        volatility_indicator : object
            Volatility indicator to monitor
        interval : int
            Monitoring interval in seconds
        """
        last_regimes = {}

        while self.monitoring_active:
            try:
                # Get current data from indicator
                assets = volatility_indicator.get_monitored_assets()

                for asset in assets:
                    # Get volatility data
                    vol_data = volatility_indicator.get_volatility_data(asset)

                    if vol_data is None or vol_data.empty:
                        continue

                    # Check for breakouts and contractions
                    current_vol = vol_data['composite_vol'].iloc[-1] if 'composite_vol' in vol_data.columns else \
                    vol_data.iloc[-1, 0]
                    historical_vols = vol_data['composite_vol'].iloc[
                                      :-1] if 'composite_vol' in vol_data.columns else vol_data.iloc[:-1, 0]

                    breakout_alert = self.check_volatility_breakout(asset, current_vol, historical_vols)
                    if breakout_alert:
                        self.process_alert(breakout_alert)

                    contraction_alert = self.check_volatility_contraction(asset, current_vol, historical_vols)
                    if contraction_alert:
                        self.process_alert(contraction_alert)

                    # Check for regime changes
                    regime_data = volatility_indicator.get_regime_data(asset)
                    if regime_data is not None and not regime_data.empty and 'regime' in regime_data.columns:
                        current_regime = regime_data['regime'].iloc[-1]
                        previous_regime = last_regimes.get(asset)

                        if previous_regime is not None and current_regime != previous_regime:
                            regime_alert = self.check_regime_change(asset, current_regime, previous_regime)
                            if regime_alert:
                                self.process_alert(regime_alert)

                        # Update stored regime
                        last_regimes[asset] = current_regime

                    # Check predictions
                    prediction_data = volatility_indicator.get_predictions(asset)
                    if prediction_data is not None and not prediction_data.empty:
                        predicted_vol = prediction_data.iloc[-1]
                        prediction_alert = self.check_prediction_alert(asset, predicted_vol, current_vol)
                        if prediction_alert:
                            self.process_alert(prediction_alert)

                # Sleep before next check
                time.sleep(interval)

            except Exception as e:
                logger.error(f"Error in monitoring loop: {str(e)}")
                time.sleep(interval)

    def get_alerts(self, alert_type=None, asset=None, limit=None):
        """
        Get stored alerts with optional filtering.

        Parameters:
        -----------
        alert_type : str, optional
            Filter by alert type
        asset : str, optional
            Filter by asset
        limit : int, optional
            Limit the number of alerts returned

        Returns:
        --------
        list
            List of matching alerts
        """
        # Apply filters
        filtered_alerts = self.alerts

        if alert_type:
            filtered_alerts = [a for a in filtered_alerts if a.alert_type == alert_type]

        if asset:
            filtered_alerts = [a for a in filtered_alerts if a.asset == asset]

        # Apply limit
        if limit:
            filtered_alerts = filtered_alerts[-limit:]

        return filtered_alerts

    def get_recent_alerts(self, hours=24):
        """
        Get alerts from the last specified hours.

        Parameters:
        -----------
        hours : int
            Number of hours to look back

        Returns:
        --------
        list
            List of recent alerts
        """
        now = datetime.now()
        cutoff = now - timedelta(hours=hours)

        return [a for a in self.alerts if a.timestamp >= cutoff]

    def clear_alerts(self):
        """
        Clear all stored alerts.

        Returns:
        --------
        int
            Number of alerts cleared
        """
        count = len(self.alerts)
        self.alerts = []
        return count