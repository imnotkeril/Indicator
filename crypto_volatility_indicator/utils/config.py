"""
Configuration module for the Crypto Volatility Indicator.
Handles loading and parsing of configuration from files or environment variables.
"""

import os
import sys
import json
import yaml
import logging
from pathlib import Path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, project_root)
from crypto_volatility_indicator.utils.logger import get_logger
# Default configuration values
DEFAULT_CONFIG = {
    "exchange": {
        "name": "binance",
        "api_key": "",
        "api_secret": "",
        "timeout": 30
    },
    "symbols": ["BTC/USDT", "ETH/USDT", "SOL/USDT"],
    "timeframes": {
        "micro": "1m",
        "meso": "1h",
        "macro": "1d"
    },
    "volatility": {
        "lookback_period": 20,
        "fast_period": 2,
        "slow_period": 30,
        "implied_vol_weight": 0.3,
        "realized_vol_weight": 0.7
    },
    "analysis": {
        "fractal": {
            "enabled": True,
            "max_lag": 20
        },
        "cycle": {
            "enabled": True
        },
        "regime": {
            "enabled": True
        },
        "implied_vol": {
            "enabled": True,
            "min_expiry_days": 1,
            "max_expiry_days": 90
        }
    },
    "prediction": {
        "model_type": "hybrid",
        "training_interval": "1d",
        "retraining_frequency": "1w"
    },
    "visualization": {
        "charts_enabled": True,
        "dashboard_enabled": False,
        "alerts_enabled": True
    },
    "api": {
        "rest_enabled": False,
        "rest_port": 8000,
        "websocket_enabled": False,
        "websocket_port": 8001
    },
    "logging": {
        "level": "INFO",
        "file": "volatility_indicator.log",
        "console": True
    }
}


class Config:
    """Configuration manager for the Crypto Volatility Indicator."""

    def __init__(self, config_path=None):
        """
        Initialize configuration with optional path to config file.

        Parameters:
        -----------
        config_path : str or Path, optional
            Path to configuration file (JSON or YAML)
        """
        self.config = DEFAULT_CONFIG.copy()

        if config_path:
            self.load_from_file(config_path)

        # Override from environment variables
        self.load_from_env()

        self.validate()

    def load_from_file(self, config_path):
        """
        Load configuration from a file.

        Parameters:
        -----------
        config_path : str or Path
            Path to configuration file (JSON or YAML)
        """
        path = Path(config_path)

        if not path.exists():
            logging.warning(f"Config file {path} not found. Using default configuration.")
            return

        try:
            with open(path, 'r') as f:
                if path.suffix.lower() in ['.yaml', '.yml']:
                    file_config = yaml.safe_load(f)
                elif path.suffix.lower() == '.json':
                    file_config = json.load(f)
                else:
                    logging.warning(f"Unsupported config file format: {path.suffix}. Using default configuration.")
                    return

                # Update config recursively
                self._update_recursive(self.config, file_config)

            logging.info(f"Configuration loaded from {path}")
        except Exception as e:
            logging.error(f"Error loading config from {path}: {e}")

    def load_from_env(self):
        """Load configuration from environment variables."""
        # Example: CVI_EXCHANGE_NAME -> config['exchange']['name']
        prefix = "CVI_"

        for key in os.environ:
            if key.startswith(prefix):
                path = key[len(prefix):].lower().split('_')
                value = os.environ[key]

                # Convert value to appropriate type
                if value.lower() in ['true', 'yes', '1']:
                    value = True
                elif value.lower() in ['false', 'no', '0']:
                    value = False
                elif value.isdigit():
                    value = int(value)
                elif self._is_float(value):
                    value = float(value)

                # Navigate to the config item and update it
                config_item = self.config
                for i, p in enumerate(path):
                    if i == len(path) - 1:
                        config_item[p] = value
                    else:
                        if p not in config_item:
                            config_item[p] = {}
                        config_item = config_item[p]

    def _is_float(self, value):
        """Check if a string can be converted to float."""
        try:
            float(value)
            return True
        except ValueError:
            return False

    def _update_recursive(self, d, u):
        """Recursively update a dictionary."""
        for k, v in u.items():
            if isinstance(v, dict) and k in d and isinstance(d[k], dict):
                self._update_recursive(d[k], v)
            else:
                d[k] = v

    def validate(self):
        """Validate the configuration values."""
        # Basic validation
        if not self.config['symbols']:
            logging.warning("No symbols specified in configuration. Using default: ['BTC/USDT']")
            self.config['symbols'] = ["BTC/USDT"]

        # Add more validation as needed

    def get(self, key=None, default=None):
        """
        Get configuration value.

        Parameters:
        -----------
        key : str, optional
            Dot-separated path to configuration item (e.g., 'exchange.name')
            If None, returns the entire configuration
        default : any, optional
            Default value to return if key is not found

        Returns:
        --------
        any
            Configuration value
        """
        if key is None:
            return self.config

        parts = key.split('.')
        value = self.config

        try:
            for part in parts:
                value = value[part]
            return value
        except (KeyError, TypeError):
            return default

    def set(self, key, value):
        """
        Set configuration value.

        Parameters:
        -----------
        key : str
            Dot-separated path to configuration item (e.g., 'exchange.name')
        value : any
            Value to set
        """
        parts = key.split('.')
        config_item = self.config

        for i, part in enumerate(parts):
            if i == len(parts) - 1:
                config_item[part] = value
            else:
                if part not in config_item:
                    config_item[part] = {}
                config_item = config_item[part]

    def save(self, path):
        """
        Save configuration to a file.

        Parameters:
        -----------
        path : str or Path
            Path to save configuration file
        """
        path = Path(path)

        try:
            with open(path, 'w') as f:
                if path.suffix.lower() in ['.yaml', '.yml']:
                    yaml.dump(self.config, f, default_flow_style=False)
                elif path.suffix.lower() == '.json':
                    json.dump(self.config, f, indent=4)
                else:
                    raise ValueError(f"Unsupported file format: {path.suffix}")

            logging.info(f"Configuration saved to {path}")
        except Exception as e:
            logging.error(f"Error saving config to {path}: {e}")
            raise


# Global configuration instance
config = Config()


def load_config(config_path=None):
    """
    Load configuration from a file and return the global config instance.

    Parameters:
    -----------
    config_path : str or Path, optional
        Path to configuration file

    Returns:
    --------
    Config
        Global config instance
    """
    global config
    config = Config(config_path)
    return config