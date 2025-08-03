from pathlib import Path
from typing import Any, Dict

try:
    import tomllib  # Python 3.11+
except ModuleNotFoundError:  # pragma: no cover
    import toml as tomllib  # type: ignore

DEFAULT_CONFIG: Dict[str, Any] = {
    "line_notify": {
        "api": "",
        "token": "",
        "time_interval_sec": 0,
        "class_interest": [],
        "max_value": 0,
        "status_running": False,
    }
}


class AppConfig:
    """Simple application configuration loader.

    Attempts to load a TOML configuration file. If the file is not found,
    returns a default structure to avoid ``KeyError`` when accessing
    configuration values.
    """

    def __init__(self, config_path: str = "app_config.toml") -> None:
        self.config_path = Path(config_path)

    def load_toml_config(self) -> Dict[str, Any]:
        if self.config_path.exists():
            with open(self.config_path, "rb") as f:
                return tomllib.load(f)
        return DEFAULT_CONFIG.copy()
