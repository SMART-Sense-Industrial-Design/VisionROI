import json
from pathlib import Path
from typing import Dict, Any

STATE_FILE = Path(__file__).resolve().parent / "state.json"


def load_state() -> Dict[str, Dict[int, Any]]:
    try:
        with STATE_FILE.open("r", encoding="utf-8") as f:
            data = json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        return {"active_sources": {}, "active_modules": {}, "inference_started": {}}
    result: Dict[str, Dict[int, Any]] = {}
    for key in ("active_sources", "active_modules", "inference_started"):
        raw = data.get(key, {})
        if isinstance(raw, dict):
            try:
                if key == "inference_started":
                    result[key] = {int(k): bool(v) for k, v in raw.items()}
                else:
                    result[key] = {int(k): v for k, v in raw.items()}
            except Exception:
                result[key] = {}
        else:
            result[key] = {}
    return result


def save_state(
    active_sources: Dict[int, str],
    active_modules: Dict[int, str],
    inference_started: Dict[int, bool],
) -> None:
    data = {
        "active_sources": {str(k): v for k, v in active_sources.items()},
        "active_modules": {str(k): v for k, v in active_modules.items()},
        "inference_started": {str(k): bool(v) for k, v in inference_started.items()},
    }
    try:
        with STATE_FILE.open("w", encoding="utf-8") as f:
            json.dump(data, f)
    except Exception:
        pass
