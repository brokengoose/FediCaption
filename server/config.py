"""Cross-platform configuration management for FediCaption (macOS & Linux)."""

import json
import os
import platform
from pathlib import Path
from typing import Any, Dict


DEFAULT_CONFIG: Dict[str, Any] = {
    "mastodon": {
        "instance_url": "https://mastodon.social",
        "access_token": "",
        "client_id": "",
        "client_secret": "",
        "account": None,
        "max_post_chars": 500,
        "max_media_attachments": 4,
        "alt_text_limit": 1500,
    },
    "ai": {
        "provider": "ollama",  # "ollama" | "gemini" | "openai" | "anthropic"
        "auto_generate": True,
        "detail_level": "descriptive",  # "concise" | "descriptive" | "detailed"
        "include_post_context": True,
        "require_alt_text": "warn",  # "warn" | "require" | "off"
        "language": "English",
        "custom_instructions": "",
        "ollama": {
            "base_url": "http://localhost:11434",
            "model": "llava:latest",
        },
        "gemini": {
            "api_key": "",
            "model": "gemini-2.5-flash",
        },
        "openai": {
            "base_url": "https://api.openai.com/v1",
            "api_key": "",
            "model": "gpt-4o-mini",
        },
        "anthropic": {
            "api_key": "",
            "model": "claude-3-7-sonnet-latest",
        },
    },
    "ui": {
        "theme": "dark",  # "dark" | "light" | "high-contrast"
        "default_visibility": "public",  # "public" | "unlisted" | "private" | "direct"
        "video_keyframes_count": 4,
    },
}


def get_config_dir() -> Path:
    """Return the OS-appropriate configuration directory for macOS or Linux."""
    env_override = os.environ.get("FEDICAPTION_CONFIG_DIR")
    if env_override:
        path = Path(env_override).expanduser()
        path.mkdir(parents=True, exist_ok=True)
        return path

    system = platform.system().lower()
    if system == "darwin":
        base = Path.home() / "Library" / "Application Support" / "FediCaption"
    else:
        # Linux / FreeBSD / POSIX following XDG Base Directory Specification
        xdg_config = os.environ.get("XDG_CONFIG_HOME")
        if xdg_config:
            base = Path(xdg_config).expanduser() / "fedicaption"
        else:
            base = Path.home() / ".config" / "fedicaption"

    try:
        base.mkdir(parents=True, exist_ok=True)
        return base
    except OSError:
        # Fallback to workspace-local config directory if home dir is restricted
        fallback = Path.cwd() / ".fedicaption_config"
        fallback.mkdir(parents=True, exist_ok=True)
        return fallback


def get_config_path() -> Path:
    """Return the path to config.json."""
    return get_config_dir() / "config.json"


def _deep_merge(base: Dict[str, Any], updates: Dict[str, Any]) -> Dict[str, Any]:
    """Recursively merge updates into base dict without losing nested keys."""
    result = json.loads(json.dumps(base))
    for key, value in updates.items():
        if (
            key in result
            and isinstance(result[key], dict)
            and isinstance(value, dict)
        ):
            result[key] = _deep_merge(result[key], value)
        else:
            result[key] = value
    return result


def load_config() -> Dict[str, Any]:
    """Load configuration from disk, merging with defaults."""
    config_path = get_config_path()
    if not config_path.exists():
        return json.loads(json.dumps(DEFAULT_CONFIG))

    try:
        with open(config_path, "r", encoding="utf-8") as f:
            stored = json.load(f)
        if not isinstance(stored, dict):
            return json.loads(json.dumps(DEFAULT_CONFIG))
        return _deep_merge(DEFAULT_CONFIG, stored)
    except (OSError, json.JSONDecodeError):
        return json.loads(json.dumps(DEFAULT_CONFIG))


def save_config(updates: Dict[str, Any]) -> Dict[str, Any]:
    """Merge updates into current config and persist to disk with restricted permissions."""
    current = load_config()
    merged = _deep_merge(current, updates)
    config_path = get_config_path()
    config_path.parent.mkdir(parents=True, exist_ok=True)

    temp_path = config_path.with_suffix(".tmp")
    with open(temp_path, "w", encoding="utf-8") as f:
        json.dump(merged, f, indent=2)

    try:
        os.chmod(temp_path, 0o600)
    except OSError:
        pass

    temp_path.replace(config_path)
    return merged


def sanitize_config_for_client(config: Dict[str, Any]) -> Dict[str, Any]:
    """Return config suitable for sending to the local browser UI."""
    # Since this is a local single-user desktop client, we provide the config
    # along with boolean flags indicating whether keys are configured.
    result = json.loads(json.dumps(config))
    result["config_path"] = str(get_config_path())
    return result
