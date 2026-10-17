"""Process-level configuration from environment variables (everything else lives in Settings)."""

from __future__ import annotations

import os
from pathlib import Path

DATA_DIR = Path(os.environ.get("TRACKER_DATA_DIR", "data"))
DB_PATH = Path(os.environ.get("TRACKER_DB_PATH", str(DATA_DIR / "tracker.db")))
HOST = os.environ.get("TRACKER_HOST", "0.0.0.0")
PORT = int(os.environ.get("TRACKER_PORT", "8000"))

# Optional HTTP Basic auth for the web UI. Leave the password empty to disable auth.
UI_USERNAME = os.environ.get("TRACKER_UI_USERNAME", "admin")
UI_PASSWORD = os.environ.get("TRACKER_UI_PASSWORD", "")

# Secrets and deployment-specific values may come from the environment instead of the Settings page.
# A value saved in Settings wins.
ENV_FALLBACKS = {
    ("sources", "searxng", "base_url"): "SEARXNG_URL",
    ("sources", "ebay_api", "client_id"): "EBAY_CLIENT_ID",
    ("sources", "ebay_api", "client_secret"): "EBAY_CLIENT_SECRET",
    ("sources", "serpapi", "api_key"): "SERPAPI_API_KEY",
    ("sources", "serper", "api_key"): "SERPER_API_KEY",
    ("email", "resend_api_key"): "RESEND_API_KEY",
    ("email", "smtp_username"): "SMTP_USERNAME",
    ("email", "smtp_password"): "SMTP_PASSWORD",
}
