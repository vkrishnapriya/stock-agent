"""
scripts/setup_kite.py
Refresh the Kite Connect access token and write it to .env.local.

Run once per trading day before the market opens:
    python scripts/setup_kite.py

The script opens the Kite login URL in your default browser, waits for you
to paste back the request_token from the redirect URL, exchanges it for an
access_token, and persists it to .env.local.
"""

from __future__ import annotations

import os
import sys
import webbrowser
from pathlib import Path

from dotenv import dotenv_values, set_key

# Ensure project root is on sys.path when run directly.
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


def load_env() -> dict[str, str | None]:
    env_local = PROJECT_ROOT / ".env.local"
    env_example = PROJECT_ROOT / ".env.example"
    values: dict[str, str | None] = {}
    if env_example.exists():
        values.update(dotenv_values(env_example))
    if env_local.exists():
        values.update(dotenv_values(env_local))
    values.update({k: v for k, v in os.environ.items() if k in values})
    return values


def main() -> None:
    try:
        from kiteconnect import KiteConnect  # type: ignore[import-untyped]
    except ImportError:
        print("kiteconnect package not installed. Run: pip install kiteconnect")
        sys.exit(1)

    env = load_env()
    api_key = env.get("KITE_API_KEY") or ""
    api_secret = env.get("KITE_SECRET") or ""

    if not api_key or not api_secret:
        print(
            "KITE_API_KEY and KITE_SECRET must be set in .env.local\n"
            "Copy .env.example → .env.local and fill in your credentials."
        )
        sys.exit(1)

    kite = KiteConnect(api_key=api_key)
    login_url = kite.login_url()

    print(f"\nOpening Kite login page in browser:\n  {login_url}\n")
    webbrowser.open(login_url)

    print(
        "After logging in, Kite will redirect to your app's redirect URL.\n"
        "Copy the value of the 'request_token' query parameter and paste it below.\n"
    )
    request_token = input("request_token: ").strip()

    if not request_token:
        print("No token provided. Aborting.")
        sys.exit(1)

    try:
        session = kite.generate_session(request_token, api_secret=api_secret)
    except Exception as exc:  # noqa: BLE001
        print(f"Failed to generate session: {exc}")
        sys.exit(1)

    access_token: str = session["access_token"]
    env_local_path = PROJECT_ROOT / ".env.local"
    env_local_path.touch(exist_ok=True)
    set_key(str(env_local_path), "KITE_ACCESS_TOKEN", access_token)

    print(f"\nAccess token saved to {env_local_path}")
    print("Token is valid until midnight IST. Re-run this script each trading day.")


if __name__ == "__main__":
    main()
