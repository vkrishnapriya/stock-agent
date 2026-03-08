"""
scripts/download_bhav.py
Download NSE Bhavcopy (end-of-day price data) for a given date and store it
in data/cache/.

Usage:
    python scripts/download_bhav.py               # today's bhavcopy
    python scripts/download_bhav.py 2025-01-15    # specific date (YYYY-MM-DD)

Output:
    data/cache/bhav/YYYY-MM-DD/cm<DDMMMYYYY>bhav.csv.zip
    data/cache/bhav/YYYY-MM-DD/cm<DDMMMYYYY>bhav.csv      (extracted)
"""

from __future__ import annotations

import sys
import zipfile
from datetime import date, datetime
from pathlib import Path

import httpx

PROJECT_ROOT = Path(__file__).resolve().parent.parent
CACHE_DIR = PROJECT_ROOT / "data" / "cache" / "bhav"

NSE_BHAV_URL = (
    "https://nsearchives.nseindia.com/content/historical/EQUITIES"
    "/{year}/{month}/cm{day}{month_upper}{year}bhav.csv.zip"
)

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/120.0.0.0 Safari/537.36"
    ),
    "Referer": "https://www.nseindia.com",
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
}


def build_url(target_date: date) -> tuple[str, str]:
    """Return (url, filename_stem) for the given date."""
    day = target_date.strftime("%d")
    month_abbr = target_date.strftime("%b").upper()
    year = target_date.strftime("%Y")
    month_num = target_date.strftime("%m")
    filename = f"cm{day}{month_abbr}{year}bhav.csv.zip"
    url = (
        f"https://nsearchives.nseindia.com/content/historical/EQUITIES"
        f"/{year}/{month_num}/{filename}"
    )
    return url, filename


def download_bhav(target_date: date) -> Path:
    url, filename = build_url(target_date)
    out_dir = CACHE_DIR / target_date.isoformat()
    out_dir.mkdir(parents=True, exist_ok=True)
    zip_path = out_dir / filename
    csv_name = filename.replace(".zip", "")
    csv_path = out_dir / csv_name

    if csv_path.exists():
        print(f"Already downloaded: {csv_path}")
        return csv_path

    print(f"Downloading: {url}")
    with httpx.Client(headers=HEADERS, follow_redirects=True, timeout=30) as client:
        # NSE requires a session cookie — hit the homepage first.
        client.get("https://www.nseindia.com")
        response = client.get(url)
        response.raise_for_status()

    zip_path.write_bytes(response.content)
    print(f"Saved zip: {zip_path} ({zip_path.stat().st_size / 1024:.1f} KB)")

    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(out_dir)
    print(f"Extracted: {csv_path}")
    return csv_path


def parse_date(date_str: str) -> date:
    try:
        return datetime.strptime(date_str, "%Y-%m-%d").date()
    except ValueError:
        print(f"Invalid date format '{date_str}'. Expected YYYY-MM-DD.")
        sys.exit(1)


def main() -> None:
    if len(sys.argv) > 1:
        target = parse_date(sys.argv[1])
    else:
        target = date.today()

    if target.weekday() >= 5:  # Saturday=5, Sunday=6
        print(f"Warning: {target} is a weekend. NSE bhavcopy may not exist.")

    csv_path = download_bhav(target)
    print(f"\nBhavcopy ready at: {csv_path}")


if __name__ == "__main__":
    main()
