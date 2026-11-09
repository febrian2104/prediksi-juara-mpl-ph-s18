"""Capture the three live dashboard pages after Streamlit is running locally."""

import os
from pathlib import Path

from playwright.sync_api import sync_playwright

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "assets" / "screenshots"
PAGES = {
    "regular-season.png": "http://127.0.0.1:8501/",
    "playoff.png": "http://127.0.0.1:8501/page_playoff",
    "hasil-prediksi.png": "http://127.0.0.1:8501/page_results",
}


def main() -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    browser_temp = ROOT / ".cache/browser"
    browser_temp.mkdir(parents=True, exist_ok=True)
    os.environ["TMPDIR"] = str(browser_temp)
    os.environ["XDG_CACHE_HOME"] = str(ROOT / ".cache")
    with sync_playwright() as playwright:
        browser = playwright.chromium.launch(
            headless=True,
            executable_path="/usr/bin/google-chrome",
            args=["--no-sandbox"],
        )
        page = browser.new_page(viewport={"width": 1440, "height": 1100})
        for filename, url in PAGES.items():
            page.goto(url, wait_until="networkidle", timeout=30_000)
            page.wait_for_selector("h1", timeout=30_000)
            page.wait_for_timeout(1_000)
            if page.locator('[data-testid="stException"]').count():
                raise RuntimeError(f"Error aplikasi ditemukan: {url}")
            page.screenshot(path=OUTPUT / filename, full_page=False)
        browser.close()


if __name__ == "__main__":
    main()
