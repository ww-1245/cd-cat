"""跨页面深色模式抽查：切到几个非看板页面各截一张。"""

import sys
from pathlib import Path

from playwright.sync_api import sync_playwright

OUT = Path(__file__).resolve().parent / "output" / "shots"
OUT.mkdir(parents=True, exist_ok=True)

port = sys.argv[1] if len(sys.argv) > 1 else "8520"
pages = sys.argv[2].split(",") if len(sys.argv) > 2 else ["参数标定", "题库诊断", "系统说明"]


def wait_idle(page, timeout_ms: int = 180_000) -> None:
    page.wait_for_selector('div[data-testid="stApp"]', timeout=timeout_ms)
    page.wait_for_timeout(1500)
    for _ in range(240):
        busy = page.evaluate(
            """() => !!document.querySelector('[data-testid="stStatusWidget"]')"""
        )
        if not busy:
            break
        page.wait_for_timeout(500)
    page.wait_for_timeout(2500)


with sync_playwright() as p:
    browser = p.chromium.launch(channel="msedge", headless=True)
    page = browser.new_page(viewport={"width": 1680, "height": 1050})
    page.goto(f"http://localhost:{port}/", wait_until="networkidle", timeout=90_000)
    wait_idle(page)

    for i, label in enumerate(pages, start=1):
        try:
            page.get_by_text(label, exact=True).first.click(timeout=20_000)
            wait_idle(page)
            page.screenshot(path=str(OUT / f"page_{i}_{label}.png"))
            print("ok", label)
        except Exception as exc:  # noqa: BLE001
            print("fail", label, type(exc).__name__, str(exc)[:160])

    browser.close()
