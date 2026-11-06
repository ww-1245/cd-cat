"""截图工具：用系统 Edge 打开 Streamlit 页面，按需点击并截图。

仅用于本地视觉核验，不参与打包。
    python demo/_shot.py light 8511
    python demo/_shot.py dark  8512
"""

from __future__ import annotations

import sys
from pathlib import Path

from playwright.sync_api import sync_playwright

OUT = Path(__file__).resolve().parent / "output" / "shots"
OUT.mkdir(parents=True, exist_ok=True)


def _wait_idle(page, timeout: int = 90_000) -> None:
    """等 Streamlit 跑完一轮脚本（running indicator 消失）。"""
    page.wait_for_selector('div[data-testid="stApp"]', timeout=timeout)
    page.wait_for_timeout(1200)
    for _ in range(120):
        busy = page.evaluate(
            """() => !!document.querySelector('[data-testid="stStatusWidget"]')"""
        )
        if not busy:
            break
        page.wait_for_timeout(500)
    page.wait_for_timeout(600)


def main() -> None:
    tag = sys.argv[1] if len(sys.argv) > 1 else "light"
    port = sys.argv[2] if len(sys.argv) > 2 else "8511"
    url = f"http://localhost:{port}/"

    with sync_playwright() as p:
        browser = p.chromium.launch(channel="msedge", headless=True)
        page = browser.new_page(viewport={"width": 1680, "height": 1050})
        page.goto(url, wait_until="networkidle", timeout=90_000)
        _wait_idle(page)

        page.screenshot(path=str(OUT / f"dash_{tag}_01_init.png"))

        # 运行体检
        page.get_by_role("button", name="运行体检").click()
        page.get_by_text("平均用题量", exact=False).first.wait_for(timeout=300_000)
        page.wait_for_timeout(3000)  # 等 Plotly 画完

        page.screenshot(path=str(OUT / f"dash_{tag}_02_top.png"))

        def shot_at(anchor: str, name: str) -> None:
            """滚到某个区块标题再截屏。"""
            try:
                loc = page.locator(f"div.cdc-sec:has-text('{anchor}')").first
                # block='start' 才能把区块标题顶到视口顶端，
                # 默认的最短滚动会把标题留在屏幕底部，截不到内容。
                loc.evaluate("el => el.scrollIntoView({block:'start'})")
                page.wait_for_timeout(1200)
                page.screenshot(path=str(OUT / f"dash_{tag}_{name}.png"))
            except Exception as exc:  # noqa: BLE001
                print(f"{name} 截图失败:", exc)

        shot_at("4. 知识地图", "03_map")
        shot_at("5. 逐题体检", "04_table")
        shot_at("7. 导出", "05_export")

        browser.close()
    print("saved to", OUT)


if __name__ == "__main__":
    main()
