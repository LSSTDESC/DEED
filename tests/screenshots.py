"""Screenshot every page of app.py in light and dark mode, with a headless browser.

Starts the app on a spare port with the example sacc file and two chains loaded,
visits each page, waits for the figures, and writes PNGs. Also reports anything
the browser logged as a console error -- the part of a Dash app that the Flask
test-client smoke tests (tests/smoke_test.py) can't see.

Needs Playwright, which is deliberately not a DEED dependency. A throwaway env:
    module load python && python -m venv ~/.cache/deed-screenshot-venv
    ~/.cache/deed-screenshot-venv/bin/pip install playwright
    ~/.cache/deed-screenshot-venv/bin/python -m playwright install chromium

Usage (from the repo root; the app itself runs on the DEED venv's python):
    ~/.cache/deed-screenshot-venv/bin/python tests/screenshots.py [--out DIR] [--app-python .venv/bin/python]
"""
import argparse
import os
import re
import subprocess
import sys
import time

from playwright.sync_api import sync_playwright

REPO_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SACC     = os.path.join(REPO_DIR, "paul_auto_forecast_fid_3x2pt_linear_sys_limber_20_log_bins.sacc")
CHAINS   = [os.path.expanduser(f"~/DEED/example_analysis_files/{f}") for f in ("chain_2x2pt_lcdm_SR_maglim.txt", "chain_3x2pt_lcdm_SR_maglim.txt")]
PAGES    = [("compare", "/"), ("explore", "/explore"), ("registry", "/registry")]


### Dash's dev tools ask plotly.com for the latest version; the browser blocks
### the cross-origin request, which is noise rather than a DEED problem
IGNORED_CONSOLE = ("dash-version.plotly.com", "net::ERR_FAILED")


def start_app(app_python, port, log_path):
    """Launch app.py, logging to log_path, and return (process, base URL) once Dash is up."""
    args = [app_python, "app.py", "--port", str(port), "--file", SACC] + sum([["--file", c] for c in CHAINS], [])
    log  = open(log_path, "w")
    proc = subprocess.Popen(args, cwd = REPO_DIR, stdout = log, stderr = subprocess.STDOUT, text = True)
    deadline = time.time() + 90
    while time.time() < deadline:
        match = re.search(r"Dash is running on (http://[^/\s]+)", open(log_path).read())
        if match:
            return proc, match.group(1)
        time.sleep(0.5)
    proc.kill()
    raise RuntimeError(f"app.py did not start; see {log_path}")


def settle(page, timeout_ms = 20000):
    """Wait for Dash to finish its callbacks: no loading spinners, chain image filled if present."""
    page.wait_for_load_state("networkidle")
    page.wait_for_function("() => !document.querySelector('[data-dash-is-loading=\"true\"]')", timeout = timeout_ms)
    page.wait_for_function(
        "() => { const img = document.querySelector('.deed-chain-img'); return !img || img.offsetParent === null || (img.src && img.src.startsWith('data:image')); }",
        timeout = timeout_ms,
    )
    page.wait_for_timeout(800)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default = os.path.expanduser("~/DEED/layout_screenshots"))
    parser.add_argument("--app-python", default = os.path.join(REPO_DIR, ".venv", "bin", "python"))
    parser.add_argument("--port", type = int, default = 8098)
    parser.add_argument("--width", type = int, default = 1440)
    args = parser.parse_args()
    os.makedirs(args.out, exist_ok = True)

    log_path   = os.path.join(args.out, "server.log")
    proc, base = start_app(args.app_python, args.port, log_path)
    errors = []
    try:
        with sync_playwright() as p:
            browser = p.chromium.launch()
            page = browser.new_page(viewport = {"width": args.width, "height": 900})
            page.on("console", lambda m: errors.append(f"{page.url}: {m.text}")
                    if m.type == "error" and not any(s in m.text for s in IGNORED_CONSOLE) else None)
            page.on("pageerror", lambda e: errors.append(f"{page.url}: {e}"))

            for theme in ("light", "dark"):
                for name, path in PAGES:
                    page.goto(base + path)
                    settle(page)
                    switch = page.locator("#theme-switch")
                    if switch.is_checked() != (theme == "dark"):
                        switch.click()
                        ### The redraw requests start after the click returns,
                        ### so networkidle alone can resolve before they do
                        page.wait_for_timeout(3000)
                        settle(page)
                    if name == "explore" and theme == "light":
                        ### Also capture the chain view, the second kind of explore page
                        page.select_option("#exp-dataset", CHAINS[0].replace("~", os.path.expanduser("~")))
                        settle(page)
                        page.screenshot(path = os.path.join(args.out, f"explore-chain_{theme}.png"), full_page = True)
                        page.select_option("#exp-dataset", SACC)
                        settle(page)
                    out = os.path.join(args.out, f"{name}_{theme}.png")
                    page.screenshot(path = out, full_page = True)
                    print("wrote", out)

            ### A phone-width check of the stacked layout
            page.set_viewport_size({"width": 420, "height": 900})
            page.goto(base + "/")
            settle(page)
            page.screenshot(path = os.path.join(args.out, "compare_narrow.png"), full_page = False)
            browser.close()
    finally:
        proc.terminate()

    print(f"\n{len(errors)} browser console error(s)")
    for e in errors[:20]:
        print("  ", e[:300])
    tracebacks = open(log_path).read().count("Traceback")
    print(f"{tracebacks} server traceback(s){f' -- see {log_path}' if tracebacks else ''}")
    errors += ["traceback"] * tracebacks
    sys.exit(1 if errors else 0)


if __name__ == "__main__":
    main()
