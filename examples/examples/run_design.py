"""Run the KrafNova workflow programmatically, the way a developer would.

Starts the FastAPI app in a subprocess, POSTs an Arabic design prompt to
/design, prints the parsed intent and the candidate URLs, saves the chosen
SVG to examples/output/, and prints the plain-language explanation.

Usage (from the repo root):

    cp .env.example .env      # add your GEMINI_API_KEY
    # .env: IMAGE_PROVIDER=gemini-svg
    python examples/run_design.py
"""
import json
import os
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "examples" / "output"
PORT = 8000
BASE = f"http://127.0.0.1:{PORT}"

PROMPT = "عايز نقشة نجمة ثمانية إسلامية للحفر على الخشب، مقاس 100 مللي"

# The console must be able to show the Arabic prompt and explanation.
sys.stdout.reconfigure(encoding="utf-8")


def wait_for_server(proc: subprocess.Popen, timeout: float = 60.0) -> None:
    """Poll /health until uvicorn is ready (or the process dies)."""
    deadline = time.time() + timeout
    while time.time() < deadline:
        if proc.poll() is not None:
            raise RuntimeError("uvicorn exited early; check your .env and deps")
        try:
            with urllib.request.urlopen(f"{BASE}/health", timeout=2) as resp:
                if resp.status == 200:
                    return
        except (urllib.error.URLError, ConnectionError, TimeoutError):
            time.sleep(0.5)
    raise TimeoutError("server did not come up in time")


def post_design(message: str) -> dict:
    body = json.dumps({"message": message}).encode("utf-8")
    req = urllib.request.Request(
        f"{BASE}/design", data=body, headers={"Content-Type": "application/json"}
    )
    with urllib.request.urlopen(req, timeout=300) as resp:
        return json.loads(resp.read().decode("utf-8"))


def main() -> None:
    env = os.environ.copy()
    env.setdefault("IMAGE_PROVIDER", "gemini-svg")

    print("== starting the API server (uvicorn app.main:app) ==")
    server = subprocess.Popen(
        [sys.executable, "-m", "uvicorn", "app.main:app", "--port", str(PORT)],
        cwd=ROOT,
        env=env,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    try:
        wait_for_server(server)
        print(f"server is up at {BASE}\n")

        print(f"== POST /design ==\n{PROMPT}\n")
        started = time.time()
        result = post_design(PROMPT)
        elapsed = time.time() - started

        intent = result["intent"]
        print("== parsed intent (LLM -> validated JSON) ==")
        print(json.dumps(intent, ensure_ascii=False, indent=2), "\n")

        print("== candidates ==")
        for cand in result["candidates"]:
            print(f"  {cand['id']}: {BASE}{cand['image_url']}")
        print(f"\n({len(result['candidates'])} candidates in {elapsed:.1f}s)\n")

        chosen = next((c for c in result["candidates"] if c.get("svg")), None)
        if chosen is None:
            raise RuntimeError("no candidate carried an SVG to save")
        OUT_DIR.mkdir(parents=True, exist_ok=True)
        out_path = OUT_DIR / "eight_pointed_star.svg"
        out_path.write_text(chosen["svg"], encoding="utf-8")
        print(f"== saved the chosen design ==\n{out_path}\n")

        print("== explanation ==")
        print(result["explanation"])
    finally:
        server.terminate()
        server.wait(timeout=10)
        print("\n== server stopped, done ==")


if __name__ == "__main__":
    main()
