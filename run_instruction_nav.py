import threading
import json
import socket
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

print("[startup] importing RoboTHOR navigation module...", flush=True)
from robothor_object_nav import (
    PORT,
    app,
    controller,
    go_to_place_worker,
    log,
    nav_status,
    update_latest_frame,
)
print("[startup] RoboTHOR navigation module imported.", flush=True)


SCRIPT_DIR = Path(__file__).resolve().parent
SYSTEM_PYTHON = "/usr/bin/python3"


def debug(message):
    print(f"[instruction-nav] {message}", flush=True)


def choose_port(preferred_port):
    for port in range(preferred_port, preferred_port + 20):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            try:
                sock.bind(("0.0.0.0", port))
            except OSError:
                continue
            return port
    raise RuntimeError(f"No free port found from {preferred_port} to {preferred_port + 19}")


def wait_for_http_status(port, timeout_sec=10.0):
    url = f"http://127.0.0.1:{port}/status"
    deadline = time.time() + timeout_sec
    last_error = None

    while time.time() < deadline:
        try:
            with urllib.request.urlopen(url, timeout=1.0) as response:
                body = response.read().decode("utf-8", errors="replace")
                return response.status, body
        except Exception as exc:
            last_error = exc
            time.sleep(0.2)

    raise RuntimeError(f"Flask health check failed for {url}: {last_error}")


def predict_instruction(sentence):
    debug(f"predicting instruction with {SYSTEM_PYTHON}")
    completed = subprocess.run(
        [
            SYSTEM_PYTHON,
            str(SCRIPT_DIR / "predict_instruction.py"),
            "--text",
            sentence,
        ],
        check=True,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    return json.loads(completed.stdout)


def start_browser_ui():
    debug("selecting web port")
    port = choose_port(PORT)
    debug(f"selected port {port}")
    debug("capturing initial frame")
    update_latest_frame(controller.last_event)
    debug("starting Flask thread")
    flask_thread = threading.Thread(
        target=lambda: app.run(
            host="0.0.0.0",
            port=port,
            use_reloader=False,
            threaded=True,
        ),
        daemon=True,
    )
    flask_thread.start()
    debug("waiting for /status health check")
    status_code, _ = wait_for_http_status(port)
    debug(f"/status health check OK: HTTP {status_code}")
    log(f"Open browser: http://localhost:{port}")
    log("Browser UI is ready.")


def main():
    try:
        start_browser_ui()

        log("Type a natural-language instruction. Type quit or exit to stop.")

        while True:
            sentence = input("\nInstruction: ").strip()

            if sentence.lower() in ["quit", "exit"]:
                break

            if not sentence:
                continue

            if nav_status["running"]:
                print("Navigation already running.")
                continue

            try:
                result = predict_instruction(sentence)
            except subprocess.CalledProcessError as exc:
                print("Instruction prediction failed.")
                if exc.stderr:
                    print(exc.stderr)
                else:
                    print(exc)
                continue

            place = result["place"]
            difficulty = result["difficulty"]
            print(
                f"Predicted difficulty: {difficulty} "
                f"({result['difficulty_confidence']:.3f})"
            )
            print(f"Predicted place     : {place} ({result['place_confidence']:.3f})")

            thread = threading.Thread(
                target=go_to_place_worker,
                args=(place,),
                daemon=True,
            )
            thread.start()
    finally:
        debug("stopping AI2-THOR controller")
        controller.stop()
        debug("shutdown complete")


if __name__ == "__main__":
    main()
