"""ME3 Agent - the desktop window.

A real application window rather than a class exercise: it finds the engine (starting it if it is
not already running), shows the console, and cleans up after itself when the window closes.

Order of preference:

1. pywebview - a native window with no browser chrome (needs the Edge WebView2 runtime).
2. An app-mode window from Edge or Chrome (`--app=`), which looks the same to the user.
3. The default browser, as a last resort.

Run it with `pythonw me3_laptop/app_window.py` (or the `ME3 Agent.bat` launcher) so no console
window appears.
"""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
import threading
import time
import webbrowser
from pathlib import Path

BASE = Path(__file__).resolve().parent
DEFAULT_PORT = 8766
TITLE = "SONJ4 Agent"
WINDOW = (1500, 950)
MIN_SIZE = (1100, 700)

BROWSERS = (
    Path(r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe"),
    Path(r"C:\Program Files\Microsoft\Edge\Application\msedge.exe"),
    Path(r"C:\Program Files\Google\Chrome\Application\chrome.exe"),
    Path(r"C:\Program Files (x86)\Google\Chrome\Application\chrome.exe"),
    Path("/usr/bin/chromium"),
    Path("/usr/bin/chromium-browser"),
    Path("/usr/bin/google-chrome"),
    Path("/Applications/Google Chrome.app/Contents/MacOS/Google Chrome"),
    Path("/Applications/Microsoft Edge.app/Contents/MacOS/Microsoft Edge"),
)

# The Edge WebView2 runtime pywebview needs on Windows. Per-machine and per-user keys.
WEBVIEW2_KEYS = (
    r"SOFTWARE\WOW6432Node\Microsoft\EdgeUpdate\Clients\{F3017226-FE2A-4295-8BDF-00C3A9A7E4C5}",
    r"SOFTWARE\Microsoft\EdgeUpdate\Clients\{F3017226-FE2A-4295-8BDF-00C3A9A7E4C5}",
)


def webview2_version() -> str:
    """Installed Edge WebView2 runtime version, or "" when it is absent or non-Windows."""
    if sys.platform != "win32":
        return ""
    try:
        import winreg
    except ImportError:  # not Windows (e.g. the Pi): there is no WebView2 runtime
        return ""

    hklm = getattr(winreg, "HKEY_LOCAL_MACHINE", None)
    open_key = getattr(winreg, "OpenKey", None)
    query_value = getattr(winreg, "QueryValueEx", None)
    if not (hklm and open_key and query_value):
        return ""
    for key in WEBVIEW2_KEYS:
        try:
            with open_key(hklm, key) as handle:
                return str(query_value(handle, "pv")[0])
        except OSError:
            continue
    return ""


def pywebview_version() -> str:
    """Installed pywebview version, or "" when the module is not importable."""
    try:
        from importlib import metadata

        return metadata.version("pywebview")
    except Exception:  # noqa: BLE001 - absence is a valid answer here
        return ""


def browser_path() -> Path | None:
    """The first usable Chromium browser, for the app-mode fallback."""
    found = next((exe for exe in BROWSERS if exe.exists()), None)
    if found:
        return found
    for name in ("chromium", "chromium-browser", "google-chrome", "msedge"):
        which = shutil.which(name)
        if which:
            return Path(which)
    return None


def native_ready() -> bool:
    """Can we open a real native window (pywebview + WebView2)?"""
    return bool(pywebview_version()) and bool(webview2_version())


def window_ready() -> bool:
    """Can we open the full web console in a standalone window (native or Chromium app mode)?"""
    return native_ready() or browser_path() is not None


def report(port: int) -> int:
    """Print which desktop-window strategy would be used, without opening anything."""
    exe = browser_path()
    print(f"engine    : {'running' if server_is_up(port) else 'not running'} on {port}")
    print(f"console   : http://127.0.0.1:{port}/  (static/index.html)")
    print(f"pywebview : {pywebview_version() or 'not installed'}")
    print(f"WebView2  : {webview2_version() or 'not installed'}")
    print(
        f"app-mode  : {exe.name + ' at ' + str(exe) if exe else 'no Edge/Chrome found'}"
    )
    if native_ready():
        print("would open: native window (pywebview)")
    elif exe:
        print(f"would open: {exe.name} app-mode window")
    else:
        print("would open: default browser (last resort)")
    return 0 if (native_ready() or exe) else 1


def log(message: str) -> None:
    print(f"[app] {message}", flush=True)


def health_url(port: int) -> str:
    return f"http://127.0.0.1:{port}/api/health"


def server_is_up(port: int, timeout: float = 1.5) -> bool:
    import requests

    try:
        return requests.get(health_url(port), timeout=timeout).status_code == 200
    except Exception:  # noqa: BLE001 - a refused connection is the normal "not running" case
        return False


def start_engine(port: int) -> subprocess.Popen | None:
    """Start the HTTP engine if nothing is listening, so the window always has something to show."""
    if server_is_up(port):
        log(f"engine already running on {port}")
        return None
    log(f"starting the engine on {port}")
    process = subprocess.Popen(
        [
            sys.executable,
            str(BASE / "server.py"),
            "--port",
            str(port),
            "--no-browser",
            "--force",
        ],
        cwd=str(BASE),
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
    )
    deadline = time.time() + 40.0
    while time.time() < deadline:
        if server_is_up(port):
            log("engine ready")
            return process
        if process.poll() is not None:
            log("engine exited while starting")
            return None
        time.sleep(0.6)
    log("engine did not answer in time")
    return process


def open_native(url: str) -> bool:
    """pywebview window: no browser chrome, its own taskbar entry."""
    try:
        import webview  # type: ignore[import-not-found]
    except ImportError:
        log("pywebview not installed")
        return False
    try:
        webview.create_window(
            TITLE,
            url,
            width=WINDOW[0],
            height=WINDOW[1],
            min_size=MIN_SIZE,
            text_select=True,
        )
        webview.start()
        return True
    except Exception as exc:  # noqa: BLE001 - fall back rather than fail to open anything
        log(f"native window unavailable ({type(exc).__name__}: {exc})")
        return False


def open_app_mode(url: str) -> bool:
    """Chromium app-mode window: no tabs, no address bar, still a browser process."""
    exe = browser_path()
    if exe:
        log(f"using {exe.name} in app mode")
        proc = subprocess.Popen(
            [
                str(exe),
                f"--app={url}",
                f"--window-size={WINDOW[0]},{WINDOW[1]}",
                "--no-first-run",
                "--no-default-browser-check",
            ],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
        )
        proc.wait()
        return True
    return False


def _watchdog(port: int, stop_event: threading.Event) -> None:
    """Auto-restart the engine within ~8 s if it stops responding."""
    misses = 0
    while not stop_event.wait(4.0):
        if server_is_up(port):
            misses = 0
            continue
        misses += 1
        if misses < 2:
            continue
        if stop_event.is_set():
            return
        log("watchdog: engine not connecting — auto-restarting server")
        start_engine(port)
        misses = 0


def _shutdown_engine(port: int) -> None:
    import requests

    for route in ("/api/barge", "/api/stop", "/api/shutdown"):
        try:
            requests.post(f"http://127.0.0.1:{port}{route}", json={}, timeout=2.0)
        except Exception as exc:  # noqa: BLE001
            log(f"shutdown {route}: {type(exc).__name__}")


def _wait_engine_exit(engine: subprocess.Popen | None) -> None:
    if engine is None:
        return
    try:
        engine.wait(timeout=6)
    except subprocess.TimeoutExpired:
        if os.name == "nt":
            subprocess.run(
                ["taskkill", "/PID", str(engine.pid), "/T", "/F"],
                capture_output=True,
                timeout=5,
                check=False,
                creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
            )
        engine.terminate()
        try:
            engine.wait(timeout=4)
        except subprocess.TimeoutExpired:
            engine.kill()


def _supervisor_alive(sup_pid: Path) -> bool:
    if not sup_pid.is_file():
        return False
    try:
        pid = int(sup_pid.read_text(encoding="utf-8").strip())
    except (OSError, ValueError):
        return False
    if pid <= 0:
        return False
    try:
        import psutil

        return bool(psutil.pid_exists(pid))
    except Exception:  # noqa: BLE001
        try:
            os.kill(pid, 0)
            return True
        except OSError:
            return False


def _supervise(port: int, sup_pid: Path) -> None:
    """W7: keep the engine alive after the window closes; exit on `sonj stop`.

    A crash leaves engine.pid in place -> we auto-restart. An explicit `sonj stop`
    removes engine.pid -> we exit. If another supervisor claims the pid file, we
    defer to it.
    """
    misses = 0
    engine_pid_file = BASE / "out" / "engine.pid"
    while True:
        time.sleep(4.0)
        try:
            owner = (
                sup_pid.read_text(encoding="utf-8").strip() if sup_pid.is_file() else ""
            )
        except OSError:
            owner = ""
        if owner != str(os.getpid()):
            log("supervisor: stop signal received - exiting")
            return
        if server_is_up(port):
            misses = 0
            continue
        misses += 1
        if misses < 2:
            continue
        if not engine_pid_file.is_file():
            log("supervisor: engine stopped explicitly - exiting")
            return
        log("supervisor: engine not connecting - auto-restarting")
        start_engine(port)
        misses = 0


def main() -> int:
    parser = argparse.ArgumentParser(description="ME3 Agent desktop window")
    parser.add_argument("--port", type=int, default=DEFAULT_PORT)
    parser.add_argument("--browser", action="store_true", help="skip the native window")
    parser.add_argument(
        "--check", action="store_true", help="report the window strategy and exit"
    )
    parser.add_argument(
        "--shutdown-on-close",
        action="store_true",
        help="old behavior: closing the window also stops the engine",
    )
    args = parser.parse_args()

    if args.check:
        return report(args.port)

    url = f"http://127.0.0.1:{args.port}/"
    engine = start_engine(args.port)
    stop_watch = threading.Event()
    if not args.browser:
        threading.Thread(
            target=_watchdog, args=(args.port, stop_watch), daemon=True
        ).start()

    opened = False
    if not args.browser:
        opened = open_native(url)
    if not opened:
        opened = open_app_mode(url)
    if not opened:
        log("no app-mode browser found; opening the default browser")
        webbrowser.open(url)
        return 0

    # W7: closing/crashing the window must NOT kill the engine. The engine is a
    # background service; this process stays alive as its supervisor (auto-restart)
    # until `sonj stop`. Use --shutdown-on-close for the old close=quit behavior.
    stop_watch.set()
    if args.shutdown_on_close:
        log("window closed - stopping the engine (--shutdown-on-close)")
        _shutdown_engine(args.port)
        _wait_engine_exit(engine)
        return 0

    sup_pid = BASE / "out" / "supervisor.pid"
    if _supervisor_alive(sup_pid):
        try:
            owner = sup_pid.read_text(encoding="utf-8").strip()
        except OSError:
            owner = ""
        if owner != str(os.getpid()):
            log(
                "window closed - another supervisor owns the engine; leaving it running"
            )
            return 0
    sup_pid.parent.mkdir(parents=True, exist_ok=True)
    sup_pid.write_text(str(os.getpid()), encoding="utf-8")
    log("window closed - engine keeps running (supervisor active; 'sonj stop' to quit)")
    try:
        _supervise(args.port, sup_pid)
    finally:
        try:
            if sup_pid.is_file() and sup_pid.read_text(encoding="utf-8").strip() == str(
                os.getpid()
            ):
                sup_pid.unlink()
        except OSError:
            pass
    return 0


if __name__ == "__main__":
    sys.exit(main())
