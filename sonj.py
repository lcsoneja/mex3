"""SONJ - the command line face of the ME3 voice agent.

One command runs the engine, talks to it, and reports what it is doing. Every subcommand is
non-interactive and returns an exit code, so the Raspberry Pi can be driven over SSH and from
scripts. Nothing here duplicates the engine: it reads the same state and event stream the GUI
shows, through the local HTTP API.

Usage:
    sonj                     start the engine, open the window, keep a live status dash
    sonj dash                live one-view dashboard in this terminal (Ctrl+C quits)
    sonj doctor              check this machine can run the agent
    sonj status --json       engine, audio, agents, memory
    sonj ask "..."           one turn, print the answer
    sonj say "..."           speak text without a turn
    sonj listen              hands-free voice session
    sonj bench --turns 20    the standardized latency method
"""

from __future__ import annotations

import argparse
import getpass
import json
import os
import platform
import shlex
import shutil
import signal
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, cast

import numpy as np

ROOT = Path(__file__).resolve().parent  # sonj4/
PROJECT = ROOT.parent
BASE_URL = os.environ.get("SONJ_URL", "http://127.0.0.1:8766")


def _http(path: str, method: str = "GET", body: dict | None = None):
    import requests

    url = BASE_URL + path
    # Local control plane only: validate scheme and host explicitly before sending.
    from urllib.parse import urlparse

    parsed = urlparse(url)
    if parsed.scheme != "http" or parsed.hostname not in {"127.0.0.1", "localhost"}:
        raise ValueError("SONJ talks to the local engine only, got: " + url)
    try:
        resp = requests.request(method, url, json=body, timeout=15)
        try:
            return resp.status_code, resp.json()
        except ValueError:
            return resp.status_code, resp.text
    except requests.RequestException as exc:
        return 0, f"{type(exc).__name__}: {exc}"


def _alive() -> bool:
    status, _ = _http("/api/health")
    return status == 200


def _engine_process() -> str:
    """Return the PID of a running SONJ engine, cross-platform.

    Uses psutil (works on Windows / Linux / macOS). Falls back to a Windows
    PowerShell query only if psutil is unavailable on Windows.
    """
    try:
        import psutil

        for proc in psutil.process_iter(["pid", "cmdline"]):
            try:
                cmdline = " ".join(proc.info.get("cmdline") or [])
            except (psutil.Error, OSError):
                continue
            if "server.py" in cmdline and (
                ROOT.name in cmdline or str(ROOT) in cmdline
            ):
                return str(proc.info["pid"])
        return ""
    except Exception:  # noqa: BLE001 - psutil unavailable; fall back below
        if os.name != "nt":
            return ""
    try:
        out = subprocess.run(
            [
                "powershell",
                "-NoProfile",
                "-Command",
                (
                    "Get-CimInstance Win32_Process -Filter \"Name like 'python%'\" | "
                    "Where-Object { $_.CommandLine -like '*server.py*' } | "
                    "Select-Object -ExpandProperty ProcessId"
                ),
            ],
            capture_output=True,
            text=True,
            timeout=20,
            check=False,
        )
        pid = out.stdout.strip().splitlines()
        return pid[0] if pid else ""
    except Exception:  # noqa: BLE001
        return ""


def cmd_latency(args: argparse.Namespace) -> int:
    """Live per-stage latency percentiles; --export writes out/latency_report.md."""
    status, body = _http("/api/latency")
    if status != 200 or not isinstance(body, dict):
        print("engine not reachable at " + BASE_URL)
        return 2
    stages = body.get("stages") or {}
    print(f"latency over {stages.get('turns', 0)} turns (ms):")
    print(f"  {'stage':<16} {'p50':>8} {'p95':>8} {'max':>8} {'n':>5}")
    for key in (
        "asr_ms",
        "first_token_ms",
        "ttfa_ms",
        "e2e_ms",
        "filler_ms",
        "llm_ms",
        "tts_ms",
        "total_ms",
    ):
        r = stages.get(key)
        if not r:
            continue
        print(
            f"  {key:<16} {r.get('p50', 0):>8.0f} {r.get('p95', 0):>8.0f} "
            f"{r.get('max', 0):>8.0f} {r.get('n', 0):>5}"
        )
    if getattr(args, "export", False):
        st, resp = _http("/api/latency_report", "POST", {})
        if st == 200 and isinstance(resp, dict) and resp.get("ok"):
            print("report: " + str(resp.get("path")))
        else:
            print("report export failed")
            return 1
    return 0


def cmd_status(args: argparse.Namespace) -> int:
    status, state = _http("/api/state")
    if status != 200 or not isinstance(state, dict):
        payload = {"engine": "down", "url": BASE_URL, "detail": str(state)[:120]}
        print(
            json.dumps(payload, indent=2)
            if args.json
            else f"engine: DOWN at {BASE_URL}"
        )
        return 2
    settings = state.get("settings", {})
    tasks = state.get("tasks", [])
    summary = {
        "engine": "up",
        "url": BASE_URL,
        "status": state.get("status"),
        "mode": state.get("mode"),
        "speaking": state.get("speaking"),
        "turns": len(state.get("turns", [])),
        "events": len(state.get("events", [])),
        "agents_running": sum(1 for t in tasks if t.get("state") == "running"),
        "agents_total": len(tasks),
        "memory_mb": round(float(state.get("memory_mb") or 0), 1),
        "model": settings.get("model"),
        "persona_voices": {
            "nova": settings.get("nova_voice"),
            "piattos": settings.get("piattos_voice"),
        },
        "stop_words": len(settings.get("barge_triggers") or []),
        "error": state.get("error") or "",
    }
    if args.json:
        print(json.dumps(summary, indent=2))
        return 0
    print(f"engine    : {summary['engine']} ({summary['url']})")
    print(
        f"state     : {summary['status']} | mode {summary['mode']} | speaking {summary['speaking']}"
    )
    print(f"turns     : {summary['turns']} | events {summary['events']}")
    print(
        f"agents    : {summary['agents_running']} running / {summary['agents_total']} total"
    )
    print(f"memory    : {summary['memory_mb']} MB")
    print(f"model     : {summary['model']}")
    print(
        f"voices    : Nova {summary['persona_voices']['nova']} | Piattos {summary['persona_voices']['piattos']}"
    )
    print(f"stop words: {summary['stop_words']}")
    if summary["error"]:
        print(f"error     : {summary['error']}")
    return 0


def cmd_doctor(args: argparse.Namespace) -> int:
    checks: list[tuple[str, bool, str]] = []

    v = sys.version_info
    checks.append(("python >= 3.10", v >= (3, 10), f"{v.major}.{v.minor}.{v.micro}"))
    for module in ("numpy", "sounddevice", "requests", "sherpa_onnx"):
        try:
            __import__(module)
            checks.append((f"import {module}", True, "ok"))
        except Exception as exc:  # noqa: BLE001
            checks.append((f"import {module}", False, f"{type(exc).__name__}"))
    models = ROOT / "models"
    checks.append(("models directory", models.is_dir(), str(models)))
    checks.append(
        ("skills directory", (ROOT / "skills").is_dir(), str(ROOT / "skills"))
    )
    for name in ("config.json",):
        checks.append((name, (ROOT / name).is_file(), str(ROOT / name)))
    checks.append(
        (
            "ledger REQUESTS.md",
            (PROJECT / "REQUESTS.md").is_file(),
            str(PROJECT / "REQUESTS.md"),
        )
    )
    checks.append(
        (
            "objectives OBJECTIVES.md",
            (PROJECT / "OBJECTIVES.md").is_file(),
            str(PROJECT / "OBJECTIVES.md"),
        )
    )
    try:
        usage = shutil.disk_usage(PROJECT)
        free_gb = usage.free / 1e9
        checks.append(("disk free >= 2 GB", free_gb >= 2, f"{free_gb:.1f} GB free"))
    except OSError as exc:
        checks.append(("disk free", False, str(exc)))
    checks.append(("engine reachable", _alive(), BASE_URL))
    try:
        import psutil

        rss = psutil.virtual_memory()
        checks.append(
            ("RAM >= 2 GB", rss.total / 1e9 >= 2, f"{rss.total / 1e9:.1f} GB total")
        )
    except Exception:  # noqa: BLE001
        checks.append(("RAM check", True, "psutil not installed - skipped"))
    try:
        import sounddevice as sd

        devs = sd.query_devices()
        ins = sum(1 for d in devs if d.get("max_input_channels", 0) > 0)
        outs = sum(1 for d in devs if d.get("max_output_channels", 0) > 0)
        checks.append(("audio devices", ins > 0 and outs > 0, f"{ins} in / {outs} out"))
    except Exception as exc:  # noqa: BLE001
        checks.append(("audio devices", False, f"{type(exc).__name__}"))

    failed = 0
    print(
        f"sonj doctor - {platform.system()} {platform.release()} on {platform.machine()}"
    )
    for name, ok, detail in checks:
        print(f"  [{'PASS' if ok else 'FAIL'}] {name:28s} {detail}")
        failed += 0 if ok else 1
    print(f"{len(checks) - failed}/{len(checks)} checks passed")
    return 1 if failed else 0


def _active_profile() -> str:
    profile = os.environ.get("SONJ_PROFILE", "laptop").strip().lower()
    return profile if profile in {"laptop", "pi"} else "laptop"


def _engine_rss_mb() -> float | None:
    status, state = _http("/api/state")
    if status != 200 or not isinstance(state, dict):
        return None
    raw_value = state.get("memory_mb")
    if not isinstance(raw_value, (int, float, str)):
        return None
    try:
        value = float(raw_value)
    except ValueError:
        return None
    return value if value >= 0 else None


def cmd_profile(args: argparse.Namespace) -> int:
    try:
        pi_profile = json.loads((ROOT / "pi_profile.json").read_text(encoding="utf-8"))
        laptop_config = json.loads((ROOT / "config.json").read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        print(f"could not read profile configuration: {exc}")
        return 2
    if not isinstance(pi_profile, dict) or not isinstance(laptop_config, dict):
        print("profile configuration must contain JSON objects")
        return 2

    active = _active_profile()
    requested = getattr(args, "profile", None)
    selected = "pi" if args.pi else (requested or active)
    tts = pi_profile.get("tts", {})
    asr = pi_profile.get("asr", {})
    fallback = asr.get("fallback", {}) if isinstance(asr, dict) else {}
    budget_mb = float(pi_profile.get("memory_budget_mb", 1200))
    laptop_voices = ", ".join(
        str(laptop_config.get(key))
        for key in ("nova_voice", "piattos_voice")
        if laptop_config.get(key)
    ) or str(laptop_config.get("tts_voice", "unknown"))

    print("SONJ profile comparison")
    print(
        f"active profile: {active} (SONJ_PROFILE={os.environ.get('SONJ_PROFILE', 'unset')})"
    )
    print(f"selected view : {selected}" + (" (--pi)" if args.pi else ""))
    print(
        "laptop config : "
        f"voices={laptop_voices}; ASR={laptop_config.get('asr_backend', 'unknown')}; "
        f"idle_unload_s={laptop_config.get('idle_unload_s', 'unknown')}"
    )
    print(
        f"Pi profile    : {pi_profile.get('target', 'Pi')} | "
        f"TTS={tts.get('backend', 'unknown')} {tts.get('family', '')} {tts.get('voice', 'unknown')} | "
        f"ASR={asr.get('backend', 'unknown')} {asr.get('model', 'unknown')} "
        f"{asr.get('quantization', '')}"
    )
    if fallback:
        print(
            f"ASR fallback  : {fallback.get('backend', 'unknown')} "
            f"{fallback.get('model', 'unknown')} {fallback.get('quantization', '')}"
        )
    print(
        f"Pi runtime    : idle unload {pi_profile.get('idle_unload_s', 'unknown')}s; "
        f"browser {'on' if pi_profile.get('browser_enabled') else 'off'}; "
        f"RSS budget {budget_mb:.0f} MB ({budget_mb / 1000:.1f} GB)"
    )
    if active == "pi":
        print(
            "note          : SONJ_PROFILE is an env handoff; server.py load_settings() still reads config.json."
        )
    rss_mb = _engine_rss_mb()
    if rss_mb is None:
        print(
            f"engine RSS    : unavailable; budget {budget_mb:.1f} MB (engine not reachable or RSS missing)"
        )
    else:
        percent = 100 * rss_mb / budget_mb if budget_mb else 0
        result = "within budget" if rss_mb <= budget_mb else "OVER budget"
        print(
            f"engine RSS    : {rss_mb:.1f} MB / {budget_mb:.1f} MB ({percent:.1f}%) - {result}"
        )
    return 0


def _systemd_unit() -> str:
    if os.name == "posix":
        user = getpass.getuser()
        service_root = ROOT.resolve()
        python = Path(sys.executable).resolve()
    else:
        user = os.environ.get("SONJ_PI_USER", "lcsoneja")
        service_root = Path("/home") / user / "me3"
        python = service_root / ".venv" / "bin" / "python3"
    server = shlex.quote((service_root / "server.py").as_posix())
    python = shlex.quote(python.as_posix())
    workdir = shlex.quote(service_root.as_posix())
    return (
        "[Unit]\n"
        "Description=SONJ voice agent (Raspberry Pi profile)\n"
        "After=network-online.target sound.target\n"
        "Wants=network-online.target\n\n"
        "[Service]\n"
        f"User={user}\n"
        f"WorkingDirectory={workdir}\n"
        "Environment=SONJ_PROFILE=pi\n"
        "Environment=PYTHONUNBUFFERED=1\n"
        f"ExecStart=/usr/bin/env SONJ_PROFILE=pi {python} {server} --host 127.0.0.1 --port 8765 --no-browser\n"
        "Restart=always\n"
        "RestartSec=5s\n\n"
        "[Install]\n"
        "WantedBy=multi-user.target\n"
    )


def _apply_systemd_unit(unit: str) -> int:
    if os.name != "posix":
        print("--apply is supported only on the Pi/Linux systemd host")
        return 2
    install = subprocess.run(
        ["sudo", "tee", "/etc/systemd/system/sonj.service"],
        input=unit,
        capture_output=True,
        text=True,
        check=False,
    )
    if install.returncode:
        print(f"could not install sonj.service: {install.stderr.strip()}")
        return install.returncode
    for command in (
        ["sudo", "systemctl", "daemon-reload"],
        ["sudo", "systemctl", "enable", "--now", "sonj.service"],
    ):
        result = subprocess.run(command, capture_output=True, text=True, check=False)
        if result.returncode:
            print(f"command failed ({result.returncode}): {' '.join(command)}")
            if result.stderr.strip():
                print(result.stderr.strip())
            return result.returncode
    print("enabled and started sonj.service")
    return 0


def cmd_install_pi(args: argparse.Namespace) -> int:
    mode = "apply" if args.apply else "dry run"
    unit = _systemd_unit()
    print(f"sonj.service ({mode}):")
    print(unit, end="")
    if not args.apply:
        doctor_rc = cmd_doctor(args)
        print("dry run only; pass --apply on the Pi to install and enable this service")
        return doctor_rc

    install_rc = _apply_systemd_unit(unit)
    if install_rc:
        cmd_doctor(args)
        return install_rc
    for _ in range(40):
        if _alive():
            break
        time.sleep(0.5)
    return cmd_doctor(args)


def _engine_app_dir() -> str:
    status, state = _http("/api/state")
    if status == 200 and isinstance(state, dict):
        return str(state.get("app_dir") or "")
    return ""


def cmd_start(args: argparse.Namespace) -> int:
    requested_profile = getattr(args, "profile", None)
    active_profile = requested_profile or _active_profile()
    same_engine = False
    if _alive():
        running_dir = _engine_app_dir()
        if not running_dir or Path(running_dir).resolve() == ROOT.resolve():
            print(f"engine already running at {BASE_URL}")
            same_engine = True
        else:
            print(
                f"port {args.port} is in use by another engine ({running_dir}); leaving it untouched. Pass a free --port."
            )
            return 2
    if not same_engine:
        log_out = (ROOT / "out" / "server_run.txt").open("a", encoding="utf-8")
        log_err = (ROOT / "out" / "server_err.txt").open("a", encoding="utf-8")
        server_command = [
            sys.executable,
            str(ROOT / "server.py"),
            "--port",
            args.port,
            "--force",
        ]
        if active_profile == "pi":
            server_command.append("--no-browser")
        env = os.environ.copy()
        if requested_profile:
            env["SONJ_PROFILE"] = requested_profile
        creationflags = (
            getattr(subprocess, "CREATE_NO_WINDOW", 0) if os.name == "nt" else 0
        )
        out = subprocess.Popen(
            server_command,
            cwd=str(ROOT),
            stdout=log_out,
            stderr=log_err,
            env=env,
            creationflags=creationflags,
        )
        log_out.close()
        log_err.close()
        (ROOT / "out" / "engine.pid").write_text(str(out.pid), encoding="utf-8")

        for _ in range(40):
            time.sleep(0.5)
            if _alive():
                break
        print(
            "engine:",
            "up" if _alive() else f"not answering on {BASE_URL} (pid {out.pid})",
        )
    return 0 if _alive() else 2


def _pythonw() -> str:
    """The windowed interpreter, so opening the app does not flash a console."""
    candidate = Path(sys.executable).with_name("pythonw.exe")
    return str(candidate) if candidate.exists() else sys.executable


def cmd_ui(args: argparse.Namespace) -> int:
    # Kept as an alias. RULE-1 (owner): bare `sonj` opens the agent console window.
    return cmd_app(args)


def cmd_app(args: argparse.Namespace) -> int:
    """Open the agent console as a desktop window.

    Preference: the native console window (pywebview + Edge WebView2) that shows the
    full web console - the same page the engine serves at /, in a real window with no
    browser chrome. When that runtime is absent the Tk standalone build is used
    instead. It is never opened as a browser tab, so RULE-1 and RULE-36 still hold.
    """
    running_dir = _engine_app_dir() if _alive() else ""
    wrong_engine = bool(running_dir and Path(running_dir).resolve() != ROOT.resolve())
    if not _alive() or wrong_engine:
        rc = cmd_start(argparse.Namespace(port=args.port, profile=args.profile))
        if rc != 0:
            return rc
    native = False
    try:
        sys.path.insert(0, str(ROOT))
        from app_window import native_ready

        native = native_ready()
    except Exception as exc:  # noqa: BLE001 - a missing native runtime just falls back
        print("native console unavailable:", type(exc).__name__, exc)
    if native:
        proc = subprocess.Popen(
            [_pythonw(), str(ROOT / "app_window.py"), "--port", str(args.port)],
            cwd=str(ROOT),
        )
        print(f"agent console window started (pid {proc.pid})")
        return 0
    proc = subprocess.Popen([_pythonw(), str(ROOT / "sonj_tk.py")], cwd=str(ROOT))
    print(f"standalone app started (pid {proc.pid})")
    return 0


def _dash_frame(s: dict[str, Any]) -> str:
    """One screen of the live dashboard: tokens, status, agents, doctor, trace."""
    settings = s.get("settings") or {}
    agents = s.get("agents") or {}
    active = {
        (a.get("role") or a.get("name")): (
            a.get("state") or a.get("status") or "active"
        )
        for a in (s.get("agents_active") or [])
    }
    doctor = s.get("doctor") or {}
    if doctor.get("enabled") is False:
        doc = "off"
    elif doctor.get("kill_switch"):
        doc = "paused"
    else:
        doc = "ok" if doctor.get("last_run") else "idle"
    session = Path(str(s.get("session_dir") or "")).name or "\u2014"
    dollars = "$" + str(round(float(s.get("spend_usd") or 0.0), 4))
    out = [
        "ME3 dashboard   " + datetime.now().strftime("%H:%M:%S"),
        "=" * 68,
        " status      : " + str(s.get("status", "?")),
        " main dir    : " + str(s.get("cwd", "?")),
        " model       : " + str(settings.get("model", "?")),
        " session     : " + session,
        " tokens used : " + format(int(s.get("spend_tokens") or 0), ","),
        " spend       : " + dollars,
        " memory      : " + str(round(float(s.get("memory_mb") or 0))) + " MB",
        " doctor      : "
        + doc
        + "   ("
        + str(len(s.get("incidents") or []))
        + " recent incidents)",
        "-" * 68,
        " agents",
    ]
    for role, entry in agents.items():
        row = entry if isinstance(entry, dict) else {}
        out.append(
            "   "
            + str(role)[:11].ljust(11)
            + " "
            + str(row.get("model", "?"))[:34].ljust(34)
            + " "
            + "think:"
            + str(row.get("thinking", "-"))[:6].ljust(6)
            + " "
            + str(active.get(role, "idle"))
        )
    out += ["-" * 68, " reasoning trace"]
    for th in (s.get("thinking") or [])[-8:]:
        out.append(
            "   "
            + str(th.get("ts", ""))
            + " ["
            + str(th.get("kind", "step"))
            + "] "
            + str(th.get("msg", ""))[:64]
        )
    return "\n".join(out)


def _install_terminal_close_shutdown() -> None:
    """Windows: closing the launching terminal window stops the owned engine too."""
    if os.name != "nt":
        return
    try:
        import ctypes
        from ctypes import wintypes

        handler_type = ctypes.WINFUNCTYPE(wintypes.BOOL, wintypes.DWORD)

        def _on_console(event: int) -> bool:
            if event in (2, 5, 6):  # CTRL_CLOSE, CTRL_LOGOFF, CTRL_SHUTDOWN
                import contextlib

                with contextlib.suppress(Exception):
                    _http("/api/barge", "POST", {})
                    _http("/api/stop", "POST", {})
                    _http("/api/shutdown", "POST", {})
                time.sleep(2.5)
                return True
            return False

        keep = handler_type(_on_console)
        ctypes.windll.kernel32.SetConsoleCtrlHandler(keep, True)
        globals()["_SONJ_CONSOLE_HANDLER"] = keep
    except Exception:  # noqa: BLE001 - non-console hosts need no handler
        return


def cmd_dash(args: argparse.Namespace) -> int:
    """One live dashboard in the terminal (Ctrl+C to quit).

    Redraws in place - cursor home, clear the tail - so it stays a single view that
    updates instead of printing a new line every second. --once prints one frame.
    """
    running_dir = _engine_app_dir() if _alive() else ""
    wrong_engine = bool(running_dir and Path(running_dir).resolve() != ROOT.resolve())
    if not _alive() or wrong_engine:
        rc = cmd_start(argparse.Namespace(port=args.port, profile=args.profile))
        if rc != 0:
            return rc
    once = bool(getattr(args, "once", False))
    if not once:
        _install_terminal_close_shutdown()
    interval = max(0.3, float(getattr(args, "interval", 1.0) or 1.0))
    try:
        while True:
            status, body = _http("/api/state")
            if status == 200 and isinstance(body, dict):
                frame = _dash_frame(body)
            else:
                frame = "sonj dash: engine not responding at " + BASE_URL
            if once:
                print(frame)
                return 0
            sys.stdout.write("\x1b[2J\x1b[H" + frame + "\x1b[J")
            sys.stdout.flush()
            time.sleep(interval)
    except KeyboardInterrupt:
        print()
        if not once:
            _http("/api/barge", "POST", {})
            _http("/api/stop", "POST", {})
            _http("/api/shutdown", "POST", {})
        return 0


def cmd_ask(args: argparse.Namespace) -> int:
    # pi.dev-style pre-flags: --model switches before the turn; --session resumes a past
    # session first; --print (the default behaviour) keeps stdout to the answer alone.
    if getattr(args, "model", ""):
        status_m, body_m = _http("/api/settings", "POST", {"model": args.model})
        if status_m != 200 or (isinstance(body_m, dict) and body_m.get("ok") is False):
            print("model switch failed:", str(body_m)[:80])
            return 2
    resume_id = getattr(args, "session", "") or getattr(args, "resume", "")
    if resume_id:
        # #26/#34: resume makes the past session current (new turns append there);
        # load reads it back. Both routes restored 2026-11-16.
        status_s, _ = _http("/api/session/resume", "POST", {"id": resume_id})
        if status_s != 200:
            print(f"session {resume_id} not resumed (status {status_s})")
            return 2
    status, _ = _http("/api/send", "POST", {"text": args.text})
    if status != 200:
        print(f"send failed (engine {status}); is it running?")
        return 2
    deadline = time.time() + args.timeout
    # Beta finding 1: comparing against 0 accepted any turn already in the session, so a
    # fresh "ask" could print the previous answer. Snapshot the count at send time and
    # only accept a turn recorded after this request.
    before = 0
    _, state = _http("/api/state")
    if isinstance(state, dict):
        before = len(state.get("turns", []))
    seen = before
    while time.time() < deadline:
        time.sleep(0.4)
        _, state = _http("/api/state")
        if not isinstance(state, dict):
            continue
        turns = state.get("turns", [])
        if len(turns) > seen:
            seen = len(turns)
            turn = turns[-1]
            if not str(turn.get("assistant_text", "")).strip():
                continue  # the turn exists but no answer yet - keep waiting
            print(turn.get("assistant_text", ""))
            if args.json:
                print(json.dumps(turn, indent=2))
            return 0
    print(f"no answer within {args.timeout}s (still working?)")
    return 3


def cmd_say(args: argparse.Namespace) -> int:
    import sounddevice as sd

    sys.path.insert(0, str(ROOT))

    from server import Session, load_settings

    session = cast(Any, Session(load_settings()))
    voice = session._build_tts(session._active_voice())
    result = voice._engine.generate(args.text, sid=voice._sid, speed=1.0)
    sd.play(np.asarray(result.samples, dtype="float32"), int(result.sample_rate))
    sd.wait()
    print(f"spoke via {voice.voice_name} (sid {voice._sid})")
    return 0


def cmd_listen(args: argparse.Namespace) -> int:
    if not _alive():
        print("engine is not running - start it with: sonj")
        return 2
    _http("/api/stop", "POST", {})
    _http("/api/listen", "POST", {"mode": "continuous"})
    print("listening (hands-free). Ctrl+C to stop.")
    try:
        while True:
            time.sleep(1)
            _, state = _http("/api/state")
            if isinstance(state, dict) and state.get("error"):
                print("error:", state["error"])
    except KeyboardInterrupt:
        _http("/api/stop", "POST", {})
        print("stopped")
    return 0


def cmd_agents(args: argparse.Namespace) -> int:
    _, state = _http("/api/state")
    if not isinstance(state, dict):
        print("engine is not running")
        return 2
    tasks = state.get("tasks", [])
    agents = state.get("agents", {})
    print("roster (agents.json):")
    for role, cfg in (agents or {}).items():
        model = cfg.get("model", "?") if isinstance(cfg, dict) else "?"
        print(f"  {role:10s} {model}")
    print(f"tasks: {len(tasks)}")
    for t in tasks[:10]:
        print(
            f"  [{t.get('state', '?'):8s}] {str(t.get('title') or t.get('description') or t.get('id'))[:60]}"
        )
    return 0


def cmd_bench(args: argparse.Namespace) -> int:
    script = ROOT / "turn_bench.py"
    if not script.is_file():
        print("turn_bench.py not found")
        return 2
    cmd = [sys.executable, str(script)]
    print("running the standardized latency method (turn_bench.py)")
    out = subprocess.run(
        cmd, cwd=str(ROOT), capture_output=True, text=True, check=False, timeout=900
    )
    tail = (out.stdout or "").strip().splitlines()[-12:]
    print("\n".join(tail))
    print(f"exit {out.returncode}; full output: {ROOT.name}/out/")
    return 0 if out.returncode == 0 else 1


def cmd_logs(args: argparse.Namespace) -> int:
    path = ROOT / "out" / "server_run.txt"
    if not path.is_file():
        print("no log yet:", path)
        return 2
    text = path.read_text(encoding="utf-8", errors="replace").splitlines()
    lines = text[-args.lines :]
    print("\n".join(lines))
    if args.follow:
        print("(following; Ctrl+C to stop)")
        pos = path.stat().st_size
        try:
            while True:
                time.sleep(1)
                if path.stat().st_size > pos:
                    with open(path, encoding="utf-8", errors="replace") as fh:
                        fh.seek(pos)
                        print(fh.read(), end="")
                        pos = fh.tell()
        except KeyboardInterrupt:
            pass
    return 0


def _kill_pid_tree(pid: int) -> None:
    if pid <= 0:
        return
    if os.name == "nt":
        subprocess.run(
            ["taskkill", "/PID", str(pid), "/T", "/F"],
            capture_output=True,
            timeout=5,
            check=False,
            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
        )
    else:
        import contextlib

        with contextlib.suppress(OSError):
            os.kill(pid, signal.SIGTERM)


def cmd_stop(args: argparse.Namespace) -> int:
    _http("/api/barge", "POST", {})
    _http("/api/stop", "POST", {})
    _http("/api/shutdown", "POST", {})
    for _ in range(15):
        if not _alive():
            break
        time.sleep(0.2)
    pid_file = ROOT / "out" / "engine.pid"
    if pid_file.is_file():
        try:
            pid = int(pid_file.read_text(encoding="utf-8").strip())
            if _alive():
                _kill_pid_tree(pid)
            pid_file.unlink(missing_ok=True)
        except (OSError, ValueError):
            pid_file.unlink(missing_ok=True)
    if _alive():
        stray = _engine_process()
        if stray and stray.isdigit():
            _kill_pid_tree(int(stray))
    sup = ROOT / "out" / "supervisor.pid"
    try:
        sup.unlink(missing_ok=True)
    except OSError as exc:
        print(f"note: supervisor pid cleanup skipped ({exc})")
    print("engine stopped")
    return 0


def cmd_restart(args: argparse.Namespace) -> int:
    pid = _engine_process()
    if pid:
        try:
            os.kill(int(pid), signal.SIGTERM)
            print(f"sent stop to engine pid {pid}")
            time.sleep(2)
        except Exception as exc:  # noqa: BLE001
            print("could not stop:", exc)
    return cmd_start(argparse.Namespace(port=args.port, profile=args.profile))


BENCH_PROMPTS = (
    "In one sentence, what is a voice command model?",
    "Name three uses of an offline speech recognizer.",
    "What time is it right now? Use the get_time tool.",
    "Explain in two sentences why streaming speech lowers perceived latency.",
    "Summarise in one sentence: the Pi has 4 GB of RAM and runs a local 7B router.",
)
BENCH_QUICK_PROMPTS = (
    0,
    2,
)  # a chat answer and a tool call, the two shapes that matter


def _bench_models(selected: str) -> list[str]:
    """M3.4: every model the roster names (or the explicit list)."""
    if selected.strip():
        return [m.strip() for m in selected.split(",") if m.strip()]
    sys.path.insert(0, str(ROOT))
    from server import load_agents_config

    roster = load_agents_config()
    models: list[str] = []
    for entry in roster.values():
        entry = entry if isinstance(entry, dict) else {}
        for model in [entry.get("model"), *(entry.get("fallbacks") or [])]:
            text = str(model or "").strip()
            if text and text not in models:
                models.append(text)
    return models


def cmd_bench_roles(args: argparse.Namespace) -> int:
    """M3.4 (D-14): which free model actually works best for each role.

    Runs the same fixed prompt suite against every candidate model through a
    normal Agent turn (tools included), records latency and whether the answer
    passed, writes BENCH_ROLES.md and names the winner. The suite is small on
    purpose: it is a fitness check for role routing, not a benchmark harness.
    --quick runs the two prompts that exercise both shapes (chat and tool call)
    on at most two models, which is the smoke version.
    """
    sys.path.insert(0, str(ROOT))
    from agent import Agent
    from llm import LlmError, create_llm, find_openrouter_key
    from toolbox import ToolBox

    models = _bench_models(args.models)
    if args.quick:
        models = models[:2]
    if not models:
        print("no models: agents.json has no roster and --models was empty")
        return 2
    prompts = (
        [BENCH_PROMPTS[i] for i in BENCH_QUICK_PROMPTS]
        if args.quick
        else list(BENCH_PROMPTS)
    )
    key = find_openrouter_key()
    rows: list[dict[str, Any]] = []
    print(f"bench-roles: {len(models)} model(s) x {len(prompts)} prompt(s)")
    for model in models:
        client = create_llm("openrouter", model, key)
        agent = Agent(client, ToolBox())
        for index, prompt in enumerate(prompts, start=1):
            started = time.perf_counter()
            note = "ok"
            answer = ""
            tools_used = 0
            try:
                reply = agent.respond(prompt, raise_errors=True)
                answer = str(reply.text or "").strip()
                tools_used = len(reply.tools)
                if not answer:
                    note = "empty answer"
                elif index == 3 and not any(c.isdigit() for c in answer):
                    # The tool prompt must come back with a time in it.
                    note = "no time in tool answer"
            except LlmError as exc:
                note = "model error: " + str(exc)[:60]
            seconds = time.perf_counter() - started
            rows.append(
                {
                    "model": model,
                    "prompt": index,
                    "seconds": round(seconds, 2),
                    "tools": tools_used,
                    "note": note,
                    "answer": answer[:120],
                }
            )
            print(
                f"  {model[:44]:44s} p{index} {seconds:6.2f}s tools={tools_used} {note}"
            )
    out = (ROOT.parent if ROOT.name == "me3_laptop" else ROOT) / "BENCH_ROLES.md"
    by_model: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        by_model.setdefault(row["model"], []).append(row)
    lines = [
        "# BENCH_ROLES - which model carries which role",
        "",
        f"Run {datetime.now().strftime('%Y-%m-%d %H:%M')} - "
        f"{len(models)} model(s) x {len(prompts)} prompt(s) - suite: "
        + ("quick (chat + tool call)" if args.quick else "full (5 prompts)"),
        "",
        "| model | prompts | passed | avg s | tools |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]
    for model, group in by_model.items():
        passed = sum(1 for r in group if r["note"] == "ok")
        avg = sum(r["seconds"] for r in group) / max(1, len(group))
        lines.append(
            f"| {model} | {len(group)} | {passed}/{len(group)} | {avg:.2f} | "
            f"{sum(r['tools'] for r in group)} |"
        )
    lines += ["", "## Per prompt", ""]
    for row in rows:
        lines.append(
            f"- {row['model']} p{row['prompt']} ({row['seconds']}s, "
            f"tools={row['tools']}): {row['note']} - {row['answer'][:100]!r}"
        )
    winners = sorted(
        by_model,
        key=lambda m: (
            -sum(1 for r in by_model[m] if r["note"] == "ok"),
            sum(r["seconds"] for r in by_model[m]) / max(1, len(by_model[m])),
        ),
    )
    if winners:
        lines += ["", f"**Winner: {winners[0]}** - most passing prompts, then fastest."]
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"wrote {out}")
    if winners:
        print(f"winner: {winners[0]}")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="sonj", description="SONJ - voice agent command line"
    )
    parser.add_argument("--port", default="8766", help="engine port (default 8766)")
    parser.add_argument(
        "--profile",
        choices=("laptop", "pi"),
        default=None,
        help="profile forwarded to the server as SONJ_PROFILE",
    )
    parser.add_argument(
        "--no-dash",
        action="store_true",
        help="bare sonj: open the window but do not keep a terminal dashboard",
    )
    sub = parser.add_subparsers(dest="cmd")

    p_status = sub.add_parser("status", help="engine, audio, agents, memory")
    p_status.add_argument("--json", action="store_true")
    p_status.set_defaults(func=cmd_status)

    p_doctor = sub.add_parser("doctor", help="can this machine run the agent?")
    p_doctor.set_defaults(func=cmd_doctor)

    p_profile = sub.add_parser("profile", help="show the active and Pi profiles")
    p_profile.add_argument(
        "--pi", action="store_true", help="select the Pi profile view"
    )
    p_profile.set_defaults(func=cmd_profile)

    p_install_pi = sub.add_parser(
        "install-pi", help="print the Pi systemd service and doctor result"
    )
    p_install_pi.add_argument(
        "--apply", action="store_true", help="install and enable the systemd service"
    )
    p_install_pi.set_defaults(func=cmd_install_pi)

    p_ui = sub.add_parser("ui", help="alias for app - open the agent console window")
    p_ui.set_defaults(func=cmd_ui)

    p_app = sub.add_parser(
        "app", help="start the engine and open the agent console window"
    )
    p_app.set_defaults(func=cmd_app)

    p_dash = sub.add_parser(
        "dash", help="live one-view status dashboard in the terminal"
    )
    p_dash.add_argument("--once", action="store_true", help="print one frame and exit")
    p_dash.add_argument(
        "--interval", type=float, default=1.0, help="seconds between redraws"
    )
    p_dash.set_defaults(func=cmd_dash)

    p_ask = sub.add_parser("ask", help="one turn, print the answer")
    p_ask.add_argument("text")
    p_ask.add_argument("--json", action="store_true")
    p_ask.add_argument("--timeout", type=float, default=60.0)
    p_ask.add_argument("--model", default="", help="switch the model first (pi-style)")
    p_ask.add_argument("--session", default="", help="resume a past session id first")
    p_ask.add_argument(
        "--print", action="store_true", help="answer only on stdout (default)"
    )
    p_ask.add_argument(
        "-c",
        "--continue-last",
        action="store_true",
        help="continue the last session (default)",
    )
    p_ask.add_argument(
        "-r", "--resume", metavar="ID", default="", help="resume session id"
    )
    p_ask.set_defaults(func=cmd_ask)

    p_say = sub.add_parser("say", help="speak text (voice test)")
    p_say.add_argument("text")
    p_say.set_defaults(func=cmd_say)

    p_listen = sub.add_parser("listen", help="hands-free session")
    p_listen.set_defaults(func=cmd_listen)

    p_agents = sub.add_parser("agents", help="roster and running tasks")
    p_agents.set_defaults(func=cmd_agents)

    p_bench = sub.add_parser("bench", help="standardized latency run")
    p_bench.add_argument("--turns", type=int, default=20)
    p_bench.set_defaults(func=cmd_bench)

    p_bench_roles = sub.add_parser(
        "bench-roles", help="5 prompts x every role model -> BENCH_ROLES.md"
    )
    p_bench_roles.add_argument(
        "--quick", action="store_true", help="2 models x 2 prompts (fast smoke)"
    )
    p_bench_roles.add_argument(
        "--models",
        default="",
        help="comma-separated model ids (default: every model in agents.json)",
    )
    p_bench_roles.set_defaults(func=cmd_bench_roles)

    p_logs = sub.add_parser("logs", help="engine log")
    p_logs.add_argument("--lines", type=int, default=40)
    p_logs.add_argument("--follow", action="store_true")
    p_logs.set_defaults(func=cmd_logs)

    p_stop = sub.add_parser("stop", help="stop listening")
    p_stop.set_defaults(func=cmd_stop)

    p_restart = sub.add_parser("restart", help="restart the engine")
    p_restart.set_defaults(func=cmd_restart)

    p_latency = sub.add_parser("latency", help="per-stage latency percentiles")
    p_latency.add_argument(
        "--export", action="store_true", help="write out/latency_report.md"
    )
    p_latency.set_defaults(func=cmd_latency)

    args = parser.parse_args(argv)
    if not os.environ.get("SONJ_URL"):
        global BASE_URL
        BASE_URL = f"http://127.0.0.1:{args.port}"
    if not args.cmd:
        # RULE-1 (owner, 2026-11-15): bare sonj opens the STANDALONE app. Never a browser.
        if (args.profile or _active_profile()) == "pi":
            return cmd_start(args)
        rc = cmd_app(args)
        if rc != 0 or getattr(args, "no_dash", False) or not sys.stdout.isatty():
            return rc
        # the launching shell then shows live status: one view, redrawn in place
        return cmd_dash(
            argparse.Namespace(
                port=args.port, profile=args.profile, once=False, interval=1.0
            )
        )
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
