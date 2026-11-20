"""S2/S3: the doctor and the verifier.

Detect -> repair from a fixed catalogue -> verify -> roll back if the repair made
things worse. The doctor never rewrites code: every action here is a named,
reversible operation on processes, settings, files or engines. The verifier is a
separate pass that re-runs the checks after a repair and returns a verdict, so a
repair is only reported as fixed when an independent check agrees.
"""

from __future__ import annotations

import contextlib
import json
import os
import shutil
import socket
import sys
import threading
import time
from collections.abc import Callable
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

BASE = Path(__file__).resolve().parent
OUT_DIR = BASE / "out"
CONFIG_PATH = BASE / "config.json"
ENGINE_PID = OUT_DIR / "engine.pid"


def fast_boot_scan() -> dict[str, Any]:
    """S4: quick log/incident scan at engine start (<1s, no LLM).

    Writes out/startup_check.md and returns a summary the UI can surface, so
    problems present at opening are detected and logged without a full diagnosis.
    """
    from datetime import datetime

    errors: list[str] = []
    warnings: list[str] = []
    log = OUT_DIR / "app.log"
    if log.is_file():
        try:
            lines = log.read_text(encoding="utf-8", errors="replace").splitlines()[
                -400:
            ]
        except OSError:
            lines = []
        for ln in lines:
            low = ln.lower()
            # Skip conversation transcript lines — they quote the user asking about
            # "errors", which is not an app fault.
            if " turn " in low and "->" in ln:
                continue
            if any(
                k in low
                for k in (
                    "traceback",
                    "exception",
                    "segfault",
                    "error:",
                    "failed:",
                    "critical",
                    "fatal",
                )
            ):
                errors.append(ln.strip()[:200])
            elif "warning:" in low or "warn:" in low:
                warnings.append(ln.strip()[:200])
    incidents = OUT_DIR / "incidents.jsonl"
    inc_count = 0
    if incidents.is_file():
        with contextlib.suppress(OSError):
            inc_count = sum(1 for _ in incidents.open(encoding="utf-8"))
    status = "OK" if not errors else "WARN"
    summary: dict[str, Any] = {
        "ok": not errors,
        "status": status,
        "recent_errors": len(errors),
        "recent_warnings": len(warnings),
        "incidents_total": inc_count,
        "top_issues": errors[-5:],
        "checked_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    }
    report = OUT_DIR / "startup_check.md"
    with contextlib.suppress(OSError):
        body = [
            "# Startup check - " + summary["checked_at"],
            "",
            f"Status: **{status}**",
            f"Recent errors (last 400 log lines): {len(errors)}",
            f"Recent warnings: {len(warnings)}",
            f"Incident log entries (total): {inc_count}",
            "",
            "## Most recent issues",
        ]
        body += ["- " + e for e in errors[-10:]] or ["- none"]
        report.write_text("\n".join(body) + "\n", encoding="utf-8")
    summary["report"] = str(report)
    return summary


# Scope names the owner can pick in the Doctor tab.
SCOPES = ("all", "engine", "voice", "models", "tasks", "memory", "turn")
# Who is allowed to narrate a diagnosis (the picker).
AGENTS = ("doctor", "verifier", "front", "router", "researcher", "harness")


@dataclass
class Finding:
    """One detected problem, with the repair that can address it (if any)."""

    check: str
    severity: str = "warning"  # info | warning | error | fatal
    title: str = ""
    detail: str = ""
    evidence: str = ""
    repair: str = ""  # catalogue id, "" when nothing safe applies
    scope: str = "all"

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class RepairResult:
    """What happened when a catalogue entry was applied (or dry-run)."""

    id: str
    title: str
    applied: bool = False
    ok: bool = False
    detail: str = ""
    verified: bool = False
    rolled_back: bool = False
    evidence: list[str] = field(default_factory=list)

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


class Doctor:
    """Runs detectors, applies catalogue repairs, and asks the verifier to agree."""

    def __init__(self, session: Any) -> None:
        self.session = session
        self.state = "idle"  # idle | diagnosing | repairing | verifying
        self.last_run = 0.0
        self.last_agent = "doctor"
        self.last_findings: list[dict[str, Any]] = []
        self.last_results: list[dict[str, Any]] = []
        self._lock = threading.RLock()
        self._repair_counts: dict[str, int] = {}
        self._failures: dict[str, int] = {}
        self._last_repair_at = 0.0
        self._disabled: set[str] = set()

    # ------------------------------------------------------------------ knobs
    @property
    def enabled(self) -> bool:
        return bool(getattr(self.session.settings, "doctor_enabled", True))

    @property
    def kill_switch(self) -> bool:
        return bool(getattr(self.session.settings, "doctor_kill_switch", False))

    def _budget_ok(self, repair_id: str) -> tuple[bool, str]:
        """S3.4: max 3 per repair, max 1 per 30 s, disable after two failures."""
        if self.kill_switch:
            return False, "doctor kill switch is on"
        if repair_id in self._disabled:
            return False, f"{repair_id} failed twice this session; disabled"
        if self._repair_counts.get(repair_id, 0) >= 3:
            return False, f"{repair_id} already applied 3 times; escalating instead"
        elapsed = time.time() - self._last_repair_at
        if self._last_repair_at > 0 and elapsed < 30.0:
            return False, f"one repair per 30 s; {30.0 - elapsed:.0f}s to wait"
        return True, ""

    # --------------------------------------------------------------- diagnose
    def diagnose(self, scope: str = "all", agent: str = "doctor") -> dict[str, Any]:
        """Run the detectors for a scope and return ranked findings (no changes)."""
        scope = scope if scope in SCOPES else "all"
        agent = agent if agent in AGENTS else "doctor"
        with self._lock:
            self.state = "diagnosing"
            self.last_agent = agent
        started = time.time()
        findings: list[Finding] = []
        for check in CHECKS:
            if scope not in {"all"} and check.scope not in {scope, "all"}:
                continue
            found = self._run_check(check)
            if found is not None:
                findings.append(found)
        order = {"fatal": 0, "error": 1, "warning": 2, "info": 3}
        findings.sort(key=lambda item: order.get(item.severity, 9))
        elapsed = round((time.time() - started) * 1000.0, 1)
        with self._lock:
            self.state = "idle"
            self.last_run = time.time()
            self.last_findings = [item.as_dict() for item in findings]
        if findings:
            self.session.incidents.record(
                "doctor",
                f"diagnosis ({scope}) found {len(findings)} issue(s): "
                + "; ".join(item.title for item in findings[:4]),
                severity="warning",
                where=f"doctor/{agent}",
                context={"scope": scope, "agent": agent, "ms": elapsed},
            )
        return {
            "ok": True,
            "scope": scope,
            "agent": agent,
            "ms": elapsed,
            "findings": [item.as_dict() for item in findings],
            "repaired": list(self.last_results[-3:]),
        }

    def _run_check(self, check: Check) -> Finding | None:
        """Run one detector; a crash is logged, recorded, and surfaced as a finding
        so one broken detector never hides the other nine."""
        try:
            return check.run(self)
        except Exception as exc:  # noqa: BLE001 - detectors are independent
            print(f"[doctor] check {check.id} crashed: {exc}", file=sys.stderr)
            self.session.incidents.record(
                "doctor",
                f"detector {check.id} crashed: {type(exc).__name__}: {exc}",
                severity="warning",
                where=f"doctor/{check.id}",
            )
            return Finding(
                check=check.id,
                severity="warning",
                title=f"check {check.id} crashed",
                detail=f"{type(exc).__name__}: {exc}"[:300],
                scope=check.scope,
            )

    # ----------------------------------------------------------------- repair
    def repair(self, repair_id: str, dry_run: bool = True) -> dict[str, Any]:
        """Apply one catalogue entry with budget, verify and rollback."""
        entry = CATALOGUE.get(repair_id)
        if entry is None:
            return {"ok": False, "error": f"no such repair: {repair_id}"}
        result = RepairResult(id=entry.id, title=entry.title)
        if dry_run:
            result.detail = f"dry run: would {entry.action}"
            result.ok = True
            return {"ok": True, "dry_run": True, "result": result.as_dict()}
        allowed, why = self._budget_ok(repair_id)
        if not allowed:
            return {"ok": False, "error": why, "result": result.as_dict()}
        with self._lock:
            self.state = "repairing"
        try:
            backup = entry.fix(self, None)
            result.applied = True
            result.detail = entry.action
            self._repair_counts[repair_id] = self._repair_counts.get(repair_id, 0) + 1
            self._last_repair_at = time.time()
        except Exception as exc:  # noqa: BLE001 - a failed repair is data, not a crash
            result.detail = f"repair failed: {type(exc).__name__}: {exc}"[:300]
            self._count_failure(repair_id)
            self.session.incidents.record(
                "doctor", result.detail, severity="error", where=f"repair/{repair_id}"
            )
            with self._lock:
                self.state = "idle"
            return {"ok": False, "error": result.detail, "result": result.as_dict()}
        with self._lock:
            self.state = "verifying"
        verdict = self.verify()
        result.verified = bool(verdict.get("ok"))
        result.evidence = list(verdict.get("evidence") or [])
        if not result.verified:
            # The repair made the app no better: put it back.
            try:
                result.rolled_back = bool(entry.rollback(self, backup))
            except Exception as exc:  # noqa: BLE001
                result.evidence.append(f"rollback failed: {exc}"[:200])
            result.detail += " | verification failed, rolled back"
            self._count_failure(repair_id)
        else:
            self._failures[repair_id] = 0
        with self._lock:
            self.state = "idle"
            self.last_results.append(result.as_dict())
        self.session.incidents.record(
            "repair",
            f"{entry.id} {entry.title}: {'verified' if result.verified else 'ROLLED BACK'} - {result.detail}",
            severity="info" if result.verified else "error",
            where="doctor",
            context={"repair": entry.id, "verified": result.verified},
        )
        return {"ok": result.verified, "result": result.as_dict()}

    def _count_failure(self, repair_id: str) -> None:
        self._failures[repair_id] = self._failures.get(repair_id, 0) + 1
        if self._failures[repair_id] >= 2:
            self._disabled.add(repair_id)

    # ----------------------------------------------------------------- verify
    def verify(self) -> dict[str, Any]:
        """The verifier: independent checks after a repair. Returns a verdict."""
        evidence: list[str] = []
        ok = True
        for name, probe, expect in probe_sequence():
            try:
                value = probe(self)
            except Exception as exc:  # noqa: BLE001
                ok = False
                evidence.append(f"{name}: FAILED ({type(exc).__name__}: {exc})"[:220])
                continue
            passed = bool(expect(value))
            ok = ok and passed
            evidence.append(f"{name}: {'ok' if passed else 'FAIL'} ({value})"[:220])
        if not ok:
            self.session.incidents.record(
                "verifier",
                "verification failed: "
                + "; ".join(item for item in evidence if "FAIL" in item)[:300],
                severity="error",
                where="verifier",
            )
        return {"ok": ok, "evidence": evidence, "ts": round(time.time(), 3)}

    # ---------------------------------------------------------------- watch
    def watch_tick(self) -> dict[str, Any] | None:
        """S4.1: demo mode - detect only, and only when something changed.

        The owner asked for an app that can notice trouble during a demo. Watching
        must stay cheap and must never repair on its own: one diagnose pass, and a
        banner (an incident) only when the finding set changed since last time.
        """
        if self.kill_switch or not self.enabled:
            return None
        if self.state != "idle":
            return None  # never pile a diagnosis on top of a repair or a turn
        if time.time() - self.last_run < 20.0:
            return None
        result = self.diagnose("all", self.last_agent)
        findings = list(result.get("findings") or [])
        signature = tuple(sorted(str(item.get("check")) for item in findings))
        if signature == getattr(self, "_watch_signature", None):
            return None
        self._watch_signature = signature
        if findings:
            self.session.incidents.record(
                "doctor/watch",
                "demo watch: "
                + "; ".join(str(item.get("title")) for item in findings[:3]),
                severity="warning",
                where="doctor/watch",
                context={"checks": list(signature)},
            )
        return result

    def fix_top_finding(self, dry_run: bool = False) -> dict[str, Any]:
        """S4.1: one-click "Fix now" - repair the most severe finding with a repair.

        The choice is the same catalogue the pane lists; this only spares the owner
        from matching title to id under pressure. No repair means a clear refusal.
        """
        findings = list(self.last_findings)
        if not findings:
            result = self.diagnose("all", self.last_agent)
            findings = list(result.get("findings") or [])
        for item in findings:
            repair_id = str(item.get("repair") or "")
            if repair_id:
                return self.repair(repair_id, dry_run)
        return {"ok": False, "error": "nothing in the last diagnosis has a safe repair"}

    # ------------------------------------------------------------- reporters
    def status(self) -> dict[str, Any]:
        return {
            "enabled": self.enabled,
            "kill_switch": self.kill_switch,
            "watch": bool(getattr(self.session.settings, "doctor_watch", False)),
            "state": self.state,
            "last_run": self.last_run,
            "last_agent": self.last_agent,
            "findings": self.last_findings[-8:],
            "results": self.last_results[-5:],
            "budget": {
                "applied": dict(self._repair_counts),
                "disabled": sorted(self._disabled),
            },
        }


# ------------------------------------------------------------------- escalate
def escalate(session: Any, question: str, evidence: str = "") -> dict[str, Any]:
    """S3.5: a code-level problem is PROPOSED here, never applied here.

    The proposal becomes an approval row (the existing gate) plus an incident, so
    the boundary is demonstrable: the doctor detects and proposes; a person or the
    harness - with an explicit approval - changes code. Nothing is rewritten on
    its own, which is the rule that keeps a demo alive.
    """
    text = " ".join(str(question).split())[:400]
    if not text:
        return {"ok": False, "error": "nothing to escalate"}
    doctor = getattr(session, "_doctor", None)
    if doctor is not None and doctor.kill_switch:
        return {"ok": False, "error": "doctor kill switch is on"}
    try:
        approval_id = session.request_approval(f"harness: {text}", str(evidence)[:400])
    except Exception as exc:  # noqa: BLE001 - no gate means no escalation, not a crash
        return {"ok": False, "error": f"could not queue the approval: {exc}"[:200]}
    incident = session.incidents.record(
        "escalation",
        f"proposed to the harness (approval #{approval_id}): {text}",
        severity="warning",
        where="doctor/escalate",
        context={"approval": approval_id, "evidence": str(evidence)[:400]},
    )
    return {"ok": True, "approval": approval_id, "incident": incident["id"]}


# --------------------------------------------------------------------- checks
@dataclass
class Check:
    """A detector: returns a Finding when something is wrong, else None."""

    id: str
    title: str
    scope: str
    run: Callable[[Doctor], Finding | None]


@dataclass
class Repair:
    """A catalogue entry: fix returns a backup for rollback; verify is separate."""

    id: str
    title: str
    action: str
    fix: Callable[[Doctor, Any], Any]
    rollback: Callable[[Doctor, Any], bool]


def _argv_of(proc: Any) -> list[str]:
    """A process command line as a list, or [] when the OS denies access."""
    try:
        return [str(a) for a in (proc.info.get("cmdline") or [])]
    except Exception:  # noqa: BLE001 - access denied is not an engine
        return []


def _script_of(argv: list[str]) -> str:
    """The .py file this process actually runs, or "" for -c / -m / shell wrappers.

    This is the difference between an engine and a terminal that mentioned one:
    a shell whose *command text* contains "server.py" is NOT an engine, and R1
    must never terminate it. The first version of this matcher counted such
    shells, so a demo-time repair could have killed an unrelated process.
    """
    if len(argv) < 2:
        return ""
    if str(argv[1]) in {"-c", "-m", "-"}:
        return ""  # the program is the text itself, not a file
    for arg in argv[1:]:
        text = str(arg)
        if text.endswith(".py"):
            return Path(text).name
    return ""


def _engine_processes() -> list[dict[str, Any]]:
    """Every python process running this engine, via psutil (never via a shell)."""
    try:
        import psutil
    except ImportError:
        return []
    rows: list[dict[str, Any]] = []
    for proc in psutil.process_iter(["pid", "name", "cmdline", "cwd"]):
        argv = _argv_of(proc)
        script = _script_of(argv)
        if script not in {"server.py", "app_gui.py"}:
            continue
        cmdline = " ".join(argv)
        try:
            proc_cwd = str(proc.info.get("cwd") or "")
        except Exception:  # noqa: BLE001
            proc_cwd = ""
        if (
            BASE.name not in cmdline
            and "me3_laptop" not in cmdline
            and BASE.name not in proc_cwd
            and "me3_laptop" not in proc_cwd
        ):
            continue
        rows.append({"pid": proc.info["pid"], "cmdline": cmdline[:200]})
    return rows


def _cmdline_of(proc: Any) -> str:
    """A process command line, or "" when the OS denies access (system processes).

    Returning "" is deliberate: an unreadable process is simply not an engine, and
    this must never raise inside a detector.
    """
    try:
        return " ".join(proc.info.get("cmdline") or [])
    except Exception:  # noqa: BLE001 - access denied is not an engine
        return ""


def _health_ok(port: int = 8766, host: str = "127.0.0.1") -> bool:
    """Does something answer as an engine on this port?"""
    import urllib.request

    try:
        with urllib.request.urlopen(
            f"http://{host}:{port}/api/health", timeout=2.0
        ) as reply:
            payload = json.loads(reply.read().decode("utf-8"))
        return bool(payload.get("ok"))
    except Exception:  # noqa: BLE001 - no answer is the answer
        return False


def _identity_state() -> bool:
    """Does the pid file agree with who owns the voice right now?"""
    try:
        pid = int(ENGINE_PID.read_text(encoding="utf-8").strip() or "0")
    except (OSError, ValueError):
        pid = 0
    if pid == os.getpid():
        return True
    return bool(pid and engine_owner() and _health_ok())


def _port_bindable(port: int = 8766, host: str = "127.0.0.1") -> bool:
    probe = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    try:
        probe.bind((host, port))
        return True
    except OSError:
        return False
    finally:
        probe.close()


# Public names for the engine guard in server.py. The detectors above are private
# because they are implementation detail; these two are the API the engine needs.
engine_health = _health_ok
engine_processes = _engine_processes


def engine_owner() -> int:
    """The pid that currently owns the engine lock, or 0 when nobody does.

    The compact window calls this before spawning an engine: a live owner means
    the voice is already taken, and starting a second engine would give the demo
    two voices fighting for the speakers.
    """
    try:
        pid = int(ENGINE_PID.read_text(encoding="utf-8").strip() or "0")
    except (OSError, ValueError):
        return 0
    if pid <= 0:
        return 0
    try:
        import psutil

        return pid if psutil.pid_exists(pid) else 0
    except Exception:  # noqa: BLE001 - without psutil, os.kill(0) still answers
        try:
            os.kill(pid, 0)
            return pid
        except OSError:
            return 0


def check_duplicate_engines(doc: Doctor) -> Finding | None:
    """R1: more than one engine process = two voices, two session folders."""
    procs = _engine_processes()
    if len(procs) <= 1:
        return None
    pids = ", ".join(str(item["pid"]) for item in procs)
    return Finding(
        check="R1",
        severity="error",
        title=f"{len(procs)} engine processes are running (one engine, one voice is the rule)",
        detail=(
            "Multiple engines compete for the same port and speakers: requests can land "
            "on either, sessions split, and two voices can speak at once."
        ),
        evidence=f"pids: {pids}; answering: {_health_ok()}",
        repair="R1",
        scope="engine",
    )


def check_engine_pid(doc: Doctor) -> Finding | None:
    """R2: out/engine.pid must name the engine that is actually serving."""
    if not ENGINE_PID.exists():
        return Finding(
            check="R2",
            severity="warning",
            title="engine.pid is missing",
            detail="The CLI and the app use it to stop the right process.",
            repair="R2",
            scope="engine",
        )
    try:
        recorded = int(ENGINE_PID.read_text(encoding="utf-8").strip())
    except (OSError, ValueError):
        return Finding(
            check="R2",
            severity="warning",
            title="engine.pid is unreadable",
            detail="Whatever it holds cannot be a pid.",
            repair="R2",
            scope="engine",
        )
    if recorded != os.getpid():
        return Finding(
            check="R2",
            severity="warning",
            title=f"engine.pid names pid {recorded}, but this engine is {os.getpid()}",
            detail="A stale pid file makes stop/restart act on the wrong process.",
            evidence=f"file={recorded} self={os.getpid()}",
            repair="R2",
            scope="engine",
        )
    return None


def check_port(doc: Doctor) -> Finding | None:
    """R3: the port must be ours; a foreign holder is reported, never killed."""
    port = int(getattr(doc.session, "_listen_port", 8766) or 8766)
    if not _port_bindable(port=port) and not _health_ok(port=port):
        return Finding(
            check="R3",
            severity="error",
            title=f"port {port} is held by something that is not this app",
            detail="Another program is using the port. Nothing here kills a foreign process.",
            repair="",
            scope="engine",
        )
    return None


def check_config(doc: Doctor) -> Finding | None:
    """R8: config.json must parse; a broken file is restored from the last-good copy."""
    if not CONFIG_PATH.exists():
        return None
    try:
        json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        backup = CONFIG_PATH.with_suffix(".json.bak")
        return Finding(
            check="R8",
            severity="error",
            title="config.json is unreadable",
            detail=f"{type(exc).__name__}: {exc}"[:240],
            evidence=f"last-good backup present: {backup.exists()}",
            repair="R8" if backup.exists() else "",
            scope="engine",
        )
    return None


def check_voice(doc: Doctor) -> Finding | None:
    """R5: the voice engine must construct and name itself."""
    try:
        name = str(doc.session.tts().voice_name)
        if not name:
            raise RuntimeError("voice has no name")
        return None
    except Exception as exc:  # noqa: BLE001 - any failure here is a finding
        return Finding(
            check="R5",
            severity="error",
            title="the voice engine is not usable",
            detail=f"{type(exc).__name__}: {exc}"[:240],
            repair="R5",
            scope="voice",
        )


def check_asr(doc: Doctor) -> Finding | None:
    """R6: the recogniser must load."""
    try:
        doc.session.asr()
        return None
    except Exception as exc:  # noqa: BLE001
        return Finding(
            check="R6",
            severity="error",
            title="the speech recogniser cannot load",
            detail=f"{type(exc).__name__}: {exc}"[:240],
            repair="R6",
            scope="voice",
        )


def check_models(doc: Doctor) -> Finding | None:
    """R4 (detect only): a provider with no key, or an empty model id."""
    settings = doc.session.settings
    if (
        settings.provider == "openrouter"
        and not settings.openrouter_key
        and not bool(getattr(settings, "front_uses_pi", True))
    ):
        return Finding(
            check="R4",
            severity="error",
            title="OpenRouter is selected but no key is set",
            detail="Every turn will fail until a key is pasted in Controls.",
            repair="",
            scope="models",
        )
    if not str(settings.model).strip():
        return Finding(
            check="R4",
            severity="error",
            title="no model is selected",
            detail="Pick one in Controls, or the next turn has nothing to call.",
            repair="",
            scope="models",
        )
    return None


def check_tasks(doc: Doctor) -> Finding | None:
    """R10: a task running far past its expected life is wedged."""
    try:
        tasks = doc.session.tasks_manager().list()
    except Exception:  # noqa: BLE001
        return None
    stuck = [
        task
        for task in tasks
        if task.state in {"running", "queued"} and task.seconds > 600
    ]
    if not stuck:
        return None
    return Finding(
        check="R10",
        severity="warning",
        title=f"{len(stuck)} task(s) running over 10 minutes",
        detail="A wedged task holds a model and a queue slot.",
        evidence=", ".join(f"{task.id} ({int(task.seconds)}s)" for task in stuck[:4]),
        repair="R10",
        scope="tasks",
    )


def check_memory(doc: Doctor) -> Finding | None:
    """R11: the app plus its models must stay inside the machine's budget."""
    try:
        import server as engine

        used = float(engine.process_memory_mb() or 0.0)
    except Exception:  # noqa: BLE001
        return None
    if used > 4000.0:
        return Finding(
            check="R11",
            severity="warning",
            title=f"the app is using {used:.0f} MB",
            detail="Freeing idle models keeps the demo alive on a smaller machine.",
            evidence=f"process_memory_mb={used:.0f}",
            repair="R11",
            scope="memory",
        )
    return None


def check_speech_length(doc: Doctor) -> Finding | None:
    """R12: a reply that takes a minute to say reads as a hang."""
    turns = list(getattr(doc.session, "turns", []) or [])
    if not turns:
        return None
    last = turns[-1]
    tts_ms = float((last.get("timings") or {}).get("tts_ms") or 0.0)
    chars = len(str(last.get("assistant_text") or ""))
    if tts_ms > 60000 or chars > 1200:
        return Finding(
            check="R12",
            severity="warning",
            title="the last reply took over a minute to speak",
            detail="Long spoken replies feel like a freeze; cap speech, keep the rest on screen.",
            evidence=f"tts_ms={tts_ms:.0f} chars={chars}",
            repair="R12",
            scope="turn",
        )
    return None


def check_mic(doc: Doctor) -> Finding | None:
    """R7: a capture device must exist and actually open.

    A microphone held by another app (Teams, Zoom) is the classic demo wound: the
    app looks healthy and every turn silently hears nothing.
    """
    import audio as audio_module

    try:
        devices = audio_module.input_devices()
    except Exception as exc:  # noqa: BLE001 - no device list is itself a finding
        return Finding(
            check="R7",
            severity="error",
            title="the audio device list cannot be read",
            detail=f"{type(exc).__name__}: {exc}"[:240],
            repair="R7",
            scope="voice",
        )
    if not devices:
        return Finding(
            check="R7",
            severity="error",
            title="no microphone is visible",
            detail="Plug one in, then repair; the app can still be typed into.",
            evidence="capture devices: 0",
            repair="R7",
            scope="voice",
        )
    current = doc.session.input_device
    if current is not None and audio_module.can_open_input(current):
        return None
    usable = next(
        (d.index for d in devices if audio_module.can_open_input(d.index)), None
    )
    if usable is None:
        return Finding(
            check="R7",
            severity="error",
            title="every microphone is busy",
            detail="Another program is holding the capture device. Close it and repair.",
            evidence=f"selected={current} none of {len(devices)} device(s) opens",
            repair="",
            scope="voice",
        )
    return Finding(
        check="R7",
        severity="warning",
        title="the selected microphone cannot be opened",
        detail="Switching to a device that opens keeps the demo listening.",
        evidence=f"selected={current} openable={usable}",
        repair="R7",
        scope="voice",
    )


def check_session_dir(doc: Doctor) -> Finding | None:
    """R9: the session folder must exist and accept a write.

    A full disk or a deleted folder turns every turn into a silent log failure.
    """
    folder = getattr(doc.session, "session_dir", None)
    if folder is None:
        return None  # a build without session logging is not broken
    path = Path(folder)
    if not path.is_dir():
        return Finding(
            check="R9",
            severity="warning",
            title="the session folder is missing",
            detail="Turns cannot be written to disk, so the log is lost.",
            evidence=str(path),
            repair="R9",
            scope="engine",
        )
    probe = path / ".write-probe"
    try:
        probe.write_text("ok", encoding="utf-8")
        probe.unlink()
    except OSError as exc:
        return Finding(
            check="R9",
            severity="error",
            title="the session folder is not writable",
            detail=f"{type(exc).__name__}: {exc}"[:240],
            evidence=str(path),
            repair="R9",
            scope="engine",
        )
    return None


CHECKS: tuple[Check, ...] = (
    Check("R1", "one engine only", "engine", check_duplicate_engines),
    Check("R2", "engine pid file", "engine", check_engine_pid),
    Check("R3", "port ownership", "engine", check_port),
    Check("R8", "config file", "engine", check_config),
    Check("R5", "voice engine", "voice", check_voice),
    Check("R6", "speech recogniser", "voice", check_asr),
    Check("R7", "microphone", "voice", check_mic),
    Check("R9", "session folder", "engine", check_session_dir),
    Check("R4", "model settings", "models", check_models),
    Check("R10", "wedged tasks", "tasks", check_tasks),
    Check("R11", "memory budget", "memory", check_memory),
    Check("R12", "spoken length", "turn", check_speech_length),
)


# -------------------------------------------------------------------- repairs
def _fix_duplicate_engines(doc: Doctor, _backup: Any = None) -> dict[str, Any]:
    """R1: keep THIS engine, terminate the others."""
    import psutil

    victims = [
        int(item["pid"])
        for item in _engine_processes()
        if int(item["pid"]) != os.getpid()
    ]
    for pid in victims:
        with contextlib.suppress(psutil.Error):
            psutil.Process(pid).terminate()
    deadline = time.time() + 5.0
    for pid in victims:
        with contextlib.suppress(psutil.Error):
            proc = psutil.Process(pid)
            while proc.is_running() and time.time() < deadline:
                time.sleep(0.1)
            if proc.is_running():
                proc.kill()
    return {"killed": victims}


def _rollback_duplicate_engines(doc: Doctor, backup: Any) -> bool:
    # Killed processes cannot be un-killed; the kept engine is already serving.
    return _health_ok()


def _fix_engine_pid(doc: Doctor, _backup: Any = None) -> dict[str, Any]:
    """R2: the pid file must name the engine that owns the voice.

    Refusing when a live owner exists matters: a self-test or a client session
    repairing this would otherwise STEAL the lock and the owner would look like
    the intruder. No live owner means the file is merely stale, which is fixable.
    """
    owner = engine_owner()
    if owner and owner != os.getpid():
        raise RuntimeError(
            f"pid {owner} currently owns the engine; not stealing its pid file"
        )
    previous = ENGINE_PID.read_text(encoding="utf-8") if ENGINE_PID.exists() else ""
    ENGINE_PID.parent.mkdir(parents=True, exist_ok=True)
    ENGINE_PID.write_text(str(os.getpid()), encoding="utf-8")
    return {"previous": previous, "now": os.getpid()}


def _rollback_engine_pid(doc: Doctor, backup: Any) -> bool:
    previous = str((backup or {}).get("previous", "")).strip()
    if previous:
        ENGINE_PID.write_text(previous, encoding="utf-8")
    return True


def _fix_config(doc: Doctor, _backup: Any = None) -> dict[str, Any]:
    backup = CONFIG_PATH.with_suffix(".json.bak")
    aside = CONFIG_PATH.with_suffix(".json.broken")
    if CONFIG_PATH.exists():
        shutil.copyfile(CONFIG_PATH, aside)
    shutil.copyfile(backup, CONFIG_PATH)
    return {"from": str(backup), "aside": str(aside)}


def _rollback_config(doc: Doctor, backup: Any) -> bool:
    aside = Path(str((backup or {}).get("aside", "")))
    if aside.exists():
        shutil.copyfile(aside, CONFIG_PATH)
        return True
    return False


def _fix_voice(doc: Doctor, _backup: Any = None) -> dict[str, Any]:
    """R5: drop the cached engine and rebuild it (neural, then SAPI)."""
    session = doc.session
    previous = str(getattr(session.settings, "tts_voice", ""))
    with session.lock:
        session._tts = None
    try:
        return {"previous_voice": previous, "voice": str(session.tts().voice_name)}
    except Exception:  # noqa: BLE001 - second attempt with SAPI
        session.settings.tts_voice = ""
        with session.lock:
            session._tts = None
        return {"previous_voice": previous, "voice": str(session.tts().voice_name)}


def _rollback_voice(doc: Doctor, backup: Any) -> bool:
    doc.session.settings.tts_voice = str((backup or {}).get("previous_voice", ""))
    with doc.session.lock:
        doc.session._tts = None
    return True


def _fix_mic(doc: Doctor, _backup: Any = None) -> dict[str, Any]:
    """R7: switch to the first capture device that opens right now."""
    import audio as audio_module

    previous = doc.session.input_device
    usable = next(
        (
            d.index
            for d in audio_module.input_devices()
            if audio_module.can_open_input(d.index)
        ),
        None,
    )
    if usable is None:
        raise RuntimeError("no capture device could be opened")
    doc.session.input_device = usable
    return {"previous": previous, "now": usable}


def _rollback_mic(doc: Doctor, backup: Any) -> bool:
    with contextlib.suppress(Exception):
        doc.session.input_device = (backup or {}).get("previous")
    return bool(_can_open_any_input())


def _can_open_any_input() -> bool:
    import audio as audio_module

    with contextlib.suppress(Exception):
        return any(
            audio_module.can_open_input(d.index) for d in audio_module.input_devices()
        )
    return False


def _fix_session_dir(doc: Doctor, _backup: Any = None) -> dict[str, Any]:
    """R9: recreate the folder, keeping the broken one aside for inspection."""
    folder = Path(doc.session.session_dir)
    stamp = time.strftime("%Y%m%d-%H%M%S")
    aside = folder.with_name(folder.name + f".broken-{stamp}")
    moved = False
    if folder.exists():
        with contextlib.suppress(OSError):
            folder.rename(aside)
            moved = True
    try:
        folder.mkdir(parents=True, exist_ok=True)
        (folder / ".write-probe").write_text("ok", encoding="utf-8")
    except OSError as exc:
        # A repair that cannot recreate the folder must fail loudly: the engine
        # records the incident and the verifier never sees a false "fixed".
        raise RuntimeError(f"could not recreate the session folder: {exc}") from exc
    return {"aside": str(aside) if moved else "", "folder": str(folder)}


def _rollback_session_dir(doc: Doctor, backup: Any) -> bool:
    folder = Path(doc.session.session_dir)
    aside = str((backup or {}).get("aside") or "")
    if not aside:
        return True
    try:
        if not Path(aside).exists():
            return False
        if folder.exists():
            shutil.rmtree(folder, ignore_errors=True)
        Path(aside).rename(folder)
    except OSError as exc:
        print(f"[doctor] R9 rollback failed: {exc}", file=sys.stderr)
        return False
    return True


def _fix_asr(doc: Doctor, _backup: Any = None) -> dict[str, Any]:
    """R6: clear the cached recogniser so it reloads."""
    session = doc.session
    with session.lock:
        session._asr = None
        session._vad = None
    session.asr()
    return {"reloaded": True, "backend": session.settings.asr_backend}


def _rollback_asr(doc: Doctor, backup: Any) -> bool:
    with doc.session.lock:
        doc.session._asr = None
    return True


def _fix_tasks(doc: Doctor, _backup: Any = None) -> dict[str, Any]:
    """R10: cancel wedged tasks; their briefs are kept for a manual re-run."""
    manager = doc.session.tasks_manager()
    cancelled: list[str] = []
    for task in manager.list():
        if (
            task.state in {"running", "queued"}
            and task.seconds > 600
            and manager.cancel(task.id)
        ):
            cancelled.append(task.id)
    return {"cancelled": cancelled}


def _rollback_tasks(doc: Doctor, backup: Any) -> bool:
    return True  # cancelled work is reported, never silently re-run


def _fix_memory(doc: Doctor, _backup: Any = None) -> dict[str, Any]:
    return {"freed": doc.session.release_idle_models(force=True)}


def _rollback_memory(doc: Doctor, backup: Any) -> bool:
    return True  # models reload lazily


def _patch_config(key: str, value: Any) -> None:
    """Write ONE settings key into config.json without touching the rest.

    The doctor's session may carry check-mode values (TTS disabled), so a full
    save_config() here would overwrite the owner's real preferences - which is
    exactly how tts_enabled ended up False in the live config after a repair
    ran during a check.
    """
    try:
        data = json.loads(CONFIG_PATH.read_text(encoding="utf-8-sig"))
    except (OSError, ValueError):
        data = {}
    if isinstance(data, dict):
        data[key] = value
        CONFIG_PATH.write_text(json.dumps(data, indent=2), encoding="utf-8")


def _fix_speech_cap(doc: Doctor, _backup: Any = None) -> dict[str, Any]:
    previous = getattr(doc.session.settings, "speech_cap_chars", 0)
    doc.session.settings.speech_cap_chars = 600
    _patch_config("speech_cap_chars", 600)
    return {"previous": previous, "now": 600}


def _rollback_speech_cap(doc: Doctor, backup: Any) -> bool:
    try:
        previous = int((backup or {}).get("previous", 0) or 0)
    except (TypeError, ValueError):
        previous = 0
    doc.session.settings.speech_cap_chars = previous
    _patch_config("speech_cap_chars", previous)
    return True


CATALOGUE: dict[str, Repair] = {
    "R1": Repair(
        "R1",
        "keep exactly one engine",
        "terminate the other engine processes",
        _fix_duplicate_engines,
        _rollback_duplicate_engines,
    ),
    "R2": Repair(
        "R2",
        "fix the engine pid file",
        "rewrite engine.pid with this engine's pid",
        _fix_engine_pid,
        _rollback_engine_pid,
    ),
    "R5": Repair(
        "R5",
        "rebuild the voice engine",
        "drop the cached voice and rebuild it",
        _fix_voice,
        _rollback_voice,
    ),
    "R6": Repair(
        "R6",
        "reload the recogniser",
        "clear and reload the ASR engine",
        _fix_asr,
        _rollback_asr,
    ),
    "R7": Repair(
        "R7",
        "switch to a working microphone",
        "select the first capture device that opens",
        _fix_mic,
        _rollback_mic,
    ),
    "R9": Repair(
        "R9",
        "rebuild the session folder",
        "recreate the folder, keep the broken one aside",
        _fix_session_dir,
        _rollback_session_dir,
    ),
    "R8": Repair(
        "R8",
        "restore the last-good config",
        "copy config.json.bak over config.json",
        _fix_config,
        _rollback_config,
    ),
    "R10": Repair(
        "R10",
        "cancel wedged tasks",
        "cancel tasks running over 10 minutes",
        _fix_tasks,
        _rollback_tasks,
    ),
    "R11": Repair(
        "R11",
        "free idle models",
        "release ASR/VAD/Ollama memory now",
        _fix_memory,
        _rollback_memory,
    ),
    "R12": Repair(
        "R12",
        "cap spoken replies",
        "speak at most 600 characters, rest on screen",
        _fix_speech_cap,
        _rollback_speech_cap,
    ),
}


def _config_body() -> Any:
    """The config as data, or None when it cannot be read (the verifier then fails)."""
    try:
        if not CONFIG_PATH.exists():
            return {}
        return json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        print(f"[doctor] config unreadable: {exc}", file=sys.stderr)
        return None


def probe_sequence() -> list[
    tuple[str, Callable[[Doctor], Any], Callable[[Any], bool]]
]:
    """The verifier's checks: name, probe, expectation."""
    return [
        ("engine answers", lambda doc: _health_ok(), bool),
        (
            "one engine process",
            lambda doc: len(_engine_processes()),
            lambda n: int(n) <= 1,
        ),
        # The property is coherence, not self-identity: after a repair the pid
        # file must point at this engine OR at a live engine that answers. The
        # stricter "== os.getpid()" made every repair roll back whenever a second
        # face (or a probe) ran the doctor without owning the voice.
        ("engine identity is coherent", lambda doc: _identity_state(), bool),
        (
            "config parses",
            lambda doc: _config_body(),
            lambda body: isinstance(body, dict),
        ),
        ("voice answers", lambda doc: str(doc.session.tts().voice_name), bool),
        ("recogniser loads", lambda doc: bool(doc.session.asr()), bool),
    ]
