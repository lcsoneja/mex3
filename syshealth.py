"""System health, per-feature resource distribution, introspection & Doctor gate.

Provides:
- system_health(): cached CPU %, RAM used/total, network KB/s + link, disk, temp (Pi + Win)
- feature_distribution(session): RSS MB + CPU ms/pct across ASR, TTS, Voiceprint, Pi panes
- performance_review(session): bottleneck analysis & recommendations from live session turns
- app_introspection(session): read-only system prompts, context file, skills, tool schemas
- is_asking(text, pending_clarify): lightweight Ask-Detector dropping filler backchannels
- DoctorGate: requires exact phrase "Yes, I confirm the changes" before applying settings/restart
"""

from __future__ import annotations

import contextlib
import dataclasses
import json
import os
import re
import socket
import threading
import time
from pathlib import Path
from typing import Any

import psutil

BASE = Path(__file__).resolve().parent
CONFIRM_PHRASE = "Yes, I confirm the changes"

_HEALTH_CACHE: dict[str, Any] = {"at": 0.0, "data": {}}
_NET_PREV: dict[str, Any] = {"at": 0.0, "bytes": 0}
_PROC_CACHE = psutil.Process(os.getpid())

# Prime non-blocking cpu_percent once at import
with contextlib.suppress(Exception):
    psutil.cpu_percent(interval=None)
    _PROC_CACHE.cpu_percent(interval=None)


def _read_temp_c() -> float | None:
    """SoC / CPU temperature in Celsius: reliable on Raspberry Pi, optional on Windows."""
    pi_thermal = Path("/sys/class/thermal/thermal_zone0/temp")
    if pi_thermal.is_file():
        with contextlib.suppress(OSError, ValueError):
            return round(int(pi_thermal.read_text().strip()) / 1000.0, 1)
    with contextlib.suppress(Exception):
        fn = getattr(psutil, "sensors_temperatures", None)
        temps = fn() if callable(fn) else None
        if isinstance(temps, dict):
            for entries in temps.values():
                if entries:
                    return round(float(entries[0].current), 1)
    return None


def _net_online_fast() -> bool:
    """Fast link check (checks active non-loopback NIC with an IPv4 address)."""
    with contextlib.suppress(Exception):
        stats = psutil.net_if_stats()
        addrs = psutil.net_if_addrs()
        for nic, st in stats.items():
            if not st.isup or "loopback" in nic.lower():
                continue
            for a in addrs.get(nic, []):
                if a.family == socket.AF_INET and not a.address.startswith("127."):
                    return True
    return False


def system_health() -> dict[str, Any]:
    """Cached (3 s) computer health for the status bar (Windows + Raspberry Pi)."""
    now = time.monotonic()
    if (now - float(_HEALTH_CACHE["at"])) < 3.0 and _HEALTH_CACHE["data"]:
        return dict(_HEALTH_CACHE["data"])

    vm = psutil.virtual_memory()
    disk = psutil.disk_usage(str(BASE))
    cpu = round(float(psutil.cpu_percent(interval=None)), 1)

    net = psutil.net_io_counters()
    total_bytes = int(net.bytes_recv + net.bytes_sent) if net else 0
    dt = now - float(_NET_PREV["at"])
    kbps = 0.0
    if _NET_PREV["at"] > 0 and dt > 0.2:
        kbps = round(max(0, total_bytes - int(_NET_PREV["bytes"])) / 1024.0 / dt, 1)
    _NET_PREV["at"] = now
    _NET_PREV["bytes"] = total_bytes

    data = {
        "cpu_pct": cpu,
        "ram_used_gb": round((vm.total - vm.available) / (1024**3), 1),
        "ram_total_gb": round(vm.total / (1024**3), 1),
        "ram_pct": round(float(vm.percent), 1),
        "disk_free_gb": round(disk.free / (1024**3), 1),
        "net_up": _net_online_fast(),
        "net_kbps": kbps,
        "temp_c": _read_temp_c(),
    }
    _HEALTH_CACHE["at"] = now
    _HEALTH_CACHE["data"] = data
    return dict(data)


def feature_distribution(session: Any) -> dict[str, Any]:
    """Per-feature RAM, CPU, and latency-block distribution for Performance view."""
    total_rss_mb = round(_PROC_CACHE.memory_info().rss / (1024 * 1024), 1)
    main_cpu = round(float(_PROC_CACHE.cpu_percent(interval=None)), 1)

    # Measure Pi RPC child processes (node.exe / pi)
    pi_rss_mb = 0.0
    pi_cpu_pct = 0.0
    pane_procs: list[dict[str, Any]] = []
    with contextlib.suppress(Exception):
        for name, pane in list(getattr(session, "_pi_panes", {}).items()):
            pid = getattr(
                getattr(getattr(pane, "rpc", None), "proc", None), "pid", None
            )
            if pid:
                try:
                    p = psutil.Process(int(pid))
                    rss = round(p.memory_info().rss / (1024 * 1024), 1)
                    cpu = round(float(p.cpu_percent(interval=None)), 1)
                    pi_rss_mb += rss
                    pi_cpu_pct += cpu
                    pane_procs.append(
                        {"role": name, "pid": pid, "mem_mb": rss, "cpu_pct": cpu}
                    )
                except (psutil.NoSuchProcess, psutil.AccessDenied):
                    pass

    # Approximate in-process feature memory footprints (loaded vs unloaded)
    asr_loaded = getattr(session, "_asr", None) is not None
    tts_loaded = getattr(session, "_tts", None) is not None
    vad_loaded = getattr(session, "_vad", None) is not None
    vp_dash = {}
    if hasattr(session, "_voiceprint"):
        with contextlib.suppress(Exception):
            vp_dash = session._voiceprint.dashboard()

    asr_mb = (
        620.0
        if (asr_loaded and "parakeet" in str(getattr(session, "_asr_backend", "")))
        else (240.0 if asr_loaded else 0.0)
    )
    tts_mb = 180.0 if tts_loaded else 0.0
    vp_mb = (
        float((vp_dash.get("telemetry") or {}).get("mem_mb") or 0.0)
        if (vp_dash.get("model_present"))
        else 0.0
    )
    vad_mb = 8.0 if vad_loaded else 0.0
    core_mb = max(40.0, round(total_rss_mb - (asr_mb + tts_mb + vp_mb + vad_mb), 1))

    # Latency blocks from recent turns
    turns = [
        t for t in getattr(session, "turns", []) if t.get("source") in {"mic", "text"}
    ][-30:]
    blocks: list[dict[str, Any]] = []
    for t in turns[-12:]:
        tm = t.get("timings") or {}
        asr = float(tm.get("asr_ms") or 0.0)
        ttft = float(tm.get("first_token_ms") or 0.0)
        llm_total = float(tm.get("llm_ms") or 0.0)
        llm_gen = max(0.0, llm_total - ttft) if ttft > 0 else llm_total
        tts = float(tm.get("tts_ms") or 0.0)
        total = max(1.0, asr + ttft + llm_gen + tts)
        blocks.append(
            {
                "id": t.get("id"),
                "ts": t.get("ts", ""),
                "source": t.get("source", ""),
                "user": str(t.get("user_text", ""))[:40],
                "asr_ms": round(asr, 1),
                "ttft_ms": round(ttft, 1),
                "llm_gen_ms": round(llm_gen, 1),
                "tts_ms": round(tts, 1),
                "total_ms": round(total, 1),
                "bottleneck": max(
                    [
                        ("asr", asr),
                        ("llm_ttft", ttft),
                        ("llm_gen", llm_gen),
                        ("tts", tts),
                    ],
                    key=lambda x: x[1],
                )[0],
            }
        )

    # Average CPU time share per feature from recent turn timings
    n_t = max(1, len(turns))
    avg_asr = (
        sum(float((t.get("timings") or {}).get("asr_ms") or 0) for t in turns) / n_t
    )
    avg_llm = (
        sum(float((t.get("timings") or {}).get("llm_ms") or 0) for t in turns) / n_t
    )
    avg_tts = (
        sum(float((t.get("timings") or {}).get("tts_ms") or 0) for t in turns) / n_t
    )
    avg_vp = float((vp_dash.get("telemetry") or {}).get("avg_embed_ms") or 0.0)

    clf_stats = {}
    if hasattr(session, "classifier"):
        with contextlib.suppress(Exception):
            clf_stats = dict(session.classifier().stats)
    aec_obj = getattr(session, "_echo", None)
    ns_obj = getattr(session, "_ns", None)

    features = [
        {
            "name": f"Decision Classifier ({clf_stats.get('stage', 'rules-v1')})",
            "mem_mb": 0.4,
            "avg_ms": round(
                float(
                    clf_stats.get("avg_latency_ms")
                    or clf_stats.get("last_latency_ms")
                    or 0.0
                ),
                3,
            ),
            "kind": "non-LLM <100ms",
        },
        {
            "name": f"TTS Echo Canceller ({getattr(aec_obj, 'name', 'nlms-aec')})",
            "mem_mb": 0.8,
            "avg_ms": 0.3
            if getattr(session.settings, "echo_cancellation", True)
            else 0.0,
            "kind": "local DSP"
            if getattr(session.settings, "echo_cancellation", True)
            else "off",
        },
        {
            "name": f"Telecon Noise Suppressor ({getattr(ns_obj, 'name', 'wiener-ns')})",
            "mem_mb": 0.3,
            "avg_ms": 0.25
            if getattr(session.settings, "noise_suppression", True)
            else 0.0,
            "kind": "local DSP"
            if getattr(session.settings, "noise_suppression", True)
            else "off",
        },
        {
            "name": "ASR ("
            + (
                getattr(session.settings, "asr_cloud", "")
                or getattr(session.settings, "asr_backend", "local")
            )
            + ")",
            "mem_mb": round(asr_mb, 1),
            "avg_ms": round(avg_asr, 1),
            "kind": "local"
            if not getattr(session.settings, "asr_cloud", "")
            else "cloud",
        },
        {
            "name": "TTS ("
            + (getattr(session.settings, "tts_cloud", "") or "neural")
            + ")",
            "mem_mb": round(tts_mb, 1),
            "avg_ms": round(avg_tts, 1),
            "kind": "local"
            if not getattr(session.settings, "tts_cloud", "")
            else "cloud",
        },
        {
            "name": "Wake & Voiceprint (CAM++)",
            "mem_mb": round(vp_mb, 1),
            "avg_ms": round(avg_vp, 1),
            "kind": "local",
        },
        {
            "name": "Silero VAD",
            "mem_mb": round(vad_mb, 1),
            "avg_ms": 0.5 if vad_loaded else 0.0,
            "kind": "local",
        },
        {
            "name": "Python Core + UI",
            "mem_mb": round(core_mb, 1),
            "avg_ms": 0.0,
            "cpu_pct": main_cpu,
            "kind": "process",
        },
        {
            "name": f"Pi Agent Panes ({len(pane_procs)})",
            "mem_mb": round(pi_rss_mb, 1),
            "avg_ms": round(avg_llm, 1),
            "cpu_pct": round(pi_cpu_pct, 1),
            "kind": "subprocess",
        },
    ]

    sh = system_health()
    tps_vals = [
        float((t.get("timings") or {}).get("tok_per_s") or 0.0)
        for t in turns
        if (t.get("timings") or {}).get("tok_per_s")
    ]
    avg_tps = round(sum(tps_vals) / len(tps_vals), 1) if tps_vals else 0.0
    by_model: dict[str, list[dict[str, float]]] = {}
    for t in turns:
        m_id = str(
            t.get("model") or getattr(session.settings, "model", "") or "unknown"
        )
        tm = t.get("timings") or {}
        by_model.setdefault(m_id, []).append(
            {
                "ttft": float(tm.get("first_token_ms") or 0.0),
                "tok_s": float(tm.get("tok_per_s") or 0.0),
            }
        )
    model_scorecard = []
    for m_id, m_rows in by_model.items():
        ttfts = [r["ttft"] for r in m_rows if r["ttft"] > 0]
        toks = [r["tok_s"] for r in m_rows if r["tok_s"] > 0]
        model_scorecard.append(
            {
                "model": m_id,
                "turns": len(m_rows),
                "avg_ttft_ms": round(sum(ttfts) / len(ttfts), 1) if ttfts else 0.0,
                "avg_tok_s": round(sum(toks) / len(toks), 1) if toks else 0.0,
            }
        )
    vp = getattr(session, "_voiceprint", None)
    nf = float(getattr(vp, "_noise_floor", 0.0) or 0.0)
    lvl = float(getattr(session, "level", 0.0) or 0.0)
    hd_warn = []
    if (
        float(sh.get("ram_total_gb", 0) or 0) - float(sh.get("ram_used_gb", 0) or 0)
        < 0.8
    ):
        hd_warn.append("low RAM")
    if float(sh.get("disk_free_gb", 0) or 0) < 1.0:
        hd_warn.append("low disk")

    return {
        "python_rss_mb": total_rss_mb,
        "pi_panes_rss_mb": round(pi_rss_mb, 1),
        "total_app_rss_mb": round(total_rss_mb + pi_rss_mb, 1),
        "features": features,
        "pane_procs": pane_procs,
        "reply_blocks": blocks,
        "classifier_perf": clf_stats,
        "gate_funnel": dict(getattr(session, "_gate_stats", {}) or {}),
        "snr_info": {
            "noise_floor": round(nf, 4),
            "snr_ratio": round(lvl / max(nf, 1e-4), 1) if lvl > 0 else 0.0,
        },
        "avg_tok_per_s": avg_tps,
        "model_scorecard": model_scorecard,
        "pi_headroom": {
            "ram_free_gb": round(
                max(
                    0.0,
                    float(sh.get("ram_total_gb", 0)) - float(sh.get("ram_used_gb", 0)),
                ),
                1,
            ),
            "disk_free_gb": float(sh.get("disk_free_gb", 0)),
            "warnings": hd_warn,
        },
    }


def performance_review(session: Any) -> dict[str, Any]:
    """Analyze the current session turns + logs and return actionable findings."""
    turns = [
        t for t in getattr(session, "turns", []) if t.get("source") in {"mic", "text"}
    ]
    if not turns:
        return {
            "turns": 0,
            "summary": "No chat turns recorded in this active session yet.",
            "bottleneck": "none",
            "findings": [],
        }

    def avg(key: str) -> float:
        vals = [float((t.get("timings") or {}).get(key) or 0) for t in turns]
        return round(sum(vals) / len(vals), 1) if vals else 0.0

    avg_asr = avg("asr_ms")
    avg_ttft = avg("first_token_ms")
    avg_tts = avg("tts_ms")
    avg_total = avg("total_ms")
    avg_ack = avg("filler_ms")
    tps_vals = [
        float((t.get("timings") or {}).get("tok_per_s") or 0)
        for t in turns
        if (t.get("timings") or {}).get("tok_per_s")
    ]
    avg_tps = round(sum(tps_vals) / len(tps_vals), 1) if tps_vals else 0.0
    gs = getattr(session, "_gate_stats", {}) or {}
    cut_offs = int(gs.get("cut_offs", 0))

    bottleneck = max(
        [("ASR", avg_asr), ("LLM first token", avg_ttft), ("TTS", avg_tts)],
        key=lambda x: x[1],
    )[0]
    findings: list[str] = []
    if avg_ttft > 2500:
        findings.append(
            f"LLM first token averages {avg_ttft:.0f} ms — switch Front to gpt-5.4-nano or north-mini-code:free with thinking off."
        )
    if cut_offs > 0:
        findings.append(
            f"{cut_offs} replies were cut off mid-playback — keep Stop-Word Barge-In ON in noisy rooms."
        )
    if gs.get("wake_blocked", 0) > gs.get("wake_allowed", 0):
        findings.append(
            f"Wake gate blocked {gs.get('wake_blocked')} vs {gs.get('wake_allowed')} allowed — check SNR/Voices dashboard."
        )
    if avg_tts > 3500:
        findings.append(
            f"TTS averages {avg_tts:.0f} ms — keep replies 1-2 sentences or use lessac / ElevenLabs Flash."
        )
    if avg_ack > 0 and avg_ack <= 350:
        findings.append(f"Ack filler is fast ({avg_ack:.0f} ms avg).")
    if hasattr(session, "classifier"):
        with contextlib.suppress(Exception):
            cs = session.classifier().stats
            if cs.get("calls", 0) > 0:
                findings.append(
                    f"Decision Classifier: {cs.get('calls')} calls · p95 {float(cs.get('p95_latency_ms') or 0):.2f} ms "
                    f"(max {float(cs.get('max_latency_ms') or 0):.2f} ms, budget <100 ms) · "
                    f"routes={cs.get('routes')}"
                )

    return {
        "turns": len(turns),
        "averages_ms": {
            "ack": avg_ack,
            "asr": avg_asr,
            "first_token": avg_ttft,
            "tts": avg_tts,
            "total": avg_total,
            "tok_per_s": avg_tps,
        },
        "bottleneck": bottleneck,
        "findings": findings,
        "summary": f"{len(turns)} turns · avg {avg_total:.0f} ms · {avg_tps} tok/s · cut-offs: {cut_offs} · bottleneck: {bottleneck}",
    }


def app_introspection(session: Any) -> dict[str, Any]:
    """Read-only view for the left-rail bottom inspector (prompts, context, skills, tools)."""
    roles = (
        "front",
        "classifier",
        "doctor",
        "back-1",
        "back-2",
        "back-3",
        "back-4",
        "back-5",
    )
    prompts = {}
    for r in roles:
        with contextlib.suppress(Exception):
            prompts[r] = session._role_system_prompt(r)

    # Context files: MEMORY.md + session_profile + prompts/*.md if present
    mem_path = BASE / "MEMORY.md"
    try:
        mem_text = (
            mem_path.read_text(encoding="utf-8", errors="replace")
            if mem_path.is_file()
            else "(none)"
        )
    except OSError:
        mem_text = "(unreadable)"
    prof = list(getattr(session, "_session_profile", []) or [])
    prompt_files = []
    pdir = BASE / "prompts"
    if pdir.is_dir():
        for pf in sorted(pdir.iterdir()):
            if pf.is_file():
                try:
                    prompt_files.append(
                        {
                            "name": pf.name,
                            "preview": pf.read_text(encoding="utf-8", errors="replace")[
                                :400
                            ],
                        }
                    )
                except OSError:
                    pass

    # Skill directory
    skills_dir = BASE / "skills"
    skills = []
    if skills_dir.is_dir():
        for sf in sorted(skills_dir.glob("*.md")):
            try:
                txt = sf.read_text(encoding="utf-8", errors="replace")
                skills.append(
                    {
                        "name": sf.stem,
                        "file": str(sf),
                        "bytes": len(txt),
                        "content": txt[:1200],
                    }
                )
            except OSError:
                pass

    # Available tool definitions (local toolbox + bridge + MCP)
    tools: list[dict[str, Any]] = []
    with contextlib.suppress(Exception):
        tb = getattr(session, "_tools", None)
        if tb and hasattr(tb, "schemas"):
            for sc in tb.schemas():
                fn = sc.get("function") or sc
                tools.append(
                    {
                        "source": "device_tool",
                        "name": fn.get("name", ""),
                        "description": fn.get("description", ""),
                        "parameters": fn.get("parameters", {}),
                    }
                )
    for name, desc in [
        (
            "delegate_task",
            "Run a background task on back/researcher/critic/summarizer/doctor",
        ),
        ("ask_agent", "Ask a specialist agent synchronously for an immediate answer"),
        ("start_crew", "Run a multi-agent crew pipeline across ordered roles"),
        (
            "doctor_app_control",
            "Doctor internal inspection + confirmation-gated settings/restart",
        ),
    ]:
        tools.append({"source": "sonj_bridge", "name": name, "description": desc})
    mcp_cfg = BASE / "mcp_servers.json"
    if mcp_cfg.is_file():
        try:
            raw = json.loads(mcp_cfg.read_text(encoding="utf-8"))
            for srv in raw.get("mcpServers") or raw or {}:
                tools.append(
                    {
                        "source": "mcp_server",
                        "name": str(srv),
                        "description": "configured MCP server",
                    }
                )
        except (OSError, ValueError):
            pass

    return {
        "system_prompts": prompts,
        "context": {
            "memory_file": str(mem_path),
            "memory_text": mem_text,
            "session_profile": prof,
            "prompt_files": prompt_files,
        },
        "skills": {
            "directory": str(skills_dir),
            "items": skills,
        },
        "tools": tools,
    }


_BACKCHANNELS = frozenset(
    {
        "oh",
        "uh",
        "um",
        "hmm",
        "hm",
        "mm",
        "mhm",
        "ah",
        "er",
        "uh huh",
        "uh-huh",
        "yeah",
        "yep",
        "yup",
        "ok",
        "okay",
        "alright",
        "right",
        "sure",
        "cool",
        "nice",
        "thanks",
        "thank you",
        "bye",
        "huh",
        "what",
    }
)

_ASK_CUES = re.compile(
    r"\b(nova|piattos|what|who|where|when|why|how|can you|could you|please|check|"
    r"tell|show|find|search|create|build|make|write|open|set|turn|play|pause|"
    r"stop|calculate|remember|list|read|verify|summarize|doctor|run|start|delete)\b",
    re.IGNORECASE,
)


def is_asking(text: str, pending_clarify: bool = False) -> tuple[bool, str]:
    """Ask-Detector: distinguish a real user request from ambient chatter/backchannels."""
    clean = re.sub(r"[^\w\s?']", " ", str(text or "")).strip().lower()
    if not clean:
        return False, "empty"
    if clean == "yes, i confirm the changes":
        return True, "doctor confirmation"
    if pending_clarify:
        return True, "clarification reply"
    if clean in _BACKCHANNELS:
        return False, f"backchannel ({clean!r})"
    words = clean.split()
    if len(words) <= 2 and "?" not in text and not _ASK_CUES.search(clean):
        return False, f"non-request fragment ({clean!r})"
    return True, "request detected"


DOCTOR_AUTO_CONFIRM_S = 15.0

_SETTINGS_DOCS: dict[str, str] = {
    "provider": "Active LLM provider ('github-copilot', 'openrouter', 'ollama')",
    "model": "Primary Front agent model id",
    "asr_backend": "Local offline ASR backend ('moonshine', 'parakeet', 'whisper')",
    "tts_enabled": "Enable spoken voice output (True/False)",
    "tts_voice": "Active TTS voice name",
    "tts_rate": "Speech rate (-1 default)",
    "barge_in": "Allow interrupting the agent mid-speech (True/False)",
    "barge_rms": "Minimum RMS loudness for open-mic barge-in",
    "barge_min_ms": "Minimum speech ms before barge-in triggers",
    "vad_threshold": "Silero VAD speech sensitivity (0.1..0.9)",
    "vad_min_silence": "Silence seconds before closing an utterance",
    "model_memory": "Ollama model keep-alive duration ('0s', '5m', '30m')",
    "tts_simulate": "Simulate TTS timing without audio device (test mode)",
    "idle_unload_s": "Seconds idle before freeing ASR/VAD models from RAM",
    "openrouter_order": "Preferred OpenRouter upstream routing provider",
    "session_cost_cap": "USD cap per session (0 = disabled)",
    "router_enabled": "Enable cheap/fast turn router (True/False)",
    "router_model": "Router model name",
    "router_timeout_ms": "Router timeout in ms",
    "filler_enabled": "Play fast pre-rendered acknowledgement clips (True/False)",
    "filler_delay_ms": "Delay ms before filler plays",
    "fast_tool_replies": "Speak quick tool outputs directly without extra LLM round",
    "stream_voice": "Speak sentences as they stream from LLM (True/False)",
    "smart_turn": "Hold unfinished thoughts open briefly before answering (True/False)",
    "interrupt_countdown_s": "Legacy countdown value (manual interrupt chip)",
    "paid_models_enabled": "Allow paid OpenRouter models in roster (True/False)",
    "front_uses_pi": "Run Front agent inside a live PiPane terminal (True/False)",
    "doctor_enabled": "Enable Doctor diagnostics & health checks (True/False)",
    "doctor_kill_switch": "Stop all automatic Doctor watch actions (True/False)",
    "doctor_watch": "Run background detect-only Doctor health watch (True/False)",
    "speech_cap_chars": "Max spoken characters for normal replies (0 = uncapped)",
    "empty_turn_reply": "Prompt 'Say it again?' on empty noise bursts (True/False)",
    "noise_gate": "Filter ambient-noise ASR hallucinations (True/False)",
    "speech_rms_floor": "Minimum RMS floor for utterances (0 = off)",
    "wake_gate": "Top-right wake gate: require 'Nova'/'Piattos' and speaker lock when ON",
    "approval_gate": "Require approval for risky shell/device actions (True/False)",
    "echo_cancellation": "Adaptive reference AEC: cancel agent's own TTS echo from mic (True/False)",
    "noise_suppression": "Telecon spectral noise suppression (fans/hum/typing) before VAD/ASR (True/False)",
    "chat_thinking": "Thinking level for chat turns ('off'|'low'|'medium'|'high')",
    "barge_strictness": "Barge-in sensitivity ('relaxed'|'normal'|'strict')",
    "wake_to_interrupt": "False=natural speech interrupts; True=stop-word only",
    "instant_barge": "Default False: when ON, speaking cuts TTS + LLM immediately; when OFF, stop words or Stop Reply cut",
    "front_only": "Toggle Front-Only mode: only Front answers (no Back/Doctor delegation)",
    "voiceprint_enabled": "Gate turns by enrolled owner voiceprint when wake is ON",
    "voiceprint_barge": "Require enrolled owner voiceprint to barge in",
    "session_learning": "Extract owner preferences into session profile (True/False)",
    "asr_cloud": "Optional cloud ASR override (''=local|'deepgram'|'openai')",
    "tts_cloud": "Optional cloud TTS override (''=local|'openai'|'elevenlabs')",
    "ask_detector": "Drop filler backchannels unless clarifying (True/False)",
    "wake_speaker_lock": "Multi-master 5m idle voice lock when wake gate is ON",
    "require_wake_when_idle": "Require 'Nova' after 5m idle when wake gate is ON",
    "allow_local_llm": "Allow local Ollama models (True/False)",
    "announce_task_completion": "Speak completed/failed delegated task reports when idle",
    "elevenlabs_amortize_spend": "Include amortized ElevenLabs char cost in total spend",
    "bg_voice_scan_always": "Scan background room voices even when Voices tab closed",
    "nova_voice": "Voice preset for Nova persona",
    "piattos_voice": "Voice preset for Piattos persona",
    "barge_triggers": "List of stop words that interrupt playback immediately",
}


def settings_catalog(session: Any) -> dict[str, Any]:
    """Full live catalog of every Settings field: current value, default, type, description."""
    st = getattr(session, "settings", None)
    if st is None:
        return {"settings": []}
    cls = type(st)
    defaults = cls()
    secret_keys = {
        "openrouter_key",
        "deepgram_api_key",
        "openai_api_key",
        "elevenlabs_api_key",
    }
    rows: list[dict[str, Any]] = []
    for f in dataclasses.fields(cls):
        name = f.name
        cur = getattr(st, name, None)
        dflt = getattr(defaults, name, None)
        if name in secret_keys:
            cur_out: Any = "(set)" if cur else "(empty)"
            dflt_out: Any = "(secret)"
        else:
            cur_out = cur
            dflt_out = dflt
        rows.append(
            {
                "key": name,
                "value": cur_out,
                "default": dflt_out,
                "type": type(dflt).__name__,
                "changeable_by_doctor": name not in secret_keys,
                "description": _SETTINGS_DOCS.get(name, f"Setting {name}"),
            }
        )
    return {"count": len(rows), "settings": rows}


def inspect_architecture(session: Any) -> dict[str, Any]:
    """Complete architectural map of SONJ4 modules, API endpoints, agents, and MCP servers."""
    modules = {
        "server.py": "Main HTTP API, Session engine, audio listen/VAD loop, barge-in, 7 Pi panes, announcements, Doctor control",
        "classifier.py": "Non-LLM Decision Classifier (<100ms hard deadline, +10% Front bias, front | back | doctor)",
        "agent.py": "Fallback in-process Agent loop, ToolTrace, compaction, task-agnostic Nova DEFAULT_SYSTEM_PROMPT",
        "toolbox.py": "Workspace, shell/command, web search/fetch, delegation, and agent-to-agent bridge tools",
        "tasks.py": "Background TaskManager across 5-slot Back pool (back-1..back-5), sequential/hierarchical crews, completion announcements",
        "syshealth.py": "CPU/RAM/temp telemetry, feature distribution (+classifier), performance review, introspection, DoctorGate (15s confirm)",
        "voiceprint.py": "CAM++ 3d-speaker embeddings, top-5 voices, multi-master wake lock",
        "asr.py": "Thread-safe Sherpa-ONNX Moonshine/Parakeet/Whisper + optional cloud ASR",
        "tts.py": "Piper/Kokoro/ElevenLabs/OpenAI TTS + filler bank + instant barge-in abort",
        "vad.py": "Silero VAD streaming + partial() live-speech access",
        "aec.py": "Adaptive reference NLMS Echo Canceller (nlms-aec) + Telecon Wiener Noise Suppressor (wiener-ns)",
        "pi_rpc.py": "Live Pi RPC subprocess manager (front, back-1..back-5, doctor panes)",
        "doctor.py": "Startup/runtime health checks & safe repairs",
        "computer_mcp.py": "Full desktop computer-use MCP server (Back agent only)",
        "browser_mcp.py": "Playwright browser automation MCP server (Back agent only)",
        "static/index.html": "WebView2 single-page console UI (Dashboard, Chat, Community, Pipeline, Voices, Settings, SysInfo)",
    }
    routes = [
        "GET /api/state",
        "GET /api/voices",
        "GET /api/sysinfo",
        "POST /api/send",
        "POST /api/listen",
        "POST /api/stop",
        "POST /api/barge",
        "POST /api/settings",
        "POST /api/delegate",
        "POST /api/ask_agent",
        "POST /api/crew",
        "POST /api/mcp/call",
        "POST /api/doctor/control",
        "POST /api/doctor/confirm",
        "POST /api/doctor/cancel",
        "POST /api/pi/prompt",
        "POST /api/pi/abort",
        "POST /api/shutdown",
    ]
    mcp_servers: list[str] = []
    mcp_cfg = BASE / "mcp_servers.json"
    if mcp_cfg.is_file():
        try:
            raw = json.loads(mcp_cfg.read_text(encoding="utf-8"))
            mcp_servers = [
                str(s.get("name"))
                for s in (raw.get("servers") or [])
                if isinstance(s, dict) and s.get("name")
            ]
        except (OSError, ValueError):
            mcp_servers = []
    agents_cfg: dict[str, Any] = {}
    agents_path = BASE / "agents.json"
    if agents_path.is_file():
        try:
            agents_cfg = json.loads(agents_path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            agents_cfg = {}
    return {
        "base_dir": str(BASE),
        "workspace": str(BASE.parent),
        "modules": modules,
        "api_routes": routes,
        "agents": {
            k: {
                "model": (v or {}).get("model", ""),
                "thinking": (v or {}).get("thinking", ""),
                "description": (v or {}).get("description", ""),
            }
            for k, v in agents_cfg.items()
            if isinstance(v, dict)
        },
        "mcp_servers": mcp_servers,
        "doctor_auto_confirm_s": DOCTOR_AUTO_CONFIRM_S,
    }


class DoctorGate:
    """Safety gate for Doctor: no self-coding; changes pop up with 15s auto-confirm."""

    def __init__(self, auto_confirm_s: float = DOCTOR_AUTO_CONFIRM_S) -> None:
        self._lock = threading.Lock()
        self._pending: dict[str, Any] | None = None
        self._auto_confirm_s = float(auto_confirm_s)

    def pending(self) -> dict[str, Any] | None:
        with self._lock:
            if not self._pending:
                return None
            out = dict(self._pending)
            rem = max(0.0, float(out.get("expires_at", 0.0)) - time.time())
            out["remaining_s"] = round(rem, 1)
            return out

    def propose(
        self, action: str, payload: dict[str, Any], summary: str
    ) -> dict[str, Any]:
        # Hard guard: never allow self-coding actions
        if action in {"edit_code", "write_file", "run_shell", "patch"}:
            return {
                "ok": False,
                "error": "Self-coding is disabled for Doctor. Only settings, models, and restart can be proposed.",
            }
        now = time.time()
        with self._lock:
            self._pending = {
                "action": action,
                "payload": payload,
                "summary": summary,
                "created_at": time.strftime("%H:%M:%S"),
                "expires_at": now + self._auto_confirm_s,
                "auto_confirm_s": self._auto_confirm_s,
                "remaining_s": self._auto_confirm_s,
            }
            snap = dict(self._pending)
        return {
            "ok": True,
            "status": "confirmation_required",
            "auto_confirm_s": self._auto_confirm_s,
            "prompt": (
                f"Pop-up opened: {summary}. Auto-confirms in {int(self._auto_confirm_s)}s "
                f"unless cancelled (or confirm now / say '{CONFIRM_PHRASE}')."
            ),
            "pending": snap,
        }

    def cancel(self) -> dict[str, Any] | None:
        with self._lock:
            prev = self._pending
            self._pending = None
            return dict(prev) if prev else None

    def _apply_locked_act(self, act: dict[str, Any], session: Any) -> dict[str, Any]:
        action = act["action"]
        payload = act["payload"]
        if action == "set_settings":
            session.update_settings(payload)
            return {
                "ok": True,
                "applied": action,
                "summary": act["summary"],
                "payload": payload,
            }
        if action == "set_role_model":
            msg = session.set_role_model(
                str(payload.get("role") or ""), str(payload.get("model") or "")
            )
            return {
                "ok": True,
                "applied": action,
                "summary": act["summary"],
                "message": msg,
            }
        if action == "restart":
            session.schedule_restart()
            return {
                "ok": True,
                "applied": "restart",
                "summary": act["summary"],
                "message": "Restarting SONJ engine now.",
            }
        if action == "reset_session":
            session.reset()
            return {
                "ok": True,
                "applied": "reset_session",
                "summary": act["summary"],
                "message": "Session cleared.",
            }
        return {"ok": False, "error": f"unknown Doctor action: {action}"}

    def auto_confirm_if_expired(self, session: Any) -> dict[str, Any] | None:
        """Apply pending Doctor change automatically if its 15s window elapsed."""
        with self._lock:
            if self._pending is None:
                return None
            if time.time() < float(self._pending.get("expires_at", 0.0)):
                return None
            act = self._pending
            self._pending = None
        res = self._apply_locked_act(act, session)
        res["auto_confirmed"] = True
        return res

    def confirm_and_apply(
        self, reply_text: str, session: Any, force: bool = False
    ) -> dict[str, Any]:
        with self._lock:
            if self._pending is None:
                return {"ok": False, "error": "no pending Doctor change"}
            if not force and str(reply_text).strip() != CONFIRM_PHRASE:
                return {
                    "ok": False,
                    "error": f"Exact confirmation phrase required: {CONFIRM_PHRASE}",
                    "pending": dict(self._pending),
                }
            act = self._pending
            self._pending = None
        return self._apply_locked_act(act, session)
