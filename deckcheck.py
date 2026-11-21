"""QA checks for MEX3 Deck alignment, topbar buttons/tooltips, half-way interrupt, and 5-task Front-Only cycle.

Run unit checks: python deckcheck.py
Run full live UI + 5-task Front-Only QA against running engine: python deckcheck.py --live --port 8766
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import urllib.request
from pathlib import Path

from console import use_utf8_stdout

RESULTS: list[tuple[str, bool, str]] = []


def check(name: str, ok: bool, detail: str = "") -> None:
    RESULTS.append((name, bool(ok), detail))
    print(f"[{'PASS' if ok else 'FAIL'}] {name} {detail}", flush=True)


def run_unit() -> int:
    os.environ["ME3_NO_PERSIST"] = "1"
    from agent import DEFAULT_SYSTEM_PROMPT
    from server import Session, load_settings, sanitize_for_speech

    st = load_settings()
    st.front_uses_pi = False
    st.tts_enabled = False
    st.tts_simulate = True
    s = Session(st)

    # 1. Task-agnostic Front prompt + versioned prompt file
    pfile = Path(__file__).resolve().parent / "prompts" / "front_system_v4.txt"
    check("prompt_file_versioned_exists", pfile.is_file(), str(pfile.name))

    front_prompt = s._role_system_prompt("front")
    forbidden_tasks = (
        "weather",
        "get_time",
        "set_timer",
        "set_alarm",
        "set_lights",
        "thermostat",
        "calculate",
        "snake",
        "gazette",
        "music",
        "volume",
    )
    for txt_name, txt in [
        ("server_front", front_prompt),
        ("agent_default", DEFAULT_SYSTEM_PROMPT),
    ]:
        hits = [w for w in forbidden_tasks if w in txt.lower()]
        check(f"{txt_name}_no_specific_tasks", not hits, f"forbidden_hits={hits}")
        low = txt.lower()
        check(
            f"{txt_name}_has_deck_rules",
            "nova" in low
            and "markdown" in low
            and "url" in low
            and "mcp" in low
            and "stop and listen" in low,
            "missing required deck directive",
        )
    pfile_txt = pfile.read_text(encoding="utf-8").strip()
    check(
        "front_prompt_single_source_exact_match",
        front_prompt.strip() == pfile_txt == DEFAULT_SYSTEM_PROMPT.strip(),
        f"lens=({len(front_prompt.strip())},{len(pfile_txt)},{len(DEFAULT_SYSTEM_PROMPT.strip())})",
    )

    # 2. Never read markdown or URLs aloud
    dirty = "**Hello** `world`! Check [the Python docs](https://docs.python.org/3/library/pathlib.html) and visit https://example.com/a/b?x=1 #note"
    clean = sanitize_for_speech(dirty)
    check(
        "sanitize_strips_markdown_and_urls",
        ("https://" not in clean)
        and ("http://" not in clean)
        and ("**" not in clean)
        and ("`" not in clean)
        and ("the Python docs" in clean),
        f"clean={clean!r}",
    )

    # 3. instant_barge default False + realistic gates + interrupt() aborts LLM & filler & audio
    check(
        "instant_barge_default_false",
        getattr(st, "instant_barge", None) is False,
        f"val={getattr(st, 'instant_barge', None)}",
    )
    thresh, min_ms, _tail = s._barge_gates()
    check(
        "barge_gates_conversational_reachable",
        thresh <= 0.10 and min_ms <= 300.0,
        f"thresh={thresh:.3f} min_ms={min_ms:.0f}",
    )
    aborted: list[str] = []
    s.pi_abort = lambda role="": aborted.append(role) or "aborted"  # type: ignore[method-assign]
    s._speaking = True
    s._speech_run = [object()]  # type: ignore[list-item]
    s._held_text = "stale partial"
    seq0 = s._interrupt_seq
    s.interrupt()
    check(
        "interrupt_stops_llm_filler_audio_and_bumps_seq",
        ("front" in aborted)
        and (s._interrupt_seq == seq0 + 1)
        and (not s._speaking)
        and (len(s._speech_run) == 0)
        and (s._held_text == ""),
        f"aborted={aborted} seq={s._interrupt_seq}",
    )

    # 4. Mid-thought ask() abort must NOT fall back to other LLMs when interrupted
    fallback_calls: list[str] = []

    class _AbortingAgent:
        def __init__(self) -> None:
            self.history: list = []

        def respond(self, *a, **kw):
            s.interrupt()
            raise RuntimeError("aborted mid-turn")

    s._role_fallback_chain = lambda: (
        fallback_calls.append("called") or ["openrouter/x:free"]
    )  # type: ignore[method-assign]
    ab_agent = _AbortingAgent()
    s._should_use_front_pi = lambda: False  # type: ignore[method-assign]
    s._agent = ab_agent  # type: ignore[assignment]
    s.agent = lambda: ab_agent  # type: ignore[method-assign]
    turn_cut = s.ask("tell me a long explanation", source="text")
    s._agent = None
    check(
        "interrupted_ask_skips_fallback_chain",
        not fallback_calls and "(cut off)" in str(turn_cut.get("assistant_text", "")),
        f"fallbacks={fallback_calls} reply={turn_cut.get('assistant_text')!r}",
    )

    # 5. 5 tasks under Front-Only ON -> then toggle OFF -> then ON -> OFF
    five_tasks = [
        "Build a tiny Python palindrome checker script and test it",
        "Research the latest Raspberry Pi 5 benchmark numbers",
        "Diagnose audio and model health errors in the app",
        "Compare Piper and Kokoro TTS speed in a summary",
        "Audit the workspace files and report how many files exist",
    ]
    check(
        "front_only_default_false",
        getattr(st, "front_only", None) is False,
        f"val={st.front_only}",
    )
    s.update_settings({"front_only": True})
    routes_on = [s.classify_decision(q).route for q in five_tasks]
    delegations_on = [s.delegate_task(q, role="back") for q in five_tasks]
    doctor_on = s.ask_agent_turn("doctor", "check logs")
    check(
        "five_tasks_front_only_on_stay_on_front",
        all(r == "front" for r in routes_on)
        and all("front-only" in d.lower() for d in delegations_on)
        and ("front-only" in doctor_on.lower())
        and len(s.tasks_manager().list()) == 0,
        f"routes={routes_on} tasks_queued={len(s.tasks_manager().list())}",
    )

    s.update_settings({"front_only": False})
    routes_off = [s.classify_decision(q).route for q in five_tasks]
    check(
        "front_only_off_restores_back_and_doctor_routing",
        ("back" in routes_off) and ("doctor" in routes_off),
        f"routes={routes_off}",
    )

    # Cycle ON -> OFF again
    s.update_settings({"front_only": True})
    cycle_on = s.classify_decision(five_tasks[0]).route
    s.update_settings({"front_only": False})
    cycle_off = s.classify_decision(five_tasks[0]).route
    check(
        "front_only_toggle_on_off_cycle",
        cycle_on == "front" and cycle_off == "back",
        f"cycle_on={cycle_on} cycle_off={cycle_off}",
    )

    # 6. Reference-based NLMS AEC cancels own TTS echo while keeping double-talk user speech
    import numpy as np

    from aec import EchoSuppressor, NoiseSuppressor, feed_reference

    rng = np.random.default_rng(7)
    tts_ref = (
        0.35 * np.sin(2 * np.pi * 240.0 * np.arange(16000) / 16000.0)
        + 0.25 * np.sin(2 * np.pi * 620.0 * np.arange(16000) / 16000.0)
    ).astype(np.float32)
    feed_reference(tts_ref, 16000)
    # Delayed + room-attenuated echo (30 ms delay + second reflection tap)
    delay = 480
    echo_sig = (
        0.55 * np.roll(tts_ref, delay) + 0.18 * np.roll(tts_ref, delay + 32)
    ).astype(np.float32)
    aec = EchoSuppressor()
    raw_echo_rms: list[float] = []
    clean_echo_rms: list[float] = []
    echo_flags: list[bool] = []
    for pos in range(4096, 12288, 512):
        blk = echo_sig[pos : pos + 512]
        out, is_e = aec.process(blk)
        raw_echo_rms.append(float(np.sqrt(np.mean(blk * blk))))
        clean_echo_rms.append(float(np.sqrt(np.mean(out * out))))
        echo_flags.append(is_e)
    erle_db = 20.0 * np.log10((sum(raw_echo_rms) + 1e-9) / (sum(clean_echo_rms) + 1e-9))
    # Double-talk: mix distinct user voice frequency (1180 Hz) over TTS echo
    user_voice = (
        0.28 * np.sin(2 * np.pi * 1180.0 * np.arange(16000) / 16000.0)
    ).astype(np.float32)
    dt_out, _ = aec.process(
        (echo_sig[8192:8704] + user_voice[8192:8704]).astype(np.float32)
    )
    dt_rms = float(np.sqrt(np.mean(dt_out * dt_out)))
    check(
        "nlms_aec_cancels_tts_echo_and_preserves_double_talk",
        erle_db >= 8.0 and any(echo_flags) and dt_rms >= 0.08,
        f"erle_db={erle_db:.1f}dt_rms={dt_rms:.3f}",
    )

    # 7. Telecon Wiener NoiseSuppressor suppresses steady fan/room noise while passing speech
    ns = NoiseSuppressor()
    steady_noise = rng.normal(0.0, 0.05, 16000).astype(np.float32)
    for pos in range(0, 8192, 512):
        ns.process(steady_noise[pos : pos + 512])
    noisy_blk = steady_noise[8192:8704]
    cleaned_noise = ns.process(noisy_blk)
    noise_att_db = 20.0 * np.log10(
        (float(np.sqrt(np.mean(noisy_blk * noisy_blk))) + 1e-9)
        / (float(np.sqrt(np.mean(cleaned_noise * cleaned_noise))) + 1e-9)
    )
    speech_plus_noise = (
        noisy_blk + 0.35 * np.sin(2 * np.pi * 440.0 * np.arange(512) / 16000.0)
    ).astype(np.float32)
    cleaned_speech = ns.process(speech_plus_noise)
    speech_rms = float(np.sqrt(np.mean(cleaned_speech * cleaned_speech)))
    check(
        "telecon_wiener_ns_reduces_room_noise_preserves_speech",
        noise_att_db >= 6.0 and speech_rms >= 0.14,
        f"noise_att_db={noise_att_db:.1f} speech_rms={speech_rms:.3f}",
    )

    # 8. Decision Classifier performance stats & feature_distribution telemetry
    from syshealth import feature_distribution

    s.classify_decision("what time is it")
    s.classify_decision("build a python calculator script")
    dist = feature_distribution(s)
    clf_perf = dist.get("classifier_perf") or {}
    feat_names = " | ".join(f.get("name", "") for f in (dist.get("features") or []))
    check(
        "classifier_perf_telemetry_in_distribution",
        int(clf_perf.get("calls") or 0) >= 2
        and len(clf_perf.get("recent_latencies") or []) >= 2
        and int(clf_perf.get("front_bias_pct") or 0) == 10
        and "Decision Classifier" in feat_names
        and "Echo Canceller" in feat_names
        and "Noise Suppressor" in feat_names,
        f"calls={clf_perf.get('calls')} p95={clf_perf.get('p95_latency_ms')}ms",
    )

    passed = sum(1 for _, ok, _ in RESULTS if ok)
    print(f"\nUNIT RESULT: {passed}/{len(RESULTS)} passed")
    return 0 if passed == len(RESULTS) else 1


def _http(
    base: str, path: str, body: dict | None = None, timeout: float = 30.0
) -> dict:
    data = None if body is None else json.dumps(body).encode("utf-8")
    req = urllib.request.Request(
        f"{base}{path}",
        data=data,
        headers={"Content-Type": "application/json"} if data else {},
        method="POST" if data is not None else "GET",
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        raw = resp.read().decode("utf-8", errors="replace")
    try:
        return json.loads(raw)
    except ValueError:
        return {"ok": False, "raw": raw[:200]}


def _wait_idle(
    base: str, prev_turns: int, timeout_s: float = 45.0
) -> tuple[dict | None, dict]:
    deadline = time.time() + timeout_s
    last = None
    while time.time() < deadline:
        st = _http(base, "/api/state")
        turns = st.get("turns") or []
        if len(turns) > prev_turns:
            last = turns[-1]
        if last is not None and st.get("status") == "idle":
            return last, st
        time.sleep(0.35)
    return last, _http(base, "/api/state")


def run_live(port: int) -> int:
    from playwright.sync_api import sync_playwright  # type: ignore[import-not-found]

    base = f"http://127.0.0.1:{port}"
    orig_st = _http(base, "/api/state").get("settings") or {}
    orig_tts = bool(orig_st.get("tts_enabled", True))
    orig_ib = bool(orig_st.get("instant_barge", True))
    orig_fo = bool(orig_st.get("front_only", False))

    try:
        # Silence speaker during automated 5-task QA while starting at exact shipped defaults
        _http(
            base,
            "/api/settings",
            {
                "tts_enabled": False,
                "instant_barge": False,
                "front_only": False,
                "echo_cancellation": True,
                "noise_suppression": True,
            },
        )
        _http(base, "/api/session/new", {})

        with sync_playwright() as p:
            browser = p.chromium.launch(channel="msedge", headless=True)
            page = browser.new_page(viewport={"width": 1440, "height": 900})
            js_errs: list[str] = []
            page.on("pageerror", lambda err: js_errs.append(str(err)))
            page.goto(base, wait_until="networkidle")
            page.wait_for_selector("#btn-instant-barge")
            page.wait_for_selector("#btn-front-only")
            page.wait_for_selector("#btn-barge")
            page.wait_for_timeout(450)

            # 1. Button visibility, clear labels, and specific tooltips
            btn_info = page.evaluate(
                """() => ['btn-instant-barge','btn-front-only','btn-aec','btn-ns','btn-barge','btn-listen','btn-wake'].map(id => {
                    const el = document.getElementById(id);
                    const r = el.getBoundingClientRect();
                    return {
                        id, text: el.textContent.trim(),
                        title: el.getAttribute('title') || '',
                        tip: el.getAttribute('data-tip') || '',
                        y: Math.round(r.top), vis: r.width > 10 && r.height > 10
                    };
                })"""
            )
            check(
                "live_topbar_buttons_and_specific_tooltips",
                all(
                    b["vis"]
                    and b["y"] < 60
                    and len(b["title"]) >= 80
                    and len(b["tip"]) >= 70
                    for b in btn_info
                ),
                ", ".join(
                    f"{b['id']}(y={b['y']},title={len(b['title'])}c)" for b in btn_info
                ),
            )

            # Hover check on #btn-front-only and #btn-instant-barge
            page.hover("#btn-instant-barge")
            page.wait_for_timeout(120)
            page.hover("#btn-front-only")
            page.wait_for_timeout(120)

            # 2. Auto-Interrupt starts OFF by default -> toggle ON -> toggle back OFF
            txt_init = page.text_content("#btn-instant-barge") or ""
            tip_init = page.get_attribute("#btn-instant-barge", "title") or ""
            page.click("#btn-instant-barge")
            page.wait_for_timeout(450)
            s_ib_on = _http(base, "/api/state")
            txt_on = page.text_content("#btn-instant-barge") or ""
            page.click("#btn-instant-barge")
            page.wait_for_timeout(450)
            s_ib_off2 = _http(base, "/api/state")
            txt_off2 = page.text_content("#btn-instant-barge") or ""
            check(
                "live_auto_interrupt_default_off_toggle_on_off",
                ("off" in txt_init.lower())
                and ("off (default)" in tip_init.lower())
                and bool(s_ib_on["settings"]["instant_barge"])
                and ("on" in txt_on.lower())
                and (not s_ib_off2["settings"]["instant_barge"])
                and ("off" in txt_off2.lower()),
                f"init={txt_init!r} -> on={txt_on!r} -> off={txt_off2!r}",
            )

            # 3. Half-way Stop Reply button (#btn-barge) test during active prompt
            b0 = int(_http(base, "/api/state").get("barge_ins") or 0)
            barge_reqs: list[str] = []
            stop_reqs: list[str] = []
            page.on(
                "request",
                lambda req: (
                    barge_reqs.append(req.url)
                    if req.url.endswith("/api/barge")
                    else (
                        stop_reqs.append(req.url)
                        if req.url.endswith("/api/stop")
                        else None
                    )
                ),
            )
            page.evaluate(
                "fetch('/api/send',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({text:'Write a very long 6 paragraph explanation of Fourier transforms step by step.'})})"
            )
            # Wait briefly until status becomes thinking or speaking, then click Stop Reply
            saw_busy = False
            for _ in range(30):
                st_b = _http(base, "/api/state")
                if st_b.get("status") in ("thinking", "speaking") or st_b.get(
                    "speaking"
                ):
                    saw_busy = True
                    break
                time.sleep(0.1)
            page.click("#btn-barge")
            page.wait_for_timeout(600)
            st_after = _http(base, "/api/state")
            b1 = int(st_after.get("barge_ins") or 0)
            check(
                "live_stop_reply_button_fires_barge_mid_turn",
                len(barge_reqs) >= 1 and len(stop_reqs) == 0 and b1 > b0,
                f"saw_busy={saw_busy} barge_reqs={len(barge_reqs)} stop_reqs={len(stop_reqs)} barge_ins {b0}->{b1}",
            )
            # Wait for worker to settle before 5-task suite
            _wait_idle(base, len(st_after.get("turns") or []), timeout_s=20.0)

            # 4. Click Front Only ON in UI -> run 5 live tasks -> verify 0 delegations -> toggle OFF -> ON -> OFF
            page.click("#btn-front-only")
            page.wait_for_timeout(450)
            s_fo_on = _http(base, "/api/state")
            tasks_before = len(s_fo_on.get("tasks") or [])
            five_live_prompts = [
                "Reply in 1 short sentence: write a one-line Python lambda that checks if a string is a palindrome.",
                "Reply in 1 short sentence: name two ways Raspberry Pi 5 improves on Raspberry Pi 4.",
                "Reply in 1 short sentence: what does Silero VAD check in a voice pipeline?",
                "Reply in 1 short sentence: compare Piper and Kokoro TTS in one contrast.",
                "Reply in 1 short sentence: why should spoken replies never read raw URLs aloud?",
            ]
            five_ok = 0
            for idx, q in enumerate(five_live_prompts, 1):
                cur_n = len(_http(base, "/api/state").get("turns") or [])
                _http(base, "/api/send", {"text": q})
                turn, _st_t = _wait_idle(base, cur_n, timeout_s=45.0)
                ans = str((turn or {}).get("assistant_text") or "").strip()
                ok_t = bool(ans) and "(cut off)" not in ans.lower()
                if ok_t:
                    five_ok += 1
                print(f"    [5-task #{idx}] ok={ok_t} reply={ans[:80]!r}", flush=True)
            tasks_after = len(_http(base, "/api/state").get("tasks") or [])
            check(
                "live_five_tasks_with_front_only_on_zero_delegations",
                bool(s_fo_on["settings"]["front_only"])
                and five_ok == 5
                and tasks_after == tasks_before,
                f"five_ok={five_ok}/5 tasks={tasks_before}->{tasks_after}",
            )

            # Turn OFF -> ON -> OFF via UI button
            page.click("#btn-front-only")
            page.wait_for_timeout(450)
            s_off1 = _http(base, "/api/state")
            page.click("#btn-front-only")
            page.wait_for_timeout(450)
            s_on2 = _http(base, "/api/state")
            page.click("#btn-front-only")
            page.wait_for_timeout(450)
            s_off2 = _http(base, "/api/state")
            check(
                "live_front_only_off_on_off_ui_cycle",
                (not s_off1["settings"]["front_only"])
                and bool(s_on2["settings"]["front_only"])
                and (not s_off2["settings"]["front_only"])
                and ("OFF" in (page.text_content("#btn-front-only") or ""))
                and not js_errs,
                f"cycle=[{s_off1['settings']['front_only']},{s_on2['settings']['front_only']},{s_off2['settings']['front_only']}] js_errs={js_errs}",
            )
            # 5. Live AEC & Telecon NS topbar button toggles + tooltip checks
            page.click("#btn-aec")
            page.wait_for_timeout(400)
            s_aec_off = _http(base, "/api/state")
            page.click("#btn-aec")
            page.wait_for_timeout(400)
            s_aec_on = _http(base, "/api/state")
            page.click("#btn-ns")
            page.wait_for_timeout(400)
            s_ns_off = _http(base, "/api/state")
            page.click("#btn-ns")
            page.wait_for_timeout(400)
            s_ns_on = _http(base, "/api/state")
            check(
                "live_aec_and_ns_topbar_toggles",
                (not s_aec_off["settings"]["echo_cancellation"])
                and bool(s_aec_on["settings"]["echo_cancellation"])
                and (not s_ns_off["settings"]["noise_suppression"])
                and bool(s_ns_on["settings"]["noise_suppression"])
                and s_ns_on.get("echo_engine") == "nlms-aec"
                and s_ns_on.get("ns_engine") == "wiener-ns",
                f"aec={s_aec_off['settings']['echo_cancellation']}->{s_aec_on['settings']['echo_cancellation']} ns={s_ns_off['settings']['noise_suppression']}->{s_ns_on['settings']['noise_suppression']}",
            )

            # 6. Live Workflow pane + Lower panes + Pipeline pane verification
            page.click(".rail button[data-view='chat']")
            page.wait_for_timeout(350)
            wf_txt = page.inner_text("#tab-workflow")
            pl_txt = page.inner_text("#tab-pipeline")
            pl_count = page.locator("#tab-pipeline .pv").count()
            check(
                "live_workflow_and_pipeline_panes_updated",
                ("audio cleanup" in wf_txt)
                and ("classifier" in wf_txt)
                and ("back-1" in wf_txt)
                and ("back-5" in wf_txt)
                and ("doctor" in wf_txt)
                and ("researcher" not in wf_txt)
                and pl_count == 8
                and ("AEC" in pl_txt),
                f"pv={pl_count} wf_has_back1_to_5={'back-1' in wf_txt and 'back-5' in wf_txt}",
            )

            # 7. Decision Classifier Performance card + chart + Docs view verification
            page.click(".rail button[data-view='performance']")
            page.wait_for_timeout(450)
            clf_card_txt = page.inner_text("#perf-clf-card")
            clf_stage_txt = page.inner_text("#perf-clf-stage")
            page.click(".rail button[data-view='docs']")
            page.wait_for_timeout(450)
            docs_full_txt = page.inner_text("#docs-body")
            page.click("#docs-tab-summary")
            page.wait_for_timeout(250)
            docs_sum_txt = page.inner_text("#docs-body")
            check(
                "live_classifier_performance_and_docs_updated",
                ("decision classifier performance" in clf_card_txt.lower())
                and ("calls" in clf_card_txt.lower())
                and ("p95 ms" in clf_card_txt.lower())
                and bool(clf_stage_txt.strip())
                and ("nlms-aec" in docs_full_txt)
                and ("wiener-ns" in docs_full_txt)
                and ("nlms-aec" in docs_sum_txt)
                and ("Auto-Interrupt" in docs_sum_txt),
                f"clf_stage={clf_stage_txt!r} docs_ok=True",
            )

            # 8. Zero horizontal scrollbar across viewports & primary views
            overflow_issues: list[str] = []
            for vw, vh in ((1440, 900), (1280, 800)):
                page.set_viewport_size({"width": vw, "height": vh})
                for vname in (
                    "chat",
                    "dashboard",
                    "performance",
                    "pipeline",
                    "docs",
                    "settings",
                ):
                    page.click(f".rail button[data-view='{vname}']")
                    page.wait_for_timeout(200)
                    dims = page.evaluate(
                        "() => ({sw: document.documentElement.scrollWidth, cw: document.documentElement.clientWidth})"
                    )
                    if int(dims["sw"]) > int(dims["cw"]):
                        overflow_issues.append(
                            f"{vw}x{vh}/{vname}:{dims['sw']}>{dims['cw']}"
                        )
            check(
                "live_zero_horizontal_scrollbar",
                not overflow_issues and not js_errs,
                f"issues={overflow_issues or 'none'} js_errs={js_errs}",
            )
            browser.close()
    finally:
        _http(
            base,
            "/api/settings",
            {
                "tts_enabled": orig_tts,
                "instant_barge": False,
                "front_only": False,
                "echo_cancellation": True,
                "noise_suppression": True,
            },
        )
        _http(base, "/api/session/new", {})

    passed = sum(1 for _, ok, _ in RESULTS if ok)
    print(
        f"\nLIVE RESULT: {passed}/{len(RESULTS)} passed (orig instant_barge={orig_ib}, front_only={orig_fo} -> restored ON/OFF)"
    )
    return 0 if passed == len(RESULTS) else 1


def main() -> int:
    use_utf8_stdout()
    ap = argparse.ArgumentParser()
    ap.add_argument("--live", action="store_true")
    ap.add_argument("--port", type=int, default=8766)
    args = ap.parse_args()
    rc = run_unit()
    if rc != 0:
        return rc
    if args.live:
        return run_live(args.port)
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception as exc:  # noqa: BLE001
        print(f"[FAIL] deckcheck error: {type(exc).__name__}: {exc}")
        sys.exit(1)
