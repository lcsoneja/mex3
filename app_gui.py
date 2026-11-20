"""Standalone desktop app for the ME3 laptop voice agent (Tkinter, no browser).

The engine is the same one the headless server uses: this file only builds the
window, renders turns, and forwards button presses. Audio runs on the laptop
itself.

Run:
    python me3_laptop/app_gui.py
"""

from __future__ import annotations

import contextlib
import sys
import time
import tkinter as tk
import traceback
import types
from tkinter import ttk
from typing import Any

from agentsdialog import AgentsDialog
from asr import discover_models
from chatview import ChatView, StateIndicator
from doctor import AGENTS as DOCTOR_AGENTS
from doctor import CATALOGUE as DOCTOR_REPAIRS
from doctor import SCOPES as DOCTOR_SCOPES
from doctorpane import DoctorPane
from explorerpane import ExplorerPane
from historypane import HistoryPane
from mindmappane import MindMapPane
from server import (
    MIN_BARGE_RMS,
    MODELS_DIR,
    Session,
    Settings,
    claim_engine,
    load_settings,
    release_engine,
)
from subagentspane import SubagentsPane
from thinkingpane import ThinkingPane
from tts import TtsError, list_all_voices
from workflowview import WorkflowView


def _threshold_words(value: float) -> str:
    """Plain-language meaning of a threshold, so the slider is not a mystery.

    The original control only showed a number, which is how a user ended up
    dragging it to maximum sensitivity and making the agent interrupt itself.
    """
    if value <= 0.05:
        return "very easy to interrupt"
    if value <= 0.09:
        return "balanced"
    return "needs a loud voice"


def _as_number(value: Any, default: float) -> float:
    """Slider callbacks and config files both reach the UI as untrusted text."""
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


MEMORY_CHOICES = {
    "unload immediately (0s)": "0s",
    "keep 5 min (default)": "5m",
    "keep 30 min (fastest)": "30m",
}

POLL_MS = 400
FONT_UI = ("Segoe UI", 10)
FONT_SMALL = ("Segoe UI", 8)
FONT_MONO = ("Consolas", 9)
FONT_BOLD = ("Segoe UI", 10, "bold")


class VoiceAgentApp:
    """The desktop window: controls on the left, transcript on the right."""

    def __init__(self, root: tk.Tk, session: Session) -> None:
        self.root = root
        self.session = session
        self.rendered: set[int] = set()
        self.models: list[str] = []
        self.voices: list[str] = []
        self._applying = False
        # S1.2: a Tk callback that raises becomes an incident, not a dead window.
        # Installed here, not in main(), so every construction path has it
        # (the probe found this: main() was the only caller that set it).
        root.report_callback_exception = self.report_callback_exception

        root.title("ME3 Voice Agent - laptop dev build")
        # Fit the screen: at 150% display scaling a fixed 740-tall window ran off
        # the bottom, hiding the talk buttons entirely.
        screen_w = root.winfo_screenwidth()
        screen_h = root.winfo_screenheight()
        width = max(900, min(1120, int(screen_w * 0.82)))
        height = max(520, min(760, int(screen_h * 0.82)))
        root.geometry(f"{width}x{height}+40+20")
        root.minsize(820, 500)

        # Order matters for pack: the control rows claim their height first, then
        # the transcript takes what is left. Otherwise a short screen clips them.
        self._build_top()
        self._build_bottom()
        self._build_body()
        self._load_models()
        self._load_voices()
        self._apply_settings_to_widgets()
        self._validate_model()
        self._poll()

    # ------------------------------------------------------------------ layout
    def _build_top(self) -> None:
        bar = ttk.Frame(self.root, padding=(10, 8))
        bar.pack(fill="x")
        self.status_var = tk.StringVar(value="idle")
        self.mode_var = tk.StringVar(value="mic off")
        self.model_var = tk.StringVar(value="-")
        # Who is speaking, in words as well as colour - the clearest possible answer to
        # "is it listening, thinking, or talking right now?"
        self.speaker_var = tk.StringVar(value="idle")
        self.indicator = StateIndicator(bar)
        self.indicator.pack(side="left", padx=(0, 14))
        ttk.Label(bar, textvariable=self.mode_var, font=FONT_SMALL).pack(
            side="left", padx=(0, 14)
        )
        ttk.Label(bar, textvariable=self.model_var, font=FONT_SMALL).pack(side="left")
        self.speaker_label = ttk.Label(
            bar, textvariable=self.speaker_var, font=FONT_BOLD
        )
        self.speaker_label.pack(side="left", padx=(14, 0))
        ttk.Button(bar, text="New chat", command=self._new_chat).pack(side="right")
        ttk.Button(bar, text="Agents", command=self._open_agents).pack(
            side="right", padx=(0, 6)
        )
        self.view_var = tk.StringVar(value="workflow")
        # M1.3: Thinking joins the right-pane stack. S2 adds the Doctor.
        for view_name in ("doctor", "thinking", "mindmap", "subagents", "workflow"):
            ttk.Radiobutton(
                bar,
                text=view_name.title(),
                value=view_name,
                variable=self.view_var,
                command=self._show_view,
            ).pack(side="right", padx=(0, 4))

        # S4: which agents are working right now - always visible on the window,
        # so "is anything actually running?" never needs a tab or a console.
        strip = ttk.Frame(self.root, padding=(10, 0, 10, 6))
        strip.pack(fill="x")
        self.agents_var = tk.StringVar(value="agents: starting")
        self.agents_label = ttk.Label(
            strip, textvariable=self.agents_var, font=FONT_SMALL, foreground="#4b5563"
        )
        self.agents_label.pack(side="left")

    def _build_body(self) -> None:
        body = ttk.Frame(self.root)
        body.pack(fill="both", expand=True)

        # The control column is taller than a short screen, so it scrolls. The
        # canvas gets an explicit small width to stop it sizing itself from its
        # contents (which would push the chat off screen).
        # Left rail: the controls, plus the panes a session actually needs (history, files,
        # and where the roster lives). Tabs keep the width small and the window usable.
        rail = ttk.Notebook(body, width=350)
        rail.pack(side="left", fill="y")
        self.rail = rail
        controls_tab = ttk.Frame(rail)
        rail.add(controls_tab, text="Controls")
        # #26/#9/#61: the pane owns its selection and its buttons; the app only
        # supplies the callbacks, so no cross-file attribute access is needed.
        self.history = HistoryPane(
            rail,
            on_open=self._open_session,
            on_resume=self._resume_session,
            on_fork=self._fork_session,
            on_export=self._export_markdown,
        )
        rail.add(self.history, text="Sessions")
        self.explorer = ExplorerPane(rail, on_open_file=self._open_file)
        rail.add(self.explorer, text="Files")
        agents_tab = ttk.Frame(rail, padding=(6, 4))
        rail.add(agents_tab, text="Agents")
        # W1.2/W1.3: a per-role roster - every role gets its own model dropdown,
        # fed by the free-model picker, written back to agents.json (routing is data).
        ttk.Label(agents_tab, text="Per-role models", font=FONT_BOLD).pack(anchor="w")
        self._role_rows: dict[str, ttk.Combobox] = {}
        self._role_frame = ttk.Frame(agents_tab)
        self._role_frame.pack(fill="x", pady=(4, 6))
        ttk.Button(
            agents_tab, text="Refresh free models", command=self._refresh_free_models
        ).pack(anchor="w")
        ttk.Button(
            agents_tab, text="Open roster editor", command=self._open_agents
        ).pack(anchor="w", pady=(4, 0))
        # M3.1 (D-3): run a crew without going through chat - the swarm button.
        crew_box = ttk.LabelFrame(agents_tab, text="Run a crew", padding=6)
        crew_box.pack(fill="x", pady=(8, 0))
        ttk.Label(crew_box, text="Roles (comma-separated)", font=FONT_SMALL).pack(
            anchor="w"
        )
        self.crew_roles = ttk.Entry(crew_box)
        self.crew_roles.pack(fill="x")
        ttk.Label(crew_box, text="Process", font=FONT_SMALL).pack(
            anchor="w", pady=(4, 0)
        )
        self.crew_process = ttk.Combobox(
            crew_box,
            values=["sequential", "hierarchical"],
            state="readonly",
            width=16,
        )
        self.crew_process.set("sequential")
        self.crew_process.pack(anchor="w")
        ttk.Label(crew_box, text="Brief", font=FONT_SMALL).pack(anchor="w", pady=(4, 0))
        self.crew_brief = ttk.Entry(crew_box)
        self.crew_brief.pack(fill="x")
        ttk.Button(crew_box, text="Start crew", command=self._start_crew_ui).pack(
            anchor="e", pady=(6, 0)
        )
        self.crew_hint = tk.StringVar(value="roles come from the roster above")
        ttk.Label(
            agents_tab,
            textvariable=self.crew_hint,
            font=FONT_SMALL,
            wraplength=320,
            foreground="#6b7280",
        ).pack(anchor="w", pady=(4, 0))
        self.agents_summary = tk.StringVar(value="roster loads from the engine")
        ttk.Label(
            agents_tab,
            textvariable=self.agents_summary,
            font=FONT_SMALL,
            wraplength=320,
        ).pack(anchor="w", pady=(4, 0))

        background = ttk.Style().lookup("TFrame", "background") or "#f0f0f0"
        canvas = tk.Canvas(
            controls_tab,
            width=300,
            height=200,
            highlightthickness=0,
            borderwidth=0,
            background=background,
        )
        scrollbar = ttk.Scrollbar(controls_tab, orient="vertical", command=canvas.yview)
        canvas.configure(yscrollcommand=scrollbar.set)
        scrollbar.pack(side="right", fill="y")
        canvas.pack(side="left", fill="both", expand=True)

        left = ttk.Frame(canvas, padding=(6, 4))
        window_id = canvas.create_window((0, 0), window=left, anchor="nw")
        left.bind(
            "<Configure>",
            lambda _e: canvas.configure(scrollregion=canvas.bbox("all")),
        )
        canvas.bind(
            "<Configure>",
            lambda event: canvas.itemconfigure(window_id, width=event.width),
        )

        def scroll_controls(event: tk.Event) -> None:
            canvas.yview_scroll(int(-event.delta / 120), "units")

        canvas.bind("<MouseWheel>", scroll_controls, add="+")
        left.bind("<MouseWheel>", scroll_controls, add="+")
        self._build_controls(left)

        right = ttk.Frame(body, padding=(0, 6, 10, 0))
        right.pack(side="left", fill="both", expand=True)
        self.chat = ChatView(right)
        self.chat.pack(fill="both", expand=True)

        # Right pane: the workflow canvas, the subagent list, or the request mind map.
        # Hidden until asked for, so the chat keeps the room.
        self.panes = ttk.Frame(body, width=340)
        self.workflow = WorkflowView(self.panes, on_cancel=self._cancel_task)
        self.subagents = SubagentsPane(self.panes, on_cancel=self._cancel_task)
        self.mindmap = MindMapPane(self.panes)
        # M1.2 (D-5): the per-turn record + the A/B runner.
        self.thinking = ThinkingPane(self.panes, on_ab=self._run_ab)
        # S2/S3: the Doctor pane - who investigates, the repair catalogue, the
        # verifier, and the incident log. Its catalogue comes from doctor.py so
        # the pane can never offer a repair the engine does not have.
        self.doctor = DoctorPane(
            self.panes,
            on_diagnose=self._doctor_diagnose,
            on_repair=self._doctor_repair,
            on_verify=self._doctor_verify,
            on_settings=self._doctor_settings,
            on_watch=self._doctor_watch,
            on_fix_top=self._doctor_fix_top,
            on_escalate=self._doctor_escalate,
        )
        self.doctor.set_catalogue(
            [
                {"id": key, "title": entry.title, "action": entry.action}
                for key, entry in sorted(DOCTOR_REPAIRS.items())
            ],
            list(DOCTOR_SCOPES),
            list(DOCTOR_AGENTS),
        )
        self.view_stack = {
            "workflow": self.workflow,
            "subagents": self.subagents,
            "mindmap": self.mindmap,
            "thinking": self.thinking,
            "doctor": self.doctor,
        }
        self.panes_visible = False

    def _build_controls(self, parent: ttk.Frame) -> None:
        model_box = ttk.LabelFrame(parent, text="Language model", padding=8)
        model_box.pack(fill="x", pady=(0, 8))

        ttk.Label(model_box, text="Provider", font=FONT_SMALL).pack(anchor="w")
        self.provider_cb = ttk.Combobox(
            model_box,
            values=["github-copilot", "openrouter", "ollama"],
            state="readonly",
        )
        self.provider_cb.pack(fill="x")
        self.provider_cb.bind(
            "<<ComboboxSelected>>", lambda _e: self._on_provider_change()
        )

        ttk.Label(model_box, text="Model", font=FONT_SMALL).pack(
            anchor="w", pady=(6, 0)
        )
        self.model_cb = ttk.Combobox(model_box, values=[], state="readonly")
        self.model_cb.pack(fill="x")
        self.model_cb.bind(
            "<<ComboboxSelected>>",
            lambda _e: self._push({"model": self.model_cb.get()}),
        )

        ttk.Button(model_box, text="Refresh list", command=self._load_models).pack(
            anchor="e", pady=(6, 0)
        )

        ttk.Label(
            model_box, text="OpenRouter key (session only)", font=FONT_SMALL
        ).pack(anchor="w", pady=(8, 0))
        self.key_entry = ttk.Entry(model_box, show="\u2022")
        self.key_entry.pack(fill="x")
        self.key_entry.bind("<FocusOut>", lambda _e: self._on_key_entered())

        speech_box = ttk.LabelFrame(parent, text="Speech", padding=8)
        speech_box.pack(fill="x", pady=(0, 8))

        ttk.Label(speech_box, text="Recognition (offline)", font=FONT_SMALL).pack(
            anchor="w"
        )
        self.asr_cb = ttk.Combobox(speech_box, values=[], state="readonly")
        self.asr_cb.pack(fill="x")
        self.asr_cb.bind(
            "<<ComboboxSelected>>",
            lambda _e: self._push({"asr_backend": self.asr_cb.get()}),
        )

        ttk.Label(
            speech_box, text="Voice (SAPI or offline neural)", font=FONT_SMALL
        ).pack(anchor="w", pady=(6, 0))
        self.voice_cb = ttk.Combobox(speech_box, values=[], state="readonly")
        self.voice_cb.pack(fill="x")
        self.voice_cb.bind(
            "<<ComboboxSelected>>",
            lambda _e: self._push({"tts_voice": self.voice_cb.get()}),
        )

        self.rate_var = tk.IntVar(value=0)
        self.rate_label = ttk.Label(speech_box, text="Rate 0", font=FONT_SMALL)
        self.rate_label.pack(anchor="w", pady=(6, 0))
        self.rate_scale = ttk.Scale(
            speech_box,
            from_=-6,
            to=6,
            orient="horizontal",
            command=self._on_rate_change,
        )
        self.rate_scale.pack(fill="x")

        ttk.Label(
            speech_box, text="Pause tolerance (end of turn)", font=FONT_SMALL
        ).pack(anchor="w", pady=(6, 0))
        self.silence_label = ttk.Label(
            speech_box, text="0.80 s of silence ends a turn", font=FONT_SMALL
        )
        self.silence_label.pack(anchor="w")
        self.silence_scale = ttk.Scale(
            speech_box,
            from_=0.3,
            to=2.0,
            orient="horizontal",
            command=self._on_silence_change,
        )
        self.silence_scale.pack(fill="x")
        ttk.Label(
            speech_box,
            text="Too short and it answers before you finish a sentence.",
            font=FONT_SMALL,
            wraplength=290,
            foreground="#6b7280",
        ).pack(anchor="w", pady=(2, 0))

        options_box = ttk.LabelFrame(parent, text="Conversation", padding=8)
        options_box.pack(fill="x", pady=(0, 8))

        self.tts_var = tk.BooleanVar(value=True)
        self.barge_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(
            options_box,
            text="Speak replies",
            variable=self.tts_var,
            command=lambda: self._push({"tts_enabled": self.tts_var.get()}),
        ).pack(anchor="w")
        ttk.Checkbutton(
            options_box,
            text="Stop when I interrupt (barge-in)",
            variable=self.barge_var,
            command=lambda: self._push({"barge_in": self.barge_var.get()}),
        ).pack(anchor="w")
        # M2.2/M2.5: the streamed-voice switch and the smart-turn judge.
        self.stream_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(
            options_box,
            text="Speak while the model is still writing (streamed voice)",
            variable=self.stream_var,
            command=lambda: self._push({"stream_voice": self.stream_var.get()}),
        ).pack(anchor="w")
        self.smart_turn_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(
            options_box,
            text="Keep the turn open when a thought is unfinished",
            variable=self.smart_turn_var,
            command=lambda: self._push({"smart_turn": self.smart_turn_var.get()}),
        ).pack(anchor="w")
        # M3.3: the interrupt words, editable as data.
        ttk.Label(
            options_box, text="Interrupt words (comma-separated)", font=FONT_SMALL
        ).pack(anchor="w", pady=(6, 0))
        self.words_entry = ttk.Entry(options_box)
        self.words_entry.pack(fill="x")
        self.words_entry.bind("<FocusOut>", lambda _e: self._on_words_entered())
        self.words_entry.bind("<Return>", lambda _e: self._on_words_entered())

        self.rms_label = ttk.Label(
            options_box, text="Barge-in threshold 0.040", font=FONT_SMALL
        )
        self.rms_label.pack(anchor="w", pady=(6, 0))
        self.rms_scale = ttk.Scale(
            options_box,
            from_=MIN_BARGE_RMS,
            to=0.2,
            orient="horizontal",
            command=self._on_rms_change,
        )
        self.rms_scale.pack(fill="x")

        self.level_label = ttk.Label(options_box, text="mic level: -", font=FONT_SMALL)
        self.level_label.pack(anchor="w", pady=(4, 0))
        self.level_bar = ttk.Progressbar(options_box, maximum=0.2, length=280)
        self.level_bar.pack(fill="x")

        ttk.Label(
            options_box,
            text="Higher = needs a louder voice to interrupt. The bar shows the live mic level; "
            "if it fills while the agent is talking, its own speaker is being heard - "
            "raise the threshold or use headphones.",
            font=FONT_SMALL,
            wraplength=290,
            foreground="#6b7280",
        ).pack(anchor="w", pady=(4, 0))

        memory_box = ttk.LabelFrame(parent, text="Memory", padding=8)
        memory_box.pack(fill="x", pady=(0, 8))
        # Approval gate (#30/#21): risky actions queue here until answered.
        approve_box = ttk.LabelFrame(parent, text="Approvals", padding=8)
        approve_box.pack(fill="x", pady=(0, 8))
        self.approve_hint = tk.StringVar(value="no pending approvals")
        ttk.Label(
            approve_box, textvariable=self.approve_hint, font=FONT_SMALL, wraplength=290
        ).pack(anchor="w", pady=(0, 4))
        self._approve_buttons: list[ttk.Button] = []
        self.approve_box = approve_box

        ttk.Label(memory_box, text="Model memory after a reply", font=FONT_SMALL).pack(
            anchor="w"
        )
        self.memory_cb = ttk.Combobox(
            memory_box, values=list(MEMORY_CHOICES), state="readonly"
        )
        self.memory_cb.pack(fill="x")
        self.memory_cb.bind("<<ComboboxSelected>>", lambda _e: self._on_memory_change())
        ttk.Label(
            memory_box,
            text="Idle models are evicted from RAM automatically, and the app frees "
            "its own speech models after 5 minutes without use.",
            font=FONT_SMALL,
            wraplength=290,
            foreground="#6b7280",
        ).pack(anchor="w", pady=(4, 0))

        self.memory_label = ttk.Label(memory_box, text="app memory: -", font=FONT_SMALL)
        self.memory_label.pack(anchor="w", pady=(6, 0))
        self.loaded_label = ttk.Label(
            memory_box, text="models in RAM: -", font=FONT_SMALL
        )
        self.loaded_label.pack(anchor="w")
        # #62: read-only MEMORY.md view with per-fact Forget buttons.
        ttk.Button(memory_box, text="Memory inspector", command=self._open_memory).pack(
            anchor="w", pady=(4, 0)
        )
        ttk.Button(
            memory_box, text="Free memory now", command=self._on_free_memory
        ).pack(anchor="e", pady=(6, 0))

        device_box = ttk.LabelFrame(parent, text="Devices", padding=8)
        device_box.pack(fill="x")
        self.device_var = tk.StringVar(value="-")
        ttk.Label(
            device_box, textvariable=self.device_var, font=FONT_SMALL, wraplength=290
        ).pack(anchor="w")

    def _build_bottom(self) -> None:
        # Packed before the transcript and anchored to the bottom, so these rows
        # are always on screen even when the display is short.
        bottom = ttk.Frame(self.root, padding=(10, 8))
        bottom.pack(side="bottom", fill="x")

        self.entry = ttk.Entry(bottom, font=FONT_UI)
        self.entry.pack(side="left", fill="x", expand=True)
        self.entry.bind("<Return>", lambda _e: self._send_typed())
        ttk.Button(bottom, text="Send", command=self._send_typed).pack(
            side="left", padx=(8, 0)
        )

        row = ttk.Frame(self.root, padding=(10, 0, 10, 10))
        row.pack(side="bottom", fill="x")
        self.ptt_button = ttk.Button(row, text="Hold to talk")
        self.ptt_button.pack(side="left")
        self.ptt_button.bind("<ButtonPress-1>", self._ptt_start)
        self.ptt_button.bind("<ButtonRelease-1>", self._ptt_stop)
        self.ptt_button.bind("<Leave>", lambda _e: self._ptt_stop(None))

        self.listen_button = ttk.Button(
            row, text="Listen continuously", command=self._toggle_listen
        )
        self.listen_button.pack(side="left", padx=(8, 0))
        ttk.Button(row, text="Interrupt now", command=self.session.interrupt).pack(
            side="left", padx=(8, 0)
        )

        self.error_label = ttk.Label(
            row, text="", font=FONT_SMALL, foreground="#b91c1c"
        )
        self.error_label.pack(side="left", padx=(12, 0))

    # ------------------------------------------------------------------ actions
    def _push(self, patch: dict[str, Any]) -> None:
        self.session.update_settings(patch)
        self._apply_settings_to_widgets()

    def _on_provider_change(self) -> None:
        """Switching provider must also switch to one of THAT provider's models.

        Committing the provider first is the whole point: loading the model list
        before the switch pairs the new provider with the old provider's model
        name, and Ollama answers an OpenRouter id with HTTP 400.
        """
        provider = self.provider_cb.get()
        if provider == "openrouter" and not self.session.settings.openrouter_key:
            self._set_error(
                "OpenRouter needs an API key - paste one in the box, or stay on ollama"
            )
            self.provider_cb.set(self.session.settings.provider)
            return

        self._push({"provider": provider})
        models = self._load_models()
        if not models:
            self._set_error(f"no {provider} models available")
            return
        self._push({"model": models[0]})

    def _on_memory_change(self) -> None:
        choice = self.memory_cb.get()
        self._push({"model_memory": MEMORY_CHOICES.get(choice, "5m")})

    def _on_free_memory(self) -> None:
        freed = self.session.release_idle_models(force=True)
        self._set_error("freed: " + ", ".join(freed) if freed else "nothing was loaded")

    def _validate_model(self) -> None:
        """Repair a stored model name that belongs to the other provider."""
        settings = self.session.settings
        available = (
            self.session.available_models().get(settings.provider, []) or self.models
        )
        if available and settings.model not in available:
            fallback = available[0]
            self._set_error(
                f"'{settings.model}' is not a {settings.provider} model - using '{fallback}'"
            )
            self._push({"model": fallback})

    def _on_key_entered(self) -> None:
        key = self.key_entry.get().strip()
        if key:
            self._push({"openrouter_key": key})

    def _on_rate_change(self, value: str) -> None:
        rate = int(_as_number(value, 0))
        self.rate_label.configure(text=f"Rate {rate:+d}")
        if not self._applying:
            self._push({"tts_rate": rate})

    def _on_rms_change(self, value: str) -> None:
        threshold = round(max(MIN_BARGE_RMS, _as_number(value, 0.06)), 3)
        self.rms_label.configure(
            text=f"Interrupt above {threshold:.3f} - {_threshold_words(threshold)}"
        )
        if not self._applying:
            self._push({"barge_rms": threshold})

    def _on_silence_change(self, value: str) -> None:
        silence = round(_as_number(value, 0.8), 2)
        self.silence_label.configure(text=f"{silence:.2f} s of silence ends a turn")
        if not self._applying:
            self._push({"vad_min_silence": silence})

    def _send_typed(self) -> None:
        text = self.entry.get().strip()
        if not text:
            return
        self.entry.delete(0, "end")
        self.session.submit_text(text)

    def _ptt_start(self, _event: tk.Event) -> None:
        self.session.start_listening("ptt")

    def _ptt_stop(self, _event: tk.Event | None) -> None:
        if self.session.mode == "ptt":
            self.session.stop_listening()

    def _toggle_listen(self) -> None:
        if self.session.mode == "off":
            self.session.start_listening("continuous")
        else:
            self.session.stop_listening()

    def _new_chat(self) -> None:
        self.session.reset()
        self.rendered.clear()
        self.chat.clear()
        self._write_welcome()

    def _refresh_panes(self, state: dict[str, Any]) -> None:
        """Refresh the panes lazily: history every couple of seconds, the file tree rarely."""
        now = time.time()
        agents = state.get("agents") or {}
        if agents:
            self.agents_summary.set(
                "\n".join(
                    f"{role}: {(cfg or {}).get('model') or '-'!s}"
                    for role, cfg in agents.items()
                )
            )
        if now - getattr(self, "_history_at", 0.0) > 2.0:
            self._history_at = now
            try:
                self.history.refresh(self.session.list_sessions())
            except Exception as exc:  # noqa: BLE001
                self.history.set_hint(f"history unavailable: {exc}")
        if now - getattr(self, "_files_at", 0.0) > 15.0:
            self._files_at = now
            try:
                self.explorer.refresh()
            except Exception as exc:  # noqa: BLE001
                self.explorer.status.set(f"file tree unavailable: {exc}")
        # M1.2: the Thinking pane reads the same recorded state as everything else.
        # A pane must never kill the poll, so failures here are swallowed by design
        # and recorded in the status line instead.
        with contextlib.suppress(Exception):
            self.thinking.refresh(
                list(state.get("turns") or []),
                list(state.get("tasks") or []),
                sorted(str(r) for r in (agents or {})),
            )
        # S1/S2/S4: the Doctor pane reads the incident log, the repair state and
        # the live agent strip from the same state as everything else.
        with contextlib.suppress(Exception):
            self.doctor.refresh(
                list(state.get("incidents") or []),
                list(state.get("agents_active") or []),
                state.get("doctor") or {},
                int(state.get("incident_count") or 0),
            )

    def _speaker_text(self, state: dict[str, Any]) -> str:
        """Who is speaking right now, in one word."""
        if state.get("status") == "speaking":
            return "assistant speaking"
        if state.get("recording"):
            return "you (mic)"
        if state.get("status") == "thinking":
            return "thinking"
        if state.get("mode") in {"continuous", "ptt"}:
            return "listening"
        return "idle"

    # ------------------------------------------------------------------ doctor
    def _doctor_diagnose(self, scope: str, agent: str) -> dict[str, Any]:
        """S2: read-only pass; nothing changes until a Repair is pressed."""
        return self.session.diagnose(scope, agent)

    def _doctor_repair(self, repair_id: str, dry_run: bool) -> dict[str, Any]:
        """S3: one catalogue entry; the engine verifies and rolls back on failure."""
        return self.session.repair(repair_id, dry_run)

    def _doctor_verify(self) -> dict[str, Any]:
        """S3: the verifier on its own, independent of any repair."""
        return self.session.verify()

    def _doctor_settings(self, enabled: bool, kill_switch: bool) -> dict[str, Any]:
        """S2: the doctor's own switches, persisted with the rest of the config."""
        self.session.update_settings(
            {"doctor_enabled": bool(enabled), "doctor_kill_switch": bool(kill_switch)}
        )
        return {"ok": True}

    def _doctor_watch(self, on: bool) -> dict[str, Any]:
        """S4.1: demo mode - watch in the background, detect only."""
        self.session.update_settings({"doctor_watch": bool(on)})
        return {"ok": True}

    def _doctor_fix_top(self) -> dict[str, Any]:
        """S4.1: one press repairs the worst finding with a safe repair."""
        return self.session.fix_top_finding(False)

    def _doctor_escalate(self, question: str) -> dict[str, Any]:
        """S3.5: propose a code-level fix; it stays behind the approval gate."""
        return self.session.escalate(question, evidence="from the Doctor pane")

    @staticmethod
    def _agents_line(rows: list[dict[str, Any]]) -> str:
        """S4: one line naming every agent and whether it is busy, always on screen."""
        if not rows:
            return "agents: no report yet"
        quiet = {"idle", "ready", "local-only"}
        busy = [
            f"{row.get('role')}:{row.get('state')}"
            for row in rows
            if str(row.get("state")) not in quiet
        ]
        rest = [str(row.get("role")) for row in rows if str(row.get("state")) in quiet]
        head = " · ".join(busy) if busy else "nothing running"
        tail = f"   |   idle: {', '.join(rest)}" if rest else ""
        return f"agents active: {head}{tail}"

    def report_callback_exception(
        self,
        exc_type: type[BaseException],
        exc_value: BaseException,
        exc_tb: types.TracebackType | None = None,
    ) -> None:
        """S1.2: Tk callback crashes were printed to a console nobody watches.

        That is why the owner's runtime error left no trace. This records it,
        shows it in the window, and keeps the app alive.
        """
        text = "".join(traceback.format_exception(exc_type, exc_value, exc_tb))
        print(text, file=sys.stderr)
        with contextlib.suppress(Exception):
            self.session.incidents.record(
                "ui",
                f"{exc_type.__name__}: {exc_value}",
                severity="error",
                where="app_gui",
                context={"traceback": text[-1200:]},
            )
        with contextlib.suppress(Exception):
            self._set_error(f"{exc_type.__name__}: {exc_value}")

    def _open_agents(self) -> None:
        """Open the roster: which model plays each role, and the session spend cap."""
        try:
            AgentsDialog(self.root, self.session)
        except Exception as exc:  # noqa: BLE001 - the UI must not crash on a dialog
            self._set_error(f"could not open the agents panel: {exc}")

    def _show_view(self) -> None:
        """Show the chosen right-hand view: workflow, subagents or mind map."""
        chosen = getattr(self, "view_stack", {}).get(self.view_var.get())
        if chosen is None:
            return
        if not self.panes_visible:
            self.panes.pack(side="right", fill="y", padx=(6, 0))
            self.panes_visible = True
        for view in self.view_stack.values():
            view.pack_forget()
        chosen.pack(fill="both", expand=True)

    def _open_session(self, session_id: str) -> None:
        """Replay a past conversation into the chat view."""
        try:
            turns = self.session.load_session(session_id)
        except Exception as exc:  # noqa: BLE001 - a bad session must not crash the UI
            self.history.set_hint(f"could not load {session_id}: {exc}")
            return
        if not turns:
            self.history.set_hint(f"{session_id} has no messages")
            return
        self.chat.clear()
        self.rendered.clear()
        for turn in turns:
            self.chat.add_turn(turn)
            # Only ints go in the rendered set; a malformed session file must not crash the UI.
            turn_id = turn.get("id")
            if isinstance(turn_id, int):
                self.rendered.add(turn_id)
        self.history.set_hint(f"showing {len(turns)} turns from {session_id}")

    def _open_file(self, path: str) -> None:
        """Ask the assistant to read a file the user picked, so citations follow."""
        self.chat.add_note(f"Reading {path}")
        try:
            self.session.submit_text(f"Read {path} and summarise what matters in it.")
        except Exception as exc:  # noqa: BLE001
            self._set_error(f"could not ask about that file: {exc}")

    def _cancel_task(self, task_id: str) -> None:
        """Cancel one background task from the canvas."""
        try:
            if not self.session.cancel_task(task_id):
                self._set_error(f"task {task_id} was already finished")
        except Exception as exc:  # noqa: BLE001 - the UI must not crash on this
            self._set_error(f"could not cancel {task_id}: {exc}")

    def _run_ab(self, text: str, role_a: str, role_b: str) -> dict[str, Any]:
        """M1.4: the same question to two roles - thinking panes side by side."""
        try:
            result = self.session.ab_compare(text, role_a, role_b)
        except Exception as exc:  # noqa: BLE001 - the UI must not crash
            return {"ok": False, "error": str(exc)}
        if result.get("ok"):
            self.view_var.set("thinking")
            self._show_view()
        return result

    def _start_crew_ui(self) -> None:
        """M3.1: start a crew from the Agents tab."""
        roles = [r.strip() for r in self.crew_roles.get().split(",") if r.strip()]
        brief = self.crew_brief.get().strip()
        if not roles:
            # Default to the roster's typical crew when the field is empty.
            known = self.session.roles_config()
            roles = [r for r in ("researcher", "critic", "summarizer") if r in known]
        if not brief:
            self.crew_hint.set("type a brief first")
            return
        try:
            message = self.session.start_crew(
                brief, roles=roles, process=self.crew_process.get() or "sequential"
            )
        except Exception as exc:  # noqa: BLE001
            self.crew_hint.set(f"crew failed to start: {exc}")
            return
        self.crew_hint.set(message[:200])
        self.view_var.set("workflow")
        self._show_view()

    def _on_words_entered(self) -> None:
        """M3.3: the interrupt words are settings data, written back as a list."""
        words = [w.strip() for w in self.words_entry.get().split(",") if w.strip()]
        if words:
            self._push({"barge_triggers": words})

    def _resume_session(self, session_id: str) -> None:
        """#26 callback: make the past session current; new turns append there."""
        try:
            result = self.session.resume_session(session_id)
        except Exception as exc:  # noqa: BLE001 - the UI must not crash
            self.history.set_hint(f"could not resume {session_id}: {exc}")
            return
        self.history.set_hint("resumed " + result + " - new turns append there")

    def _fork_session(self, session_id: str) -> None:
        """#9 callback: same past, independent future."""
        try:
            new_id = self.session.fork_session(session_id)
        except Exception as exc:  # noqa: BLE001
            self.history.set_hint(f"could not fork {session_id}: {exc}")
            return
        if new_id.startswith("no such"):
            self.history.set_hint(new_id)
            return
        self.history.set_hint(f"forked to {new_id}")

    def _export_markdown(self) -> None:
        """#61: write the current transcript to a Markdown file the owner can read."""
        from pathlib import Path as _Path

        try:
            rows = list(getattr(self.session, "turns", []))
        except Exception:  # noqa: BLE001
            rows = []
        if not rows:
            self.history.set_hint("nothing to export yet")
            return
        out = _Path.cwd() / "SONJ_transcript.md"
        lines = ["# SONJ transcript", ""]
        for row in rows:
            stamp = str(row.get("ts", ""))
            lines.append(f"- **{stamp}** you: {str(row.get('user_text', ''))[:200]}")
            lines.append(f"  - nova: {str(row.get('assistant_text', ''))[:200]}")
            for tool in row.get("tools") or []:
                name = tool.get("name") if isinstance(tool, dict) else str(tool)
                lines.append(f"    - tool: {name}")
        out.write_text(chr(10).join(lines), encoding="utf-8")
        self.history.set_hint(f"exported {len(rows)} turns to {out.name}")

    def _render_approvals(self, approvals: list[dict]) -> None:
        """#30: draw the pending queue with Approve/Deny buttons."""
        for button in self._approve_buttons:
            button.destroy()
        self._approve_buttons.clear()
        if not approvals:
            self.approve_hint.set("no pending approvals")
            return
        self.approve_hint.set(f"{len(approvals)} awaiting your approval")
        for entry in approvals[:4]:
            row = ttk.Frame(self.approve_box)
            row.pack(fill="x", pady=(2, 0))
            label = f"#{entry.get('id')} {str(entry.get('action', ''))[:26]} - {str(entry.get('detail', ''))[:40]}"
            ttk.Label(row, text=label, font=FONT_SMALL, wraplength=250).pack(
                side="left"
            )
            approve = ttk.Button(
                row,
                text="Approve",
                command=lambda e=entry: self._answer_approval(e, True),
            )
            approve.pack(side="left", padx=(4, 0))
            deny = ttk.Button(
                row,
                text="Deny",
                command=lambda e=entry: self._answer_approval(e, False),
            )
            deny.pack(side="left", padx=(2, 0))
            self._approve_buttons.extend((approve, deny))
        if len(approvals) > 4:
            ttk.Label(
                self.approve_box,
                text=f"... and {len(approvals) - 4} more",
                font=FONT_SMALL,
            ).pack(anchor="w")

    def _refresh_free_models(self) -> None:
        """W1.3: pull OpenRouter's free models and fill every role dropdown."""
        try:
            raw = self.session.fetch_free_models()
        except Exception as exc:  # noqa: BLE001 - the UI must not crash
            self.agents_summary.set(f"free-model list failed: {exc}")
            return
        if not raw.get("ok"):
            self.agents_summary.set(
                "free-model list failed: " + str(raw.get("error", ""))[:80]
            )
            return
        models = [str(m) for m in raw.get("models", [])]
        self._free_models = models
        self._rebuild_role_rows()
        self.agents_summary.set(f"{len(models)} free models loaded - pick per role")

    def _rebuild_role_rows(self) -> None:
        """W1.2: one row per role: name, model dropdown, fallbacks label."""
        roles = self.session.roles_config()
        if not roles:
            self.agents_summary.set("no roles in agents.json")
            return
        for child in self._role_frame.winfo_children():
            child.destroy()
        self._role_rows.clear()
        choices = list(getattr(self, "_free_models", [])) or [""]
        for name in sorted(roles):
            cfg = roles[name] if isinstance(roles.get(name), dict) else {}
            row = ttk.Frame(self._role_frame)
            row.pack(fill="x", pady=(2, 0))
            ttk.Label(row, text=name, width=11, font=FONT_SMALL).pack(side="left")
            combo = ttk.Combobox(row, values=choices, width=34, state="readonly")
            combo.set(str(cfg.get("model", ""))[:34])
            combo.pack(side="left", fill="x", expand=True)
            combo.bind(
                "<<ComboboxSelected>>",
                lambda _e, n=name, c=combo: self._on_role_model_change(n, c.get()),
            )
            fallbacks = ", ".join(str(f) for f in (cfg.get("fallbacks") or [])) or "-"
            ttk.Label(
                self._role_frame,
                text=f"    fallbacks: {fallbacks[:70]}",
                font=FONT_SMALL,
                foreground="#6b7280",
            ).pack(anchor="w")
            self._role_rows[name] = combo

    def _on_role_model_change(self, role: str, model: str) -> None:
        """W1.2: write the role's new model to agents.json; the next turn uses it."""
        try:
            message = self.session.set_role_model(role, model)
            self.agents_summary.set(message)
        except Exception as exc:  # noqa: BLE001
            self.agents_summary.set(f"could not set {role}: {exc}")

    def _open_memory(self) -> None:
        """#62: read-only MEMORY.md view; per-fact Forget uses the agent's tool."""
        try:
            facts = self.session.memory_facts()
        except Exception as exc:  # noqa: BLE001
            self._set_error(f"could not read memory: {exc}")
            return
        win = tk.Toplevel(self.root)
        win.title("SONJ memory")
        win.geometry("600x420")
        body = ttk.Frame(win, padding=8)
        body.pack(fill="both", expand=True)
        if not facts:
            ttk.Label(body, text="memory is empty", font=FONT_SMALL).pack(anchor="w")
            return
        for row in facts[:40]:
            line = ttk.Frame(body)
            line.pack(fill="x", pady=1)
            ttk.Label(
                line,
                text=f"[{row['stamp']}] {row['fact'][:58]}",
                font=FONT_SMALL,
                wraplength=400,
            ).pack(side="left")
            fact = row["fact"]
            ttk.Button(
                line,
                text="Forget",
                width=7,
                command=lambda f=fact: self._forget_fact(win, f),
            ).pack(side="right")
        if len(facts) > 40:
            ttk.Label(
                body, text=f"... and {len(facts) - 40} more", font=FONT_SMALL
            ).pack(anchor="w")

    def _forget_fact(self, win: tk.Toplevel, fact: str) -> None:
        """Forget one fact, then refresh the inspector."""
        try:
            message = self.session.forget_fact(fact)
        except Exception as exc:  # noqa: BLE001
            self._set_error(f"could not forget: {exc}")
            return
        self._set_error(message[:120])
        win.destroy()
        self._open_memory()

    def _answer_approval(self, entry: dict, approved: bool) -> None:
        """The owner's answer goes back to the queue."""
        try:
            self.session.respond_approval(str(entry.get("id", "")), approved)
        except Exception as exc:  # noqa: BLE001
            self._set_error(f"could not answer approval: {exc}")

    def _load_models(self) -> list[str]:
        """Refresh the dropdown for the provider that is committed right now."""
        models = self.session.available_models()
        provider = self.session.settings.provider
        self.models = models.get(provider, []) or []
        self.model_cb.configure(values=self.models)
        if self.models and self.model_cb.get() not in self.models:
            self.model_cb.set(self.models[0])
        return self.models

    def _load_voices(self) -> None:
        try:
            self.voices = list_all_voices()
        except TtsError as exc:
            self.voices = []
            self._set_error(f"voices unavailable: {exc}")
        self.voice_cb.configure(values=self.voices)

    def _apply_settings_to_widgets(self) -> None:
        settings: Settings = self.session.settings
        # Widget callbacks write straight back to settings, so guard the
        # programmatic updates here or moving a slider would recurse.
        self._applying = True
        try:
            self._apply_inner(settings)
        finally:
            self._applying = False

    def _apply_inner(self, settings: Settings) -> None:
        self.provider_cb.set(settings.provider)
        if settings.model and (not self.models or settings.model in self.models):
            self.model_cb.set(settings.model)
        self.asr_cb.configure(
            values=sorted(discover_models(MODELS_DIR)) or ["moonshine", "whisper"]
        )
        self.asr_cb.set(settings.asr_backend)
        if settings.tts_voice:
            self.voice_cb.set(settings.tts_voice)
        elif self.voices:
            self.voice_cb.set(self.voices[0])
        self.tts_var.set(bool(settings.tts_enabled))
        self.barge_var.set(bool(settings.barge_in))
        with contextlib.suppress(AttributeError):  # built after the first poll
            self.stream_var.set(bool(settings.stream_voice))
            self.smart_turn_var.set(bool(settings.smart_turn))
            words = ", ".join(str(w) for w in (settings.barge_triggers or []))
            if self.words_entry.get().strip() != words:
                self.words_entry.delete(0, "end")
                self.words_entry.insert(0, words)
        rate = _as_number(settings.tts_rate, 0)
        threshold = max(MIN_BARGE_RMS, _as_number(settings.barge_rms, 0.06))
        silence = _as_number(getattr(settings, "vad_min_silence", 0.8), 0.8)

        self.rate_label.configure(text=f"Rate {int(rate):+d}")
        self.rate_scale.set(rate)
        self.rms_label.configure(
            text=f"Interrupt above {threshold:.3f} - {_threshold_words(threshold)}"
        )
        self.rms_scale.set(threshold)
        self.silence_label.configure(text=f"{silence:.2f} s of silence ends a turn")
        self.silence_scale.set(silence)
        stored = str(getattr(settings, "model_memory", "5m"))
        for label, value in MEMORY_CHOICES.items():
            if value == stored:
                self.memory_cb.set(label)
                break
        self.model_var.set(f"{settings.provider}/{settings.model}")

    def _set_error(self, message: str) -> None:
        self.error_label.configure(text=message[:120])

    # ------------------------------------------------------------------ render
    def _write_welcome(self) -> None:
        self.chat.add_note(
            "Ready. Hold 'Hold to talk' and speak, or type below and press Enter. "
            "'Listen continuously' watches for when you stop speaking. Talk over a "
            "reply to interrupt it."
        )

    def _append_turn(self, turn: dict[str, Any]) -> None:
        self.chat.add_turn(turn)

    def _poll(self) -> None:
        try:
            state = self.session.state()
            self.status_var.set(state["status"])
            recording = bool(state.get("recording"))
            self.indicator.update_state(str(state["status"]), recording)
            self.root.title(
                ("RECORDING - " if recording else "")
                + "ME3 Voice Agent - laptop dev build"
            )
            self.mode_var.set(
                "mic off" if state["mode"] == "off" else f"mic {state['mode']}"
            )
            self.listen_button.configure(
                text="Listen continuously"
                if state["mode"] == "off"
                else "Stop listening"
            )
            devices = state["devices"]
            self.device_var.set(
                f"in: {devices['input_name']}\nout: {devices['output_name']}"
            )
            level = _as_number(state.get("level", 0.0), 0.0)
            self.level_bar["value"] = min(level, 0.2)
            self.level_label.configure(text=f"mic level: {level:.3f}")

            memory = state.get("memory_mb")
            self.memory_label.configure(
                text=f"app memory: {memory:.0f} MB"
                if isinstance(memory, (int, float))
                else "app memory: -"
            )
            loaded = state.get("loaded_models") or []
            self.loaded_label.configure(
                text="models in RAM: " + (", ".join(loaded) if loaded else "none")
            )
            self._set_error(state.get("error", "") or "")
            self.agents_var.set(
                self._agents_line(list(state.get("agents_active") or []))
            )

            self.workflow.update_view(
                state.get("workflow") or {"nodes": [], "edges": []}
            )
            self.subagents.update_view(
                state.get("tasks") or [], state.get("workflow") or {}
            )
            self.mindmap.update_map(self.session.request_map())
            self._refresh_panes(state)
            self.workflow.update_spend(
                int(state.get("spend_tokens", 0)),
                _as_number(state.get("spend_usd", 0.0), 0.0),
            )

            fresh = [t for t in state["turns"] if t["id"] not in self.rendered]
            for turn in fresh:
                self._append_turn(turn)
                self.rendered.add(turn["id"])
        except Exception as exc:  # noqa: BLE001 - the UI poll must keep running
            self._set_error(f"{type(exc).__name__}: {exc}")
        finally:
            self.root.after(POLL_MS, self._poll)


def _demo_turns() -> list[dict[str, Any]]:
    """Sample turns for --demo: shows the chat layout without using a microphone."""
    return [
        {
            "id": 1,
            "ts": "14:02:11",
            "source": "mic",
            "user_text": "what time is it right now?",
            "assistant_text": "It is just after two in the afternoon.",
            "tools": [
                {
                    "name": "get_time",
                    "arguments": {"timezone": "local"},
                    "result": "Monday 02 November 2026, 14:02",
                    "ok": True,
                    "seconds": 0.0,
                }
            ],
            "timings": {
                "asr_ms": 21.0,
                "llm_ms": 5210.0,
                "tts_ms": 1840.0,
                "total_ms": 7100.0,
            },
            "model": "ollama/rafw007/qwen35-claude-coder:9b",
        },
        {
            "id": 2,
            "ts": "14:02:39",
            "source": "mic",
            "user_text": "search the web for the current Raspberry Pi 5 price",
            "assistant_text": "A Raspberry Pi 5 is around 80 dollars for the 8 GB model right now.",
            "tools": [
                {
                    "name": "web_search",
                    "arguments": {"query": "Raspberry Pi 5 price", "recency": "week"},
                    "result": "3 result(s) from ddgs. 1. Raspberry Pi 5 Price Tracker 2026 ...",
                    "ok": True,
                    "seconds": 1.4,
                }
            ],
            "timings": {
                "asr_ms": 188.0,
                "llm_ms": 9400.0,
                "tts_ms": 2600.0,
                "total_ms": 12300.0,
            },
            "model": "ollama/rafw007/qwen35-claude-coder:9b",
        },
    ]


def self_check() -> int:
    """Build and render the window without showing it (used by the test run)."""
    settings = load_settings()
    session = Session(settings)
    root = tk.Tk()
    root.withdraw()
    app = VoiceAgentApp(root, session)
    app._write_welcome()

    sample_turn = {
        "id": 999,
        "ts": "00:00:00",
        "source": "text",
        "user_text": "self check turn",
        "assistant_text": "self check reply",
        "tools": [
            {
                "name": "get_time",
                "arguments": {},
                "result": "ok",
                "ok": True,
                "seconds": 0.0,
            }
        ],
        "timings": {"asr_ms": 1.0, "llm_ms": 2.0, "tts_ms": 3.0, "total_ms": 6.0},
        "model": "selfcheck",
    }
    app._append_turn(sample_turn)
    root.update()

    body = app.chat.dump_text()
    problems: list[str] = []
    if "self check turn" not in body or "self check reply" not in body:
        problems.append("transcript did not render the turn")
    if "get_time" not in body:
        problems.append("tool card did not render")
    if app.asr_cb.get() not in {"moonshine", "whisper"}:
        problems.append(f"unexpected asr backend {app.asr_cb.get()!r}")
    if app.models and not app.model_cb.get():
        problems.append("model combobox is empty")

    print(f"provider : {app.provider_cb.get()}")
    print(f"models   : {len(app.models)} (selected: {app.model_cb.get()})")
    print(f"asr      : {app.asr_cb.get()}")
    print(f"voices   : {len(app.voices)} (selected: {app.voice_cb.get()})")
    print(f"devices  : {app.device_var.get()!r}")
    print(f"error    : {app.error_label.cget('text')!r}")
    print(f"rendered : {len(body.splitlines())} lines in the transcript")

    session.stop_listening()
    root.destroy()
    if problems:
        print("RESULT: FAIL -", "; ".join(problems))
        return 1
    print("RESULT: PASS")
    return 0


def main() -> int:
    if "--selftest" in sys.argv:
        return self_check()

    # S0: one engine, one voice. The desktop app owns the same lock the headless
    # engine uses, so a second window (or a stray server.py) refuses to start
    # instead of opening a second audio stack that fights for the speakers.
    ok, why = claim_engine(force="--force" in sys.argv)
    if not ok:
        print(f"[app] {why}")
        print("[app] close the other window, or start with --force to replace it")
        return 2

    settings = load_settings()
    session = Session(settings)
    root = tk.Tk()
    ttk.Style().theme_use(
        "vista" if "vista" in ttk.Style().theme_names() else "default"
    )
    app = VoiceAgentApp(root, session)
    app._write_welcome()

    if "--demo" in sys.argv:
        for turn in _demo_turns():
            app.chat.add_turn(turn)
        app.indicator.update_state("listening", True)

    def shutdown() -> None:
        session.stop_listening()
        # S4.2: what was repaired and what still needs a person, before the
        # folder is closed for the session.
        with contextlib.suppress(Exception):
            session.write_incident_report()
        session.save_config()
        release_engine()
        root.destroy()

    root.protocol("WM_DELETE_WINDOW", shutdown)
    root.mainloop()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
