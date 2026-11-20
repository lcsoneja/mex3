"""The agent roster: which model plays each role, changed without touching code.

The architecture note requires routing to be data. This dialog edits agents.json, refreshes the
running task layer, and hot-swaps the conversation model, so a session can move between a free
model and a paid one while it is running.
"""

from __future__ import annotations

import time
import tkinter as tk
from tkinter import ttk
from typing import Any

ROLES: tuple[tuple[str, str], ...] = (
    ("front", "Conversation"),
    ("back", "Background tasks"),
    ("router", "Router (local)"),
)


class AgentsDialog(tk.Toplevel):
    """Edit the roster and the session spend cap."""

    def __init__(self, master: tk.Tk, session: Any) -> None:
        super().__init__(master)
        self.session = session
        self.title("Agents - model roster")
        self.transient(master)
        self.resizable(False, False)
        try:
            state = session.state() or {}
        except Exception:  # noqa: BLE001 - a dialog must open even if state is unavailable
            state = {}
        self.roster = state.get("agents") or {}
        models = self._model_choices()
        self.rows: dict[str, tuple[ttk.Combobox, ttk.Entry]] = {}

        body = ttk.Frame(self, padding=12)
        body.pack(fill="both", expand=True)
        ttk.Label(body, text="Model per role", font=("Segoe UI", 10, "bold")).grid(
            row=0, column=0, columnspan=3, sticky="w"
        )
        ttk.Label(
            body,
            text="Routing is data (agents.json). Save hot-swaps the conversation model.",
            font=("Segoe UI", 8),
        ).grid(row=1, column=0, columnspan=3, sticky="w", pady=(0, 8))

        for index, (role, label) in enumerate(ROLES, start=2):
            ttk.Label(body, text=label).grid(row=index, column=0, sticky="w", pady=3)
            model_box = ttk.Combobox(body, values=models, width=38)
            model_box.set(str(self.roster.get(role, {}).get("model", "")))
            model_box.grid(row=index, column=1, sticky="we", padx=(8, 8))
            fallbacks = ttk.Entry(body, width=30)
            fallbacks.insert(0, ", ".join(self.roster.get(role, {}).get("fallbacks", []) or []))
            fallbacks.grid(row=index, column=2, sticky="we")
            self.rows[role] = (model_box, fallbacks)

        ttk.Label(body, text="Fallbacks (comma separated)", font=("Segoe UI", 8)).grid(
            row=len(ROLES) + 2, column=2, sticky="w"
        )

        limits = ttk.Frame(body)
        limits.grid(row=len(ROLES) + 3, column=0, columnspan=3, sticky="we", pady=(10, 0))
        back = self.roster.get("back") or {}
        self.parallel = tk.IntVar(value=int(back.get("max_parallel") or 3))
        self.steps = tk.IntVar(value=int(back.get("max_steps") or 12))
        try:
            cap_now = float(getattr(getattr(session, "settings", None), "session_cost_cap", 0.0) or 0.0)
        except (TypeError, ValueError):
            cap_now = 0.0
        self.cap = tk.DoubleVar(value=cap_now)
        ttk.Label(limits, text="Background tasks at once").grid(row=0, column=0, sticky="w")
        ttk.Spinbox(limits, from_=1, to=8, textvariable=self.parallel, width=4).grid(
            row=0, column=1, sticky="w", padx=(8, 18)
        )
        ttk.Label(limits, text="Max steps").grid(row=0, column=2, sticky="w")
        ttk.Spinbox(limits, from_=1, to=40, textvariable=self.steps, width=4).grid(
            row=0, column=3, sticky="w", padx=(8, 18)
        )
        ttk.Label(limits, text="Spend cap (USD)").grid(row=0, column=4, sticky="w")
        ttk.Spinbox(limits, from_=0.0, to=50.0, increment=0.25, textvariable=self.cap,
                    width=6).grid(row=0, column=5, sticky="w", padx=(8, 0))

        self.status = tk.StringVar(value="")
        ttk.Label(body, textvariable=self.status, font=("Segoe UI", 8),
                  wraplength=520).grid(row=len(ROLES) + 4, column=0, columnspan=3,
                                       sticky="w", pady=(10, 0))
        buttons = ttk.Frame(body)
        buttons.grid(row=len(ROLES) + 5, column=0, columnspan=3, sticky="e", pady=(10, 0))
        ttk.Button(buttons, text="Cancel", command=self.destroy).pack(side="right")
        ttk.Button(buttons, text="Test turn", command=self._test).pack(side="right", padx=(0, 6))
        ttk.Button(buttons, text="Save", command=self._save).pack(side="right", padx=(0, 6))
        self.bind("<Escape>", lambda _event: self.destroy())

    # ------------------------------------------------------------------ helpers
    def _model_choices(self) -> list[str]:
        """Every model id the app knows about, so the roster can be changed by name."""
        try:
            catalog = self.session.available_models()
        except Exception:  # noqa: BLE001 - a missing catalogue must not break the dialog
            catalog = {}
        choices: list[str] = []
        for values in (catalog or {}).values():
            for item in values or []:
                if item not in choices:
                    choices.append(str(item))
        for role in self.roster.values():
            model = str((role or {}).get("model") or "")
            if model and model not in choices:
                choices.append(model)
        return choices

    def _collect(self) -> dict[str, Any]:
        config: dict[str, Any] = {}
        for role, (model_box, fallbacks) in self.rows.items():
            model = model_box.get().strip()
            chain = [part.strip() for part in fallbacks.get().split(",") if part.strip()]
            if model or chain:
                config[role] = {"model": model, "fallbacks": chain}
        for role in ROLES:
            key = role[0]
            config.setdefault(key, {})
        try:
            parallel = max(1, int(self.parallel.get() or 3))
        except (TypeError, ValueError):
            parallel = 3
        try:
            steps = max(1, int(self.steps.get() or 12))
        except (TypeError, ValueError):
            steps = 12
        config["back"].update({
            "max_parallel": parallel,
            "max_steps": steps,
            "tools": "device, files",
        })
        config["router"].setdefault("mode", "keyword-first")
        return config

    def _test(self) -> None:
        """Run one turn with the entered conversation model and report its latency.

        This is the plan's acceptance test for hot-swap: pick a model, measure it, without
        restarting the app. It refuses to run while a turn is already in flight.
        """
        model = self.rows["front"][0].get().strip()
        if not model:
            self.status.set("pick a conversation model first")
            return
        try:
            busy = str((self.session.state() or {}).get("status")) not in {"idle", "listening"}
        except Exception as exc:  # noqa: BLE001 - a failed status read must not block the test
            busy = False
            self.status.set(f"could not read the status ({type(exc).__name__}); testing anyway")
        if busy:
            self.status.set("the assistant is busy - try again in a moment")
            return
        self.status.set(f"testing {model} ...")
        self.update_idletasks()
        try:
            self.session.settings.model = model  # hot-swap just for this probe
            started = time.perf_counter()
            record = self.session.ask("what time is it", source="panel") or {}
            elapsed = (time.perf_counter() - started) * 1000.0
            timings = record.get("timings") or {}
            self.status.set(
                f"{model}: {(timings.get('total_ms') or elapsed):.0f} ms total, "
                f"llm {timings.get('llm_ms', 0.0):.0f} ms, "
                f"first sound {timings.get('filler_ms', 0.0):.0f} ms"
            )
        except Exception as exc:  # noqa: BLE001 - show the failure in the panel
            self.status.set(f"test failed: {type(exc).__name__}: {exc}")

    def _save(self) -> None:
        try:
            path = self.session.apply_agent_config(self._collect())
        except Exception as exc:  # noqa: BLE001 - show the failure, do not crash
            self.status.set(f"could not save: {type(exc).__name__}: {exc}")
            return
        try:
            self.session.settings.session_cost_cap = float(self.cap.get() or 0.0)
        except (TypeError, ValueError):
            self.session.settings.session_cost_cap = 0.0
        self.status.set(f"saved to {path}. The next turn uses the new conversation model.")
