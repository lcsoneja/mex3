"""Chat-style transcript and state indicator for the ME3 desktop app.

Tkinter has no chat widget, so ChatView is a scrollable Canvas holding one frame
per turn: your messages on the right, the assistant's on the left, tool calls as
small cards. Keeping it here leaves app_gui.py about wiring and controls.
"""

from __future__ import annotations

import tkinter as tk
from tkinter import ttk
from typing import Any, ClassVar, Literal

# tkinter types these parameters as literals, so the helpers must too.
Side = Literal["left", "right"]
Anchor = Literal["w", "e"]

BG = "#f7f8fa"
USER_BG = "#24324a"
USER_FG = "#eaf1ff"
BOT_BG = "#e6f2ea"
BOT_FG = "#12351f"
TOOL_BG = "#f0e9fb"
TOOL_FG = "#5b21b6"
TOOL_BAD_BG = "#fdecec"
TOOL_BAD_FG = "#b91c1c"
META_FG = "#8a8f9c"
NOTE_FG = "#6b7280"
CHIP_FG = "#6b7280"

FONT_BODY = ("Segoe UI", 10)
FONT_META = ("Segoe UI", 8)
FONT_MONO = ("Consolas", 9)


def _number(value: object, default: float = 0.0) -> float:
    """Timings arrive from a settings/state dict; never crash on a bad one."""
    try:
        return float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return default


class ChatView(ttk.Frame):
    """Scrollable, bubble-styled conversation transcript."""

    def __init__(self, master: tk.Misc, wraplength: int = 560) -> None:
        super().__init__(master)
        self.wraplength = wraplength
        self._bubbles: list[tk.Label] = []
        self._chips_labels: list[tk.Label] = []
        self._notes: list[tk.Label] = []

        # An explicit small size is what keeps this column inside the window: a
        # canvas sizes itself from its contents, and unwrapped wide labels made it
        # request more than the window had, clipping the chat off the right edge.
        self.canvas = tk.Canvas(
            self,
            background=BG,
            highlightthickness=0,
            borderwidth=0,
            width=200,
            height=200,
        )
        self.scrollbar = ttk.Scrollbar(
            self, orient="vertical", command=self.canvas.yview
        )
        self.canvas.configure(yscrollcommand=self.scrollbar.set)
        self.scrollbar.pack(side="right", fill="y")
        self.canvas.pack(side="left", fill="both", expand=True)
        self.canvas.bind("<Map>", lambda _e: self.after_idle(self._reflow), add="+")

        self.inner = tk.Frame(self.canvas, background=BG)
        self._window = self.canvas.create_window((0, 0), window=self.inner, anchor="nw")
        self.inner.bind("<Configure>", self._on_inner_configure)
        self.canvas.bind("<Configure>", self._on_canvas_configure)
        self._bind_wheel(self.canvas)
        self._bind_wheel(self.inner)

    # ------------------------------------------------------------------ plumbing
    def _bind_wheel(self, widget: tk.Misc) -> None:
        widget.bind("<MouseWheel>", self._on_wheel, add="+")

    def _on_wheel(self, event: tk.Event) -> None:
        self.canvas.yview_scroll(int(-event.delta / 120), "units")

    def _on_inner_configure(self, _event: tk.Event) -> None:
        self.canvas.configure(scrollregion=self.canvas.bbox("all"))

    def _on_canvas_configure(self, event: tk.Event) -> None:
        self.canvas.itemconfigure(self._window, width=event.width)
        # The <Configure> width can arrive before the layout settles, which left
        # long lines hanging off the right edge; re-measure once idle.
        self.after_idle(self._reflow)

    def _reflow(self) -> None:
        """Wrap everything to the width the canvas actually has right now."""
        width = self.canvas.winfo_width()
        if width <= 1:
            return
        # Never wrap wider than the window can show, whatever the canvas claims.
        window = self.winfo_toplevel().winfo_width()
        if window > 1:
            width = min(width, max(240, window - 420))
        self.wraplength = max(180, width - 90)
        for label in self._bubbles:
            label.configure(wraplength=self.wraplength)
        for label in self._chips_labels:
            label.configure(wraplength=self.wraplength)

    def _scroll_to_end(self) -> None:
        self.canvas.update_idletasks()
        self.canvas.yview_moveto(1.0)

    def clear(self) -> None:
        for child in self.inner.winfo_children():
            child.destroy()
        self._bubbles.clear()
        self._chips_labels.clear()
        self._notes.clear()

    # -------------------------------------------------------------- adding turns
    def add_note(self, text: str) -> None:
        """A dim, full-width line - the welcome text and system notices."""
        label = tk.Label(
            self.inner,
            text=text,
            font=FONT_META,
            foreground=NOTE_FG,
            background=BG,
            justify="left",
            anchor="w",
            wraplength=self.wraplength,
        )
        label.pack(fill="x", padx=14, pady=(8, 2))
        self._notes.append(label)
        self._bind_wheel(label)
        self.after_idle(self._reflow)
        self._scroll_to_end()

    def add_turn(self, turn: dict[str, Any]) -> None:
        """Render one conversation turn as a chat exchange."""
        stamp = str(turn.get("ts", ""))
        who = "you (voice)" if turn.get("source") == "mic" else "you"

        if turn.get("user_text"):
            self._meta(f"{stamp}   {who}", align="e")
            self._bubble(str(turn["user_text"]), USER_BG, USER_FG, side="right")

        for tool in turn.get("tools", []):
            self._tool_card(tool)

        if turn.get("assistant_text"):
            self._meta(f"{stamp}   assistant", align="w")
            self._bubble(str(turn["assistant_text"]), BOT_BG, BOT_FG, side="left")

        self._chips(
            turn.get("timings", {}),
            str(turn.get("model", "")),
            str(turn.get("source", "")),
        )
        self.after_idle(self._reflow)
        self._scroll_to_end()

    def _meta(self, text: str, align: Anchor) -> None:
        label = tk.Label(
            self.inner,
            text=text,
            font=FONT_META,
            foreground=META_FG,
            background=BG,
            anchor=align,
        )
        label.pack(fill="x", padx=16, pady=(6, 1))
        self._bind_wheel(label)

    def _bubble(self, text: str, background: str, foreground: str, side: Side) -> None:
        anchor: Anchor = "e" if side == "right" else "w"
        row = tk.Frame(self.inner, background=BG)
        row.pack(fill="x", padx=12, pady=(0, 4))
        bubble = tk.Label(
            row,
            text=text,
            font=FONT_BODY,
            background=background,
            foreground=foreground,
            justify="left",
            anchor="w",
            wraplength=self.wraplength,
            padx=12,
            pady=9,
        )
        bubble.pack(side=side, anchor=anchor)
        self._bubbles.append(bubble)
        self._bind_wheel(row)
        self._bind_wheel(bubble)

    def _tool_card(self, tool: dict[str, Any]) -> None:
        ok = bool(tool.get("ok", True))
        row = tk.Frame(self.inner, background=BG)
        row.pack(fill="x", padx=12, pady=(2, 4))
        card = tk.Frame(
            row,
            background=TOOL_BAD_BG if not ok else TOOL_BG,
            padx=10,
            pady=6,
        )
        card.pack(side="left", anchor="w")

        arguments = tool.get("arguments") or {}
        args = ", ".join(f"{key}={value!r}" for key, value in arguments.items())
        header = tk.Label(
            card,
            text=f"{'x' if not ok else '>'} {tool.get('name', 'tool')}({args})   "
            f"{_number(tool.get('seconds')):.2f}s",
            font=FONT_MONO,
            background=TOOL_BAD_BG if not ok else TOOL_BG,
            foreground=TOOL_BAD_FG if not ok else TOOL_FG,
            justify="left",
            anchor="w",
            wraplength=self.wraplength,
        )
        header.pack(anchor="w")
        result = tk.Label(
            card,
            text=str(tool.get("result", "")),
            font=FONT_MONO,
            background=TOOL_BAD_BG if not ok else TOOL_BG,
            foreground="#374151",
            justify="left",
            anchor="w",
            wraplength=self.wraplength - 20,
        )
        result.pack(anchor="w")
        for widget in (row, card, header, result):
            self._bind_wheel(widget)

    def _chips(self, timings: dict[str, Any], model: str, source: str) -> None:
        parts = [
            f"asr {_number(timings.get('asr_ms')):.0f} ms",
            f"llm {_number(timings.get('llm_ms')):.0f} ms",
            f"tts {_number(timings.get('tts_ms')):.0f} ms",
            f"total {_number(timings.get('total_ms')):.0f} ms",
        ]
        if source:
            parts.append(source)
        if model:
            parts.append(model)
        row = tk.Frame(self.inner, background=BG)
        row.pack(fill="x", padx=16, pady=(0, 10))
        label = tk.Label(
            row,
            text="  ·  ".join(parts),
            font=FONT_META,
            foreground=CHIP_FG,
            background=BG,
            anchor="w",
            justify="left",
            wraplength=self.wraplength,
        )
        label.pack(side="left", anchor="w")
        self._chips_labels.append(label)
        self._bind_wheel(row)
        self._bind_wheel(label)

    # ------------------------------------------------------------------ testing
    def dump_text(self) -> str:
        """Every label's text, for the headless self-check."""
        collected: list[str] = []

        def walk(widget: tk.Misc) -> None:
            for child in widget.winfo_children():
                if isinstance(child, tk.Label):
                    collected.append(str(child.cget("text")))
                walk(child)

        walk(self.inner)
        return chr(10).join(collected)


class StateIndicator(ttk.Frame):
    """Dot plus label saying what the app is doing right now."""

    COLORS: ClassVar[dict[str, str]] = {
        "idle": "#9aa3b2",
        "listening": "#22c55e",
        "recording": "#ef4444",
        "transcribing": "#f59e0b",
        "thinking": "#f59e0b",
        "speaking": "#3b82f6",
        "error": "#ef4444",
    }

    def __init__(self, master: tk.Misc, width: int = 18) -> None:
        super().__init__(master)
        self._pulse = False
        self._recording = False
        background = ttk.Style().lookup("TFrame", "background") or "#f0f0f0"
        self.canvas = tk.Canvas(
            self, width=width, height=width, highlightthickness=0, background=background
        )
        self.dot = self.canvas.create_oval(
            4, 4, width - 4, width - 4, fill="#9aa3b2", outline=""
        )
        self.canvas.pack(side="left")
        self.label = ttk.Label(self, text="idle", font=("Segoe UI", 9, "bold"))
        self.label.pack(side="left", padx=(6, 0))

    def update_state(self, state: str, recording: bool) -> None:
        """Show RECORDING while capturing, otherwise the engine state."""
        key = "recording" if recording else state
        color = self.COLORS.get(key, "#9aa3b2")
        text = "RECORDING" if recording else key

        if recording and not self._recording:
            self._pulse = True
        if recording:
            self._pulse = not self._pulse
            inset = 2 if self._pulse else 4
            size = int(self.canvas.cget("width"))
            self.canvas.coords(self.dot, inset, inset, size - inset, size - inset)
        else:
            self._pulse = False

        self._recording = recording
        self.canvas.itemconfigure(self.dot, fill=color)
        self.label.configure(text=text, foreground=color)
