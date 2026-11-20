"""Make CLI output able to print any unicode the models produce.

Windows consoles default to cp1252, so a reply containing an emoji or a
typographic dash crashed the checks with UnicodeEncodeError. Every CLI entry
point calls use_utf8_stdout() first.
"""

from __future__ import annotations

import sys


def use_utf8_stdout() -> None:
    """Switch stdout to UTF-8 with replacement characters, where supported."""
    reconfigure = getattr(sys.stdout, "reconfigure", None)
    if callable(reconfigure):
        reconfigure(encoding="utf-8", errors="replace")
