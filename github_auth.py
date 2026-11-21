"""GitHub API authentication & token management for SONJ4 Back agents.

Tokens are stored strictly outside the git repository in ~/.sonj4-github-token
(chmod 0600) so SONJ4 opens with GitHub API ready for Back agents without
exposing credentials to config.json, logs, or the Front agent.
"""

from __future__ import annotations

import json
import os
import sys
import threading
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

TOKEN_FILE = Path.home() / ".sonj4-github-token"
GITHUB_API_URL = "https://api.github.com"
GITHUB_TOKEN_CREATE_URL = (
    "https://github.com/settings/tokens/new"
    "?description=SONJ4-Back-MCP&scopes=repo,read:user,delete_repo"
)
_STATUS_LOCK = threading.Lock()
_CACHE: dict[str, Any] = {"status": None, "at": 0.0}


def load_github_token() -> str:
    """Return the configured GitHub API token from env or ~/.sonj4-github-token."""
    for env_name in ("SONJ4_GITHUB_TOKEN", "GITHUB_TOKEN", "GH_TOKEN"):
        val = os.environ.get(env_name, "").strip()
        if val:
            return val
    try:
        if TOKEN_FILE.is_file():
            val = TOKEN_FILE.read_text(encoding="utf-8").strip()
            if val:
                return val
    except OSError:
        pass
    return ""


def _token_source() -> str:
    for env_name in ("SONJ4_GITHUB_TOKEN", "GITHUB_TOKEN", "GH_TOKEN"):
        if os.environ.get(env_name, "").strip():
            return f"env:{env_name}"
    if TOKEN_FILE.is_file():
        return str(TOKEN_FILE)
    return ""


def verify_github_token(token: str, timeout_s: float = 12.0) -> dict[str, Any]:
    """Validate a GitHub API token against GET /user without leaking the token."""
    clean = str(token or "").strip()
    if not clean:
        raise ValueError("GitHub token is empty")
    req = urllib.request.Request(
        f"{GITHUB_API_URL}/user",
        headers={
            "Authorization": f"Bearer {clean}",
            "Accept": "application/vnd.github+json",
            "X-GitHub-Api-Version": "2022-11-28",
            "User-Agent": "SONJ4-Back-MCP/1.0",
        },
        method="GET",
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout_s) as resp:
            raw_scopes = resp.headers.get("X-OAuth-Scopes") or ""
            rate_rem = resp.headers.get("X-RateLimit-Remaining") or ""
            data = json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        if exc.code in {401, 403}:
            raise ValueError(f"GitHub rejected token (HTTP {exc.code})") from exc
        raise ValueError(f"GitHub API error (HTTP {exc.code})") from exc
    except (OSError, ValueError, urllib.error.URLError) as exc:
        raise ValueError(
            f"Could not reach api.github.com: {type(exc).__name__}"
        ) from exc

    scopes = [s.strip() for s in raw_scopes.split(",") if s.strip()]
    is_fine_grained = clean.startswith("github_pat_")
    has_repo_scope = (
        True
        if (is_fine_grained and not scopes)
        else any(s in {"repo", "public_repo"} for s in scopes)
    )
    login = str(data.get("login") or "").strip()
    if not login:
        raise ValueError("GitHub response did not include a user login")
    return {
        "ok": True,
        "signed_in": True,
        "login": login,
        "name": str(data.get("name") or login),
        "html_url": str(data.get("html_url") or f"https://github.com/{login}"),
        "scopes": scopes,
        "token_kind": "fine-grained" if is_fine_grained else "classic",
        "has_repo_scope": has_repo_scope,
        "rate_remaining": rate_rem,
        "token_url": GITHUB_TOKEN_CREATE_URL,
    }


def save_github_token(token: str) -> dict[str, Any]:
    """Validate and save the GitHub token to ~/.sonj4-github-token (chmod 0600)."""
    clean = str(token or "").strip()
    info = verify_github_token(clean)
    TOKEN_FILE.write_text(clean + "\n", encoding="utf-8")
    try:
        os.chmod(TOKEN_FILE, 0o600)
    except OSError:
        pass
    info["source"] = str(TOKEN_FILE)
    info["token_file_exists"] = True
    with _STATUS_LOCK:
        _CACHE["status"] = dict(info)
        _CACHE["at"] = time.monotonic()
    return info


def clear_github_token() -> dict[str, Any]:
    """Delete ~/.sonj4-github-token and clear cached auth metadata."""
    try:
        if TOKEN_FILE.is_file():
            TOKEN_FILE.unlink()
    except OSError as exc:
        return {"ok": False, "error": f"could not delete token file: {exc}"}
    with _STATUS_LOCK:
        _CACHE["status"] = None
        _CACHE["at"] = 0.0
    return {
        "ok": True,
        "signed_in": False,
        "login": "",
        "scopes": [],
        "has_repo_scope": False,
        "token_file_exists": False,
        "token_url": GITHUB_TOKEN_CREATE_URL,
    }


def github_auth_status(
    verify_remote: bool = False, max_age_s: float = 300.0
) -> dict[str, Any]:
    """Non-blocking status snapshot (verifies remotely only when requested or uncached)."""
    tok = load_github_token()
    src = _token_source()
    if not tok:
        return {
            "ok": True,
            "signed_in": False,
            "login": "",
            "name": "",
            "scopes": [],
            "has_repo_scope": False,
            "token_file_exists": TOKEN_FILE.is_file(),
            "source": "",
            "token_url": GITHUB_TOKEN_CREATE_URL,
        }
    with _STATUS_LOCK:
        raw_c = _CACHE.get("status")
        cached = dict(raw_c) if isinstance(raw_c, dict) else None
        age = time.monotonic() - float(_CACHE.get("at") or 0.0)
    if not verify_remote:
        if cached and cached.get("signed_in") and age < max_age_s:
            cached["source"] = src
            cached["token_file_exists"] = TOKEN_FILE.is_file()
            return cached
        return {
            "ok": True,
            "signed_in": True,
            "login": (cached or {}).get("login", ""),
            "name": (cached or {}).get("name", ""),
            "scopes": (cached or {}).get("scopes", []),
            "has_repo_scope": (cached or {}).get("has_repo_scope", True),
            "token_file_exists": TOKEN_FILE.is_file(),
            "source": src,
            "token_url": GITHUB_TOKEN_CREATE_URL,
            "verified": bool(cached and cached.get("login")),
        }
    try:
        info = verify_github_token(tok)
        info["source"] = src
        info["token_file_exists"] = TOKEN_FILE.is_file()
        info["verified"] = True
        with _STATUS_LOCK:
            global _CACHED_STATUS, _CACHED_AT
            _CACHED_STATUS = dict(info)
            _CACHED_AT = time.monotonic()
        return info
    except ValueError as exc:
        return {
            "ok": False,
            "signed_in": False,
            "login": "",
            "error": str(exc),
            "token_file_exists": TOKEN_FILE.is_file(),
            "source": src,
            "token_url": GITHUB_TOKEN_CREATE_URL,
        }


def ensure_git_askpass() -> Path:
    """Create a tiny GIT_ASKPASS script for Back-pool git CLI push/clone auth."""
    script = Path(__file__).resolve().parent / "github_askpass.py"
    body = (
        f"#!{sys.executable}\n"
        "import sys\n"
        "from pathlib import Path\n"
        "import os\n\n"
        "prompt = ' '.join(sys.argv[1:]).lower()\n"
        "if 'username' in prompt:\n"
        "    print('x-access-token')\n"
        "else:\n"
        "    tok = (os.environ.get('SONJ4_GITHUB_TOKEN') or os.environ.get('GITHUB_TOKEN') or '').strip()\n"
        "    if not tok:\n"
        "        tf = Path.home() / '.sonj4-github-token'\n"
        "        if tf.is_file():\n"
        "            tok = tf.read_text(encoding='utf-8').strip()\n"
        "    print(tok)\n"
    )
    try:
        if not script.is_file() or script.read_text(encoding="utf-8") != body:
            script.write_text(body, encoding="utf-8")
        os.chmod(script, 0o755)
    except OSError:
        pass
    return script
