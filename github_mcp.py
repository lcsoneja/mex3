"""ME3 GitHub MCP server: authenticated GitHub REST API v3 tools for Back agents.

Restrict access to Back workers (`back-1..back-5`) via SONJ4 server policy;
Front agents do not receive `github-mcp` and delegate GitHub work to Back.

Reads token from GITHUB_TOKEN / SONJ4_GITHUB_TOKEN / GH_TOKEN or ~/.sonj4-github-token.
Run standalone (stdio, speaks MCP):
    python sonj4/github_mcp.py
"""

from __future__ import annotations

import base64
import json
import urllib.error
import urllib.parse
import urllib.request
from typing import Any

from github_auth import GITHUB_API_URL, load_github_token, verify_github_token
from mcp_core import Server, ToolError, make_tool

TIMEOUT_S = 25


def _token() -> str:
    tok = load_github_token()
    if not tok:
        raise ToolError(
            "GitHub API is not logged in yet. Ask the owner to sign in via SONJ4 Settings -> GitHub API Auth or 'sonj4 github-login'."
        )
    return tok


def _api(
    method: str,
    endpoint: str,
    *,
    params: dict[str, Any] | None = None,
    body: dict[str, Any] | None = None,
) -> tuple[int, Any, dict[str, str]]:
    tok = _token()
    if endpoint.startswith("http"):
        parsed = urllib.parse.urlparse(endpoint)
        if parsed.scheme != "https" or parsed.hostname not in {
            "api.github.com",
            "github.com",
            "uploads.github.com",
        }:
            raise ToolError("only https://api.github.com endpoints are allowed")
        path = endpoint
    else:
        path = f"{GITHUB_API_URL}/{endpoint.lstrip('/')}"
    if params:
        clean_params = {
            k: str(v) for k, v in params.items() if v is not None and v != ""
        }
        if clean_params:
            sep = "&" if "?" in path else "?"
            path = f"{path}{sep}{urllib.parse.urlencode(clean_params)}"
    data = None if body is None else json.dumps(body).encode("utf-8")
    headers = {
        "Authorization": f"Bearer {tok}",
        "Accept": "application/vnd.github+json",
        "X-GitHub-Api-Version": "2022-11-28",
        "User-Agent": "SONJ4-Back-MCP/1.0",
    }
    if data is not None:
        headers["Content-Type"] = "application/json"
    req = urllib.request.Request(
        path, data=data, headers=headers, method=method.upper()
    )
    try:
        with urllib.request.urlopen(req, timeout=TIMEOUT_S) as resp:
            raw = resp.read().decode("utf-8", errors="replace")
            hdrs = {k.lower(): v for k, v in resp.headers.items()}
            parsed = json.loads(raw) if raw.strip() else {}
            return resp.status, parsed, hdrs
    except urllib.error.HTTPError as exc:
        err_body = exc.read().decode("utf-8", errors="replace")[:500]
        try:
            msg = json.loads(err_body).get("message", err_body)
        except ValueError:
            msg = err_body
        raise ToolError(
            f"GitHub API {method.upper()} {endpoint} failed (HTTP {exc.code}): {msg}"
        ) from exc
    except (OSError, ValueError, urllib.error.URLError) as exc:
        raise ToolError(
            f"GitHub API request error ({type(exc).__name__}): {exc}"
        ) from exc


def _resolve_owner(owner: str) -> str:
    clean = str(owner or "").strip()
    if clean:
        return clean
    info = verify_github_token(_token())
    return str(info["login"])


def _split_owner_repo(owner: str, repo: str) -> tuple[str, str]:
    """Accept repo='name', 'owner/name', or 'https://github.com/owner/name'."""
    raw_repo = str(repo or "").strip().rstrip("/")
    raw_repo = raw_repo.removesuffix(".git")
    if "github.com/" in raw_repo:
        raw_repo = raw_repo.split("github.com/", 1)[1].strip("/")
    parts = [p for p in raw_repo.split("/") if p]
    if len(parts) >= 2:
        return parts[0], parts[1]
    return _resolve_owner(owner), raw_repo


def _safe_int(val: Any, default: int = 0) -> int:
    try:
        return int(val)
    except (TypeError, ValueError) as exc:
        raise ToolError(f"expected integer, got {val!r}") from exc


def _json_str(obj: Any) -> str:
    return json.dumps(obj, indent=2, ensure_ascii=False)


def github_status() -> str:
    """Check authenticated GitHub user, scopes, and remaining rate limit."""
    info = verify_github_token(_token())
    return _json_str(info)


def list_repos(owner: str = "", type: str = "owner", per_page: int = 20) -> str:
    """List repositories for the authenticated user or a specified GitHub owner."""
    n = max(1, min(100, _safe_int(per_page, 20)))
    if owner.strip():
        _, data, _ = _api(
            "GET",
            f"/users/{urllib.parse.quote(owner.strip())}/repos",
            params={"per_page": n, "sort": "updated"},
        )
    else:
        _, data, _ = _api(
            "GET",
            "/user/repos",
            params={"type": type or "owner", "per_page": n, "sort": "updated"},
        )
    rows = [
        {
            "full_name": r.get("full_name"),
            "private": bool(r.get("private")),
            "default_branch": r.get("default_branch"),
            "html_url": r.get("html_url"),
            "description": r.get("description"),
        }
        for r in (data if isinstance(data, list) else [])
    ]
    return _json_str({"count": len(rows), "repos": rows})


def get_repo(owner: str = "", *, repo: str) -> str:
    """Get details of a GitHub repository."""
    own, repo = _split_owner_repo(owner, repo)
    _, data, _ = _api(
        "GET", f"/repos/{urllib.parse.quote(own)}/{urllib.parse.quote(repo.strip())}"
    )
    return _json_str(
        {
            "full_name": data.get("full_name"),
            "private": bool(data.get("private")),
            "default_branch": data.get("default_branch"),
            "html_url": data.get("html_url"),
            "clone_url": data.get("clone_url"),
            "description": data.get("description"),
            "open_issues_count": data.get("open_issues_count"),
            "stargazers_count": data.get("stargazers_count"),
        }
    )


def create_repo(
    name: str,
    description: str = "",
    private: bool = False,
    auto_init: bool = True,
    gitignore_template: str = "",
    license_template: str = "",
) -> str:
    """Create a GitHub repository (public by default for demo; auto_init=True creates README + main branch)."""
    clean_name = str(name or "").strip()
    if not clean_name:
        raise ToolError("name is required")
    payload: dict[str, Any] = {
        "name": clean_name,
        "description": str(description or ""),
        "private": bool(private),
        "auto_init": bool(auto_init),
    }
    if gitignore_template.strip():
        payload["gitignore_template"] = gitignore_template.strip()
    if license_template.strip():
        payload["license_template"] = license_template.strip()
    _, data, _ = _api("POST", "/user/repos", body=payload)
    return _json_str(
        {
            "created": True,
            "full_name": data.get("full_name"),
            "private": bool(data.get("private")),
            "default_branch": data.get("default_branch"),
            "html_url": data.get("html_url"),
            "clone_url": data.get("clone_url"),
        }
    )


def delete_repo(owner: str = "", *, repo: str) -> str:
    """Delete a GitHub repository (requires delete_repo scope)."""
    own, repo = _split_owner_repo(owner, repo)
    _api(
        "DELETE", f"/repos/{urllib.parse.quote(own)}/{urllib.parse.quote(repo.strip())}"
    )
    return _json_str({"deleted": True, "full_name": f"{own}/{repo.strip()}"})


def list_branches(owner: str = "", *, repo: str) -> str:
    """List branches in a GitHub repository."""
    own, repo = _split_owner_repo(owner, repo)
    _, data, _ = _api(
        "GET",
        f"/repos/{urllib.parse.quote(own)}/{urllib.parse.quote(repo.strip())}/branches",
    )
    return _json_str(
        [
            {
                "name": b.get("name"),
                "sha": (b.get("commit") or {}).get("sha"),
                "protected": bool(b.get("protected")),
            }
            for b in (data or [])
        ]
    )


def create_branch(
    owner: str = "", *, repo: str, branch: str, from_branch: str = ""
) -> str:
    """Create a new branch from an existing branch (defaults to repo default_branch)."""
    own, repo = _split_owner_repo(owner, repo)
    rname = repo.strip()
    base = from_branch.strip()
    if not base:
        _, rinfo, _ = _api(
            "GET", f"/repos/{urllib.parse.quote(own)}/{urllib.parse.quote(rname)}"
        )
        base = str(rinfo.get("default_branch") or "main")
    _, ref_info, _ = _api(
        "GET",
        f"/repos/{urllib.parse.quote(own)}/{urllib.parse.quote(rname)}/git/ref/heads/{urllib.parse.quote(base)}",
    )
    sha = str((ref_info.get("object") or {}).get("sha") or "")
    if not sha:
        raise ToolError(f"could not resolve SHA for base branch '{base}'")
    _, created, _ = _api(
        "POST",
        f"/repos/{urllib.parse.quote(own)}/{urllib.parse.quote(rname)}/git/refs",
        body={"ref": f"refs/heads/{branch.strip()}", "sha": sha},
    )
    return _json_str(
        {
            "created": True,
            "branch": branch.strip(),
            "from_branch": base,
            "ref": created.get("ref"),
            "sha": sha,
        }
    )


def get_file(owner: str = "", *, repo: str, path: str = "", ref: str = "") -> str:
    """Read a file (decoded UTF-8 + sha) or list a directory inside a GitHub repository."""
    own, repo = _split_owner_repo(owner, repo)
    clean_path = str(path or "").lstrip("/")
    ep = f"/repos/{urllib.parse.quote(own)}/{urllib.parse.quote(repo.strip())}/contents"
    if clean_path:
        ep += "/" + urllib.parse.quote(clean_path, safe="/")
    _, data, _ = _api("GET", ep, params={"ref": ref} if ref.strip() else None)
    if isinstance(data, list):
        return _json_str(
            {
                "type": "dir",
                "path": clean_path or "/",
                "entries": [
                    {
                        "name": item.get("name"),
                        "path": item.get("path"),
                        "type": item.get("type"),
                        "size": item.get("size"),
                        "sha": item.get("sha"),
                    }
                    for item in data
                ],
            }
        )
    content = ""
    if data.get("encoding") == "base64" and data.get("content"):
        content = base64.b64decode(str(data["content"]).encode("ascii")).decode(
            "utf-8", errors="replace"
        )
    return _json_str(
        {
            "type": data.get("type", "file"),
            "path": data.get("path"),
            "sha": data.get("sha"),
            "size": data.get("size"),
            "html_url": data.get("html_url"),
            "content": content,
        }
    )


def create_or_update_file(
    owner: str = "",
    *,
    repo: str,
    path: str,
    content: str,
    message: str = "Update file via SONJ4 Back agent",
    branch: str = "",
    sha: str = "",
) -> str:
    """Create or overwrite/update a file in a GitHub repository (auto-resolves existing sha if omitted)."""
    own, repo = _split_owner_repo(owner, repo)
    clean_path = str(path or "").lstrip("/")
    if not clean_path:
        raise ToolError("path is required")
    ep = f"/repos/{urllib.parse.quote(own)}/{urllib.parse.quote(repo.strip())}/contents/{urllib.parse.quote(clean_path, safe='/')}"
    file_sha = str(sha or "").strip()
    if not file_sha:
        try:
            _, existing, _ = _api(
                "GET", ep, params={"ref": branch} if branch.strip() else None
            )
            if isinstance(existing, dict) and existing.get("sha"):
                file_sha = str(existing["sha"])
        except ToolError:
            pass
    body: dict[str, Any] = {
        "message": message or f"Update {clean_path}",
        "content": base64.b64encode(str(content).encode("utf-8")).decode("ascii"),
    }
    if branch.strip():
        body["branch"] = branch.strip()
    if file_sha:
        body["sha"] = file_sha
    _, data, _ = _api("PUT", ep, body=body)
    return _json_str(
        {
            "ok": True,
            "path": (data.get("content") or {}).get("path", clean_path),
            "sha": (data.get("content") or {}).get("sha"),
            "commit_sha": (data.get("commit") or {}).get("sha"),
            "html_url": (data.get("content") or {}).get("html_url"),
        }
    )


def edit_file(
    owner: str = "",
    *,
    repo: str,
    path: str,
    old_text: str,
    new_text: str,
    message: str = "Edit file via SONJ4 Back agent",
    branch: str = "",
    replace_all: bool = False,
) -> str:
    """Edit an existing file in a GitHub repository by replacing old_text with new_text and committing."""
    own, repo = _split_owner_repo(owner, repo)
    clean_path = str(path or "").lstrip("/")
    if not clean_path or not old_text:
        raise ToolError("path and non-empty old_text are required")
    ep = f"/repos/{urllib.parse.quote(own)}/{urllib.parse.quote(repo.strip())}/contents/{urllib.parse.quote(clean_path, safe='/')}"
    _, existing, _ = _api("GET", ep, params={"ref": branch} if branch.strip() else None)
    if not isinstance(existing, dict) or not existing.get("sha"):
        raise ToolError(f"file '{clean_path}' not found")
    cur_text = base64.b64decode(
        str(existing.get("content") or "").encode("ascii")
    ).decode("utf-8", errors="replace")
    hits = cur_text.count(old_text)
    if hits == 0:
        raise ToolError(f"old_text not found in {clean_path}")
    if hits > 1 and not replace_all:
        raise ToolError(
            f"old_text appears {hits} times in {clean_path}; provide more context or set replace_all=true"
        )
    updated = (
        cur_text.replace(old_text, new_text)
        if replace_all
        else cur_text.replace(old_text, new_text, 1)
    )
    return create_or_update_file(
        repo=repo,
        path=clean_path,
        content=updated,
        owner=own,
        message=message or f"Edit {clean_path}",
        branch=branch,
        sha=str(existing["sha"]),
    )


def delete_file(
    owner: str = "",
    *,
    repo: str,
    path: str,
    message: str = "Delete file via SONJ4 Back agent",
    branch: str = "",
    sha: str = "",
) -> str:
    """Delete a file in a GitHub repository (auto-resolves sha if omitted)."""
    own, repo = _split_owner_repo(owner, repo)
    clean_path = str(path or "").lstrip("/")
    ep = f"/repos/{urllib.parse.quote(own)}/{urllib.parse.quote(repo.strip())}/contents/{urllib.parse.quote(clean_path, safe='/')}"
    file_sha = str(sha or "").strip()
    if not file_sha:
        _, existing, _ = _api(
            "GET", ep, params={"ref": branch} if branch.strip() else None
        )
        if not isinstance(existing, dict) or not existing.get("sha"):
            raise ToolError(f"file '{clean_path}' not found")
        file_sha = str(existing["sha"])
    body: dict[str, Any] = {
        "message": message or f"Delete {clean_path}",
        "sha": file_sha,
    }
    if branch.strip():
        body["branch"] = branch.strip()
    _, data, _ = _api("DELETE", ep, body=body)
    return _json_str(
        {
            "deleted": True,
            "path": clean_path,
            "commit_sha": (data.get("commit") or {}).get("sha"),
        }
    )


def list_commits(
    owner: str = "", *, repo: str, sha: str = "", per_page: int = 15
) -> str:
    """List recent commits in a GitHub repository."""
    own, repo = _split_owner_repo(owner, repo)
    params: dict[str, Any] = {"per_page": max(1, min(100, _safe_int(per_page, 15)))}
    if sha.strip():
        params["sha"] = sha.strip()
    _, data, _ = _api(
        "GET",
        f"/repos/{urllib.parse.quote(own)}/{urllib.parse.quote(repo.strip())}/commits",
        params=params,
    )
    return _json_str(
        [
            {
                "sha": str(c.get("sha") or "")[:10],
                "message": ((c.get("commit") or {}).get("message") or "").splitlines()[
                    0
                ],
                "author": ((c.get("commit") or {}).get("author") or {}).get("name"),
                "date": ((c.get("commit") or {}).get("author") or {}).get("date"),
            }
            for c in (data if isinstance(data, list) else [])
        ]
    )


def list_issues(
    owner: str = "", *, repo: str, state: str = "open", per_page: int = 20
) -> str:
    """List issues in a GitHub repository."""
    own, repo = _split_owner_repo(owner, repo)
    _, data, _ = _api(
        "GET",
        f"/repos/{urllib.parse.quote(own)}/{urllib.parse.quote(repo.strip())}/issues",
        params={
            "state": state or "open",
            "per_page": max(1, min(100, _safe_int(per_page, 20))),
        },
    )
    return _json_str(
        [
            {
                "number": i.get("number"),
                "title": i.get("title"),
                "state": i.get("state"),
                "is_pr": "pull_request" in i,
                "html_url": i.get("html_url"),
            }
            for i in (data if isinstance(data, list) else [])
        ]
    )


def get_issue(owner: str = "", *, repo: str, issue_number: int) -> str:
    """Get details for one GitHub issue."""
    own, repo = _split_owner_repo(owner, repo)
    _, data, _ = _api(
        "GET",
        f"/repos/{urllib.parse.quote(own)}/{urllib.parse.quote(repo.strip())}/issues/{_safe_int(issue_number)}",
    )
    return _json_str(
        {
            "number": data.get("number"),
            "title": data.get("title"),
            "state": data.get("state"),
            "body": data.get("body"),
            "html_url": data.get("html_url"),
            "user": (data.get("user") or {}).get("login"),
        }
    )


def create_issue(
    owner: str, repo: str, title: str, body: str = "", labels: list[str] | None = None
) -> str:
    """Create a new GitHub issue."""
    own, repo = _split_owner_repo(owner, repo)
    payload: dict[str, Any] = {"title": title, "body": body}
    if labels:
        payload["labels"] = [str(x) for x in labels]
    _, data, _ = _api(
        "POST",
        f"/repos/{urllib.parse.quote(own)}/{urllib.parse.quote(repo.strip())}/issues",
        body=payload,
    )
    return _json_str(
        {
            "number": data.get("number"),
            "title": data.get("title"),
            "html_url": data.get("html_url"),
        }
    )


def add_issue_comment(
    owner: str = "", *, repo: str, issue_number: int, body: str
) -> str:
    """Post a comment on a GitHub issue or pull request."""
    own, repo = _split_owner_repo(owner, repo)
    _, data, _ = _api(
        "POST",
        f"/repos/{urllib.parse.quote(own)}/{urllib.parse.quote(repo.strip())}/issues/{_safe_int(issue_number)}/comments",
        body={"body": body},
    )
    return _json_str({"id": data.get("id"), "html_url": data.get("html_url")})


def list_pull_requests(
    owner: str, repo: str, state: str = "open", per_page: int = 20
) -> str:
    """List pull requests in a GitHub repository."""
    own, repo = _split_owner_repo(owner, repo)
    _, data, _ = _api(
        "GET",
        f"/repos/{urllib.parse.quote(own)}/{urllib.parse.quote(repo.strip())}/pulls",
        params={
            "state": state or "open",
            "per_page": max(1, min(100, _safe_int(per_page, 20))),
        },
    )
    return _json_str(
        [
            {
                "number": p.get("number"),
                "title": p.get("title"),
                "state": p.get("state"),
                "head": (p.get("head") or {}).get("ref"),
                "base": (p.get("base") or {}).get("ref"),
                "html_url": p.get("html_url"),
            }
            for p in (data if isinstance(data, list) else [])
        ]
    )


def get_pull_request(owner: str = "", *, repo: str, pull_number: int) -> str:
    """Get details of one pull request."""
    own, repo = _split_owner_repo(owner, repo)
    _, data, _ = _api(
        "GET",
        f"/repos/{urllib.parse.quote(own)}/{urllib.parse.quote(repo.strip())}/pulls/{_safe_int(pull_number)}",
    )
    return _json_str(
        {
            "number": data.get("number"),
            "title": data.get("title"),
            "state": data.get("state"),
            "body": data.get("body"),
            "head": (data.get("head") or {}).get("ref"),
            "base": (data.get("base") or {}).get("ref"),
            "mergeable": data.get("mergeable"),
            "html_url": data.get("html_url"),
        }
    )


def create_pull_request(
    owner: str, repo: str, title: str, head: str, base: str = "", body: str = ""
) -> str:
    """Create a pull request from head branch into base branch."""
    own, repo = _split_owner_repo(owner, repo)
    target_base = base.strip()
    if not target_base:
        _, rinfo, _ = _api(
            "GET",
            f"/repos/{urllib.parse.quote(own)}/{urllib.parse.quote(repo.strip())}",
        )
        target_base = str(rinfo.get("default_branch") or "main")
    _, data, _ = _api(
        "POST",
        f"/repos/{urllib.parse.quote(own)}/{urllib.parse.quote(repo.strip())}/pulls",
        body={"title": title, "head": head, "base": target_base, "body": body},
    )
    return _json_str(
        {
            "number": data.get("number"),
            "title": data.get("title"),
            "html_url": data.get("html_url"),
        }
    )


def search_repos(query: str, per_page: int = 10) -> str:
    """Search GitHub repositories."""
    _, data, _ = _api(
        "GET",
        "/search/repositories",
        params={"q": query, "per_page": max(1, min(30, _safe_int(per_page, 10)))},
    )
    items = [
        {
            "full_name": r.get("full_name"),
            "stars": r.get("stargazers_count"),
            "html_url": r.get("html_url"),
            "description": r.get("description"),
        }
        for r in (data.get("items") or [])
    ]
    return _json_str({"total_count": data.get("total_count", 0), "items": items})


def search_code(query: str, per_page: int = 10) -> str:
    """Search code across GitHub repositories."""
    _, data, _ = _api(
        "GET",
        "/search/code",
        params={"q": query, "per_page": max(1, min(30, _safe_int(per_page, 10)))},
    )
    items = [
        {
            "name": i.get("name"),
            "path": i.get("path"),
            "repo": (i.get("repository") or {}).get("full_name"),
            "html_url": i.get("html_url"),
        }
        for i in (data.get("items") or [])
    ]
    return _json_str({"total_count": data.get("total_count", 0), "items": items})


def repo_overview(repo: str, owner: str = "", ref: str = "") -> str:
    """Read a GitHub repository in one call: metadata + README + top-level listing.

    Accepts 'owner/name', a bare name (authenticated owner), or a full
    https://github.com/owner/name URL — e.g. roatienza/deep-learning-experiments.
    """
    own, rname = _split_owner_repo(owner, repo)
    base = f"/repos/{urllib.parse.quote(own)}/{urllib.parse.quote(rname)}"
    _, meta, _ = _api("GET", base)
    params = {"ref": ref} if str(ref or "").strip() else None
    readme = ""
    readme_path = ""
    try:
        _, rd, _ = _api("GET", base + "/readme", params=params)
        if isinstance(rd, dict):
            readme_path = str(rd.get("path") or "")
            if rd.get("encoding") == "base64" and rd.get("content"):
                readme = base64.b64decode(str(rd["content"]).encode("ascii")).decode(
                    "utf-8", errors="replace"
                )[:6000]
    except ToolError:
        readme = "(repository has no README)"
    entries: list[dict[str, Any]] = []
    try:
        _, tree, _ = _api("GET", base + "/contents", params=params)
        if isinstance(tree, list):
            entries = [
                {
                    "name": item.get("name"),
                    "type": item.get("type"),
                    "size": item.get("size"),
                }
                for item in tree[:60]
            ]
    except ToolError:
        entries = []
    return _json_str(
        {
            "full_name": meta.get("full_name"),
            "description": meta.get("description"),
            "private": bool(meta.get("private")),
            "default_branch": meta.get("default_branch"),
            "html_url": meta.get("html_url"),
            "stargazers_count": meta.get("stargazers_count"),
            "language": meta.get("language"),
            "topics": meta.get("topics") or [],
            "readme_path": readme_path,
            "readme": readme,
            "top_level": entries,
        }
    )


def api_request(
    method: str = "GET",
    endpoint: str = "/user",
    params: dict[str, Any] | None = None,
    body: dict[str, Any] | None = None,
) -> str:
    """Run any authenticated GitHub REST API v3 request (catch-all endpoint)."""
    status, data, _ = _api(method, endpoint, params=params, body=body)
    return _json_str({"status": status, "data": data})


def build_server() -> Server:
    s = Server("me3-github", version="0.1.0")
    s.add(
        make_tool(
            "github_status",
            "Check authenticated GitHub login, token scopes, and rate limit.",
            {},
            github_status,
        ),
        make_tool(
            "list_repos",
            "List repositories for the authenticated user or a specified GitHub owner.",
            {
                "owner": {"type": "string"},
                "type": {"type": "string"},
                "per_page": {"type": "integer"},
            },
            list_repos,
        ),
        make_tool(
            "get_repo",
            "Get GitHub repository metadata (full_name, visibility, default_branch, urls).",
            {"owner": {"type": "string"}, "repo": {"type": "string"}},
            get_repo,
            required=["repo"],
        ),
        make_tool(
            "create_repo",
            "Create a new GitHub repository (public by default when private=false; auto_init=true adds initial README).",
            {
                "name": {"type": "string"},
                "description": {"type": "string"},
                "private": {"type": "boolean"},
                "auto_init": {"type": "boolean"},
                "gitignore_template": {"type": "string"},
                "license_template": {"type": "string"},
            },
            create_repo,
            required=["name"],
        ),
        make_tool(
            "delete_repo",
            "Delete a GitHub repository (requires delete_repo scope).",
            {"owner": {"type": "string"}, "repo": {"type": "string"}},
            delete_repo,
            required=["repo"],
        ),
        make_tool(
            "list_branches",
            "List branches in a GitHub repository.",
            {"owner": {"type": "string"}, "repo": {"type": "string"}},
            list_branches,
            required=["repo"],
        ),
        make_tool(
            "create_branch",
            "Create a new branch from an existing branch in a GitHub repository.",
            {
                "owner": {"type": "string"},
                "repo": {"type": "string"},
                "branch": {"type": "string"},
                "from_branch": {"type": "string"},
            },
            create_branch,
            required=["repo", "branch"],
        ),
        make_tool(
            "get_file",
            "Read a file (decoded content + sha) or list a directory in a GitHub repository.",
            {
                "owner": {"type": "string"},
                "repo": {"type": "string"},
                "path": {"type": "string"},
                "ref": {"type": "string"},
            },
            get_file,
            required=["repo"],
        ),
        make_tool(
            "create_or_update_file",
            "Create or write/update a file in a GitHub repository (auto-fetches sha if file already exists).",
            {
                "owner": {"type": "string"},
                "repo": {"type": "string"},
                "path": {"type": "string"},
                "content": {"type": "string"},
                "message": {"type": "string"},
                "branch": {"type": "string"},
                "sha": {"type": "string"},
            },
            create_or_update_file,
            required=["repo", "path", "content"],
        ),
        make_tool(
            "edit_file",
            "Edit an existing file in a GitHub repository by replacing old_text with new_text and committing.",
            {
                "owner": {"type": "string"},
                "repo": {"type": "string"},
                "path": {"type": "string"},
                "old_text": {"type": "string"},
                "new_text": {"type": "string"},
                "message": {"type": "string"},
                "branch": {"type": "string"},
                "replace_all": {"type": "boolean"},
            },
            edit_file,
            required=["repo", "path", "old_text", "new_text"],
        ),
        make_tool(
            "delete_file",
            "Delete a file in a GitHub repository (auto-resolves sha if omitted).",
            {
                "owner": {"type": "string"},
                "repo": {"type": "string"},
                "path": {"type": "string"},
                "message": {"type": "string"},
                "branch": {"type": "string"},
                "sha": {"type": "string"},
            },
            delete_file,
            required=["repo", "path"],
        ),
        make_tool(
            "list_commits",
            "List recent commits in a GitHub repository.",
            {
                "owner": {"type": "string"},
                "repo": {"type": "string"},
                "sha": {"type": "string"},
                "per_page": {"type": "integer"},
            },
            list_commits,
            required=["repo"],
        ),
        make_tool(
            "list_issues",
            "List issues in a GitHub repository.",
            {
                "owner": {"type": "string"},
                "repo": {"type": "string"},
                "state": {"type": "string"},
                "per_page": {"type": "integer"},
            },
            list_issues,
            required=["repo"],
        ),
        make_tool(
            "get_issue",
            "Read a GitHub issue.",
            {
                "owner": {"type": "string"},
                "repo": {"type": "string"},
                "issue_number": {"type": "integer"},
            },
            get_issue,
            required=["repo", "issue_number"],
        ),
        make_tool(
            "create_issue",
            "Create a GitHub issue.",
            {
                "owner": {"type": "string"},
                "repo": {"type": "string"},
                "title": {"type": "string"},
                "body": {"type": "string"},
                "labels": {"type": "array"},
            },
            create_issue,
            required=["repo", "title"],
        ),
        make_tool(
            "add_issue_comment",
            "Comment on a GitHub issue or pull request.",
            {
                "owner": {"type": "string"},
                "repo": {"type": "string"},
                "issue_number": {"type": "integer"},
                "body": {"type": "string"},
            },
            add_issue_comment,
            required=["repo", "issue_number", "body"],
        ),
        make_tool(
            "list_pull_requests",
            "List pull requests in a GitHub repository.",
            {
                "owner": {"type": "string"},
                "repo": {"type": "string"},
                "state": {"type": "string"},
                "per_page": {"type": "integer"},
            },
            list_pull_requests,
            required=["repo"],
        ),
        make_tool(
            "get_pull_request",
            "Read a GitHub pull request.",
            {
                "owner": {"type": "string"},
                "repo": {"type": "string"},
                "pull_number": {"type": "integer"},
            },
            get_pull_request,
            required=["repo", "pull_number"],
        ),
        make_tool(
            "create_pull_request",
            "Create a GitHub pull request.",
            {
                "owner": {"type": "string"},
                "repo": {"type": "string"},
                "title": {"type": "string"},
                "head": {"type": "string"},
                "base": {"type": "string"},
                "body": {"type": "string"},
            },
            create_pull_request,
            required=["repo", "title", "head"],
        ),
        make_tool(
            "search_repos",
            "Search GitHub repositories.",
            {"query": {"type": "string"}, "per_page": {"type": "integer"}},
            search_repos,
            required=["query"],
        ),
        make_tool(
            "search_code",
            "Search code across GitHub.",
            {"query": {"type": "string"}, "per_page": {"type": "integer"}},
            search_code,
            required=["query"],
        ),
        make_tool(
            "repo_overview",
            "Read a GitHub repository in one call: metadata + README + top-level files. Accepts owner/name or a full github.com URL (e.g. roatienza/deep-learning-experiments).",
            {
                "repo": {"type": "string"},
                "owner": {"type": "string"},
                "ref": {"type": "string"},
            },
            repo_overview,
            required=["repo"],
        ),
        make_tool(
            "api_request",
            "Call any GitHub REST API v3 endpoint (GET/POST/PUT/PATCH/DELETE).",
            {
                "method": {"type": "string"},
                "endpoint": {"type": "string"},
                "params": {"type": "object"},
                "body": {"type": "object"},
            },
            api_request,
            required=["endpoint"],
        ),
    )
    return s


def main() -> int:
    return build_server().serve()


if __name__ == "__main__":
    raise SystemExit(main())
