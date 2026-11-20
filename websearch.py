"""Web search and page reading for the voice agent.

Provider order: the ``ddgs`` package (general web, no key), then DuckDuckGo's
raw endpoints, then Wikipedia as a factual fallback, and Brave when a
BRAVE_API_KEY is configured. Results are plain text snippets, which is what a
spoken assistant actually needs.

Provider check:  python me3_laptop/websearch.py --selftest
"""

from __future__ import annotations

import ipaddress
import re
import socket
import sys
from dataclasses import dataclass
from html import unescape
from typing import Any
from urllib.parse import parse_qs, quote, urlparse

import requests

USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/120.0 Safari/537.36"
)
TIMEOUT_S = 20
MAX_PAGE_CHARS = 4000
DDG_HTML = "https://html.duckduckgo.com/html/"
DDG_LITE = "https://lite.duckduckgo.com/lite/"
BRAVE_API = "https://api.search.brave.com/res/v1/web/search"
WIKI_API = "https://en.wikipedia.org/w/api.php"


class SearchError(RuntimeError):
    """Raised when every provider fails or input is unusable."""


@dataclass
class Result:
    """One search hit."""

    title: str
    url: str
    snippet: str


def _clean(text: str) -> str:
    """Strip tags/entities and collapse whitespace."""
    without_tags = re.sub(r"<[^>]+>", " ", text)
    return re.sub(r"\s+", " ", unescape(without_tags)).strip()


def _console_safe(text: str) -> str:
    """Windows consoles are cp1252; snippets can contain any Unicode."""
    encoding = getattr(sys.stdout, "encoding", None) or "utf-8"
    try:
        return text.encode(encoding, errors="replace").decode(
            encoding, errors="replace"
        )
    except (LookupError, UnicodeError):
        return text.encode("ascii", errors="replace").decode("ascii")


def _is_public_url(url: str) -> bool:
    """Only allow http(s) to publicly routable hosts (no SSRF into the LAN)."""
    parsed = urlparse(url)
    if parsed.scheme not in {"http", "https"} or not parsed.hostname:
        return False
    try:
        infos = socket.getaddrinfo(parsed.hostname, None)
    except socket.gaierror:
        return False
    for info in infos:
        address = ipaddress.ip_address(info[4][0])
        if (
            address.is_private
            or address.is_loopback
            or address.is_reserved
            or address.is_link_local
        ):
            return False
    return True


def chain_is_public(urls: list[str]) -> bool:
    """True when every URL in a redirect chain is publicly routable."""
    return all(_is_public_url(url) for url in urls)


def _unwrap_ddg(href: str) -> str:
    """DuckDuckGo wraps outbound links in /l/?uddg=<encoded>."""
    if href.startswith("//"):
        href = "https:" + href
    parsed = urlparse(href)
    if "duckduckgo.com" in parsed.netloc and parsed.path.startswith("/l/"):
        target = parse_qs(parsed.query).get("uddg")
        if target:
            return target[0]
    return href


_TIMELIMIT_MAP = {"day": "d", "week": "w", "month": "m", "year": "y"}


def _normalize_timelimit(timelimit: str) -> str:
    raw = str(timelimit or "").strip().lower()
    if not raw or raw in {"any", "none"}:
        return ""
    return _TIMELIMIT_MAP.get(raw, raw)


def _rows_to_results(rows: list[dict[str, Any]], max_results: int) -> list[Result]:
    out: list[Result] = []
    for row in rows[:max_results]:
        title = _clean(str(row.get("title", "")))
        url = str(row.get("href") or row.get("url") or "")
        snippet = _clean(str(row.get("body") or row.get("excerpt") or ""))[:600]
        if title or url:
            out.append(Result(title=title, url=url, snippet=snippet))
    return out


def search_ddgs(query: str, max_results: int = 3, timelimit: str = "") -> list[Result]:
    """General web search through the ddgs package (no API key)."""
    try:
        from ddgs import DDGS
        from ddgs.exceptions import DDGSException
    except ImportError as exc:
        raise SearchError("ddgs is not installed (pip install ddgs)") from exc

    tl = _normalize_timelimit(timelimit)
    errors: list[str] = []

    # Current-event queries or explicit recency benefit from DDGS news first.
    low = query.lower()
    want_news = bool(tl in {"d", "w"}) or any(
        w in low for w in ("won", "election", "latest", "today", "news", "results")
    )
    if want_news:
        try:
            news_kwargs: dict[str, Any] = {"max_results": max_results}
            if tl:
                news_kwargs["timelimit"] = tl
            news_rows = DDGS().news(query, **news_kwargs) or []
            results = _rows_to_results(news_rows, max_results)
            if results:
                return results
        except (DDGSException, ValueError, RuntimeError) as exc:
            errors.append(f"news: {exc}")

    # ddgs 'auto' frequently picks backends that time out or return 0 hits on
    # Windows; try reliable text backends in order and retry without timelimit.
    attempts: list[tuple[str, str]] = [
        ("yahoo", tl),
        ("bing", tl),
        ("auto", tl),
    ]
    if tl:
        attempts.extend([("yahoo", ""), ("bing", "")])

    for backend, window in attempts:
        kwargs: dict[str, Any] = {"max_results": max_results, "backend": backend}
        if window:
            kwargs["timelimit"] = window
        try:
            rows = DDGS().text(query, **kwargs) or []
        except (DDGSException, ValueError, RuntimeError) as exc:
            errors.append(f"{backend}: {exc}")
            continue
        results = _rows_to_results(rows, max_results)
        if results:
            return results
        errors.append(f"{backend}: no results")

    raise SearchError("ddgs failed: " + "; ".join(errors[:4]))


def _extract_ddg(body: str, max_results: int) -> list[Result]:
    """Parse the raw DuckDuckGo markups; outbound links carry uddg=."""
    anchors = re.findall(
        r'<a[^>]+href="([^"]*uddg=[^"]+)"[^>]*>(.*?)</a>', body, flags=re.DOTALL
    )
    snippets = re.findall(
        r'class="[^"]*result__snippet[^"]*"[^>]*>(.*?)</(?:a|td|div)>',
        body,
        flags=re.DOTALL,
    )
    results: list[Result] = []
    for index, (href, title) in enumerate(anchors[:max_results]):
        snippet = _clean(snippets[index]) if index < len(snippets) else ""
        results.append(
            Result(title=_clean(title), url=_unwrap_ddg(href), snippet=snippet)
        )
    return results


def search_duckduckgo(query: str, max_results: int = 3) -> list[Result]:
    """Raw DuckDuckGo endpoints. Often blocked (HTTP 202) from datacenter IPs."""
    errors: list[str] = []
    for endpoint, method in ((DDG_HTML, "post"), (DDG_LITE, "post"), (DDG_LITE, "get")):
        try:
            if method == "post":
                response = requests.post(
                    endpoint,
                    data={"q": query},
                    headers={"User-Agent": USER_AGENT},
                    timeout=TIMEOUT_S,
                )
            else:
                response = requests.get(
                    endpoint,
                    params={"q": query},
                    headers={"User-Agent": USER_AGENT},
                    timeout=TIMEOUT_S,
                )
            response.raise_for_status()
        except requests.RequestException as exc:
            errors.append(f"{endpoint} {exc}")
            continue
        results = _extract_ddg(response.text, max_results)
        if results:
            return results
        errors.append(f"{endpoint} status={response.status_code} no parsed results")
    raise SearchError("; ".join(errors) or "duckduckgo returned nothing")


def _wiki_extracts(titles: list[str]) -> dict[str, str]:
    """One batched request for every intro: per-page calls rate-limit fast."""
    if not titles:
        return {}
    response = requests.get(
        WIKI_API,
        params={
            "action": "query",
            "prop": "extracts",
            "exintro": 1,
            "explaintext": 1,
            "redirects": 1,
            "format": "json",
            "titles": "|".join(titles),
        },
        headers={"User-Agent": USER_AGENT},
        timeout=TIMEOUT_S,
    )
    response.raise_for_status()
    pages = response.json().get("query", {}).get("pages", {})
    return {
        str(page.get("title", "")): str(page.get("extract", ""))
        for page in pages.values()
    }


def search_wikipedia(query: str, max_results: int = 3) -> list[Result]:
    """Wikipedia search plus the intro paragraph of each hit."""
    response = requests.get(
        WIKI_API,
        params={
            "action": "query",
            "list": "search",
            "srsearch": query,
            "srlimit": max_results,
            "format": "json",
        },
        headers={"User-Agent": USER_AGENT},
        timeout=TIMEOUT_S,
    )
    response.raise_for_status()
    hits = response.json().get("query", {}).get("search", [])
    titles = [str(hit.get("title", "")) for hit in hits[:max_results]]

    try:
        extracts = _wiki_extracts(titles)
    except (requests.RequestException, ValueError) as exc:
        extracts = {}
        print(f"[search] Wikipedia intros unavailable ({exc})")

    results: list[Result] = []
    for hit, title in zip(hits[:max_results], titles, strict=False):
        summary = extracts.get(title, "") or _clean(str(hit.get("snippet", "")))
        results.append(
            Result(
                title=title,
                url=f"https://en.wikipedia.org/wiki/{quote(title.replace(' ', '_'))}",
                snippet=_clean(summary)[:600],
            )
        )
    return results


def search_brave(query: str, api_key: str, max_results: int = 3) -> list[Result]:
    """Brave Search API (only used when a key is configured)."""
    response = requests.get(
        BRAVE_API,
        params={"q": query, "count": max_results},
        headers={"Accept": "application/json", "X-Subscription-Token": api_key},
        timeout=TIMEOUT_S,
    )
    response.raise_for_status()
    items = response.json().get("web", {}).get("results", [])
    return [
        Result(
            title=_clean(str(item.get("title", ""))),
            url=str(item.get("url", "")),
            snippet=_clean(str(item.get("description", "")))[:600],
        )
        for item in items[:max_results]
    ]


def web_search(
    query: str,
    max_results: int = 3,
    brave_key: str = "",
    timelimit: str = "",
) -> tuple[list[Result], str]:
    """Try each provider in order; return the hits and which one answered."""
    if not query.strip():
        raise SearchError("empty query")

    attempts: list[tuple[str, Any]] = []
    if brave_key:
        attempts.append(("brave", lambda: search_brave(query, brave_key, max_results)))
    attempts.append(("ddgs", lambda: search_ddgs(query, max_results, timelimit)))
    attempts.append(("duckduckgo", lambda: search_duckduckgo(query, max_results)))
    attempts.append(("wikipedia", lambda: search_wikipedia(query, max_results)))

    errors: list[str] = []
    for name, call in attempts:
        try:
            hits = call()
        except (requests.RequestException, ValueError, SearchError) as exc:
            errors.append(f"{name}: {exc}")
            continue
        if hits:
            return hits, name
        errors.append(f"{name}: no results")

    raise SearchError("; ".join(errors) or "all providers failed")


def read_page(url: str, max_chars: int = MAX_PAGE_CHARS) -> str:
    """Fetch one page and return readable text (guarded against LAN targets)."""
    if not _is_public_url(url):
        raise SearchError("that URL is not a public web page")
    response = requests.get(url, headers={"User-Agent": USER_AGENT}, timeout=TIMEOUT_S)
    # requests follows redirects by default, so every hop must be re-checked: a
    # public URL can bounce to http://169.254.169.254/ or into the local LAN.
    hops = [hop.url for hop in response.history] + [response.url]
    if not chain_is_public(hops):
        offending = next(hop for hop in hops if not _is_public_url(hop))
        raise SearchError(f"redirected to a non-public address ({offending})")
    response.raise_for_status()
    body = response.text
    body = re.sub(
        r"<(script|style|noscript)[^>]*>.*?</\1>",
        " ",
        body,
        flags=re.DOTALL | re.IGNORECASE,
    )
    return _clean(body)[:max_chars]


def format_results(results: list[Result], provider: str) -> str:
    """Render hits as short spoken-friendly lines."""
    lines = [f"{len(results)} result(s) from {provider}."]
    for index, item in enumerate(results, start=1):
        lines.append(f"{index}. {item.title} - {item.snippet} ({item.url})")
    return "\n".join(lines)


def _selftest() -> int:
    reconfigure = getattr(sys.stdout, "reconfigure", None)
    if callable(reconfigure):
        reconfigure(encoding="utf-8", errors="replace")

    print("provider report:")
    working: list[str] = []
    for name, call in (
        ("ddgs", search_ddgs),
        ("duckduckgo", search_duckduckgo),
        ("wikipedia", search_wikipedia),
    ):
        try:
            hits = call("Raspberry Pi 5", 3)
        except (SearchError, requests.RequestException) as exc:
            print(f"  {name:11s} FAIL  {_console_safe(str(exc))[:150]}")
            continue
        working.append(name)
        print(f"  {name:11s} ok    {len(hits)} hit(s)")
        if hits:
            print(
                f"              {_console_safe(hits[0].title)} - {_console_safe(hits[0].url)[:70]}"
            )

    print("\nqueries through the chain:")
    failures = 0
    for query in (
        "Raspberry Pi 5",
        "latest AI voice agent news",
        "Manila weather today",
    ):
        try:
            hits, provider = web_search(query)
        except SearchError as exc:
            print(f"  FAIL {query!r}: {_console_safe(str(exc))[:150]}")
            failures += 1
            continue
        print(f"  OK   [{provider}] {query!r} -> {len(hits)} hit(s)")
        for hit in hits[:2]:
            print(f"       {_console_safe(hit.title)[:70]}")
            print(f"       {_console_safe(hit.snippet)[:110]}")

    print("\npage reading:")
    try:
        page = read_page("https://en.wikipedia.org/wiki/Raspberry_Pi")
        print(f"  read {len(page)} chars, starts: {_console_safe(page[:90])!r}")
    except SearchError as exc:
        print(f"  FAILED: {_console_safe(str(exc))}")
        failures += 1
    print("\nURL guard:")
    guard_cases = [
        ("http://192.168.1.1/", False),
        ("http://127.0.0.1/", False),
        ("http://localhost/", False),
        ("http://[::1]/", False),
        ("http://2130706433/", False),
        ("http://0x7f.1/", False),
        ("http://169.254.169.254/latest/meta-data/", False),
        ("file:///C:/Windows/win.ini", False),
        ("https://example.com", True),
    ]
    for url, want in guard_cases:
        got = _is_public_url(url)
        ok = got == want
        failures += 0 if ok else 1
        print(f"  [{'PASS' if ok else 'FAIL'}] {url:42s} allowed={got} (want {want})")

    print("\nredirect chain guard:")
    chain_cases = [
        (["https://example.com", "https://example.com/"], True),
        (["https://example.com", "http://169.254.169.254/"], False),
        (["https://example.com", "http://192.168.1.1/admin"], False),
        (["https://example.com", "https://example.com", "http://127.0.0.1:9/"], False),
    ]
    for chain, want in chain_cases:
        got = chain_is_public(chain)
        ok = got == want
        failures += 0 if ok else 1
        print(
            f"  [{'PASS' if ok else 'FAIL'}] {len(chain)} hop(s) -> allowed={got} (want {want})"
        )

    print(f"\nproviders usable: {', '.join(working) or 'none'}")
    if not working:
        failures += 1
    print("RESULT:", "PASS" if failures == 0 else f"FAIL ({failures})")
    return 1 if failures else 0


if __name__ == "__main__":
    if "--selftest" in sys.argv:
        raise SystemExit(_selftest())
    print(__doc__)
    raise SystemExit(0)
