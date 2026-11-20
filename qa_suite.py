"""30-test QA suite for the SONJ app, driven over HTTP like the real UI.

Covers what a pi.dev-style harness should do: research, building simple apps,
folder/file ops, counterfeit-document detection (a scored benchmark), PDF
reading, PowerPoint writing, and general assistant tasks. Mutes TTS for the run.
Run with the engine up:  python qa_suite.py  ->  out/qa_report.md
"""

import subprocess
import sys
import time
from pathlib import Path

from front_bench import get, post

ROOT = Path(__file__).resolve().parent
QA = ROOT / "out" / "qa"
RESULTS = []
COUNTERFEIT = []  # (name, truth, verdict_ok, reply)


def compiles(path):
    if not path.is_file():
        return False, "file missing"
    p = subprocess.run(
        [sys.executable, "-m", "py_compile", str(path)],
        capture_output=True,
        timeout=30,
        check=False,
    )
    return p.returncode == 0, ("compiles" if p.returncode == 0 else "syntax errors")


def run_script(path, *args):
    if not path.is_file():
        return False, "file missing"
    p = subprocess.run(
        [sys.executable, str(path), *args],
        capture_output=True,
        timeout=40,
        check=False,
        cwd=str(ROOT),
    )
    return p.returncode == 0, (
        p.stdout.decode(errors="replace").strip()[:120]
        or p.stderr.decode(errors="replace").strip()[:120]
    )


def reply_of(turn):
    return str(
        turn.get("assistant_text") or turn.get("text") or turn.get("reply") or ""
    )


def wait_completion(before, timeout_s):
    """Wait for the turn AND any asynchronous delegation to finish.

    Front acknowledges a delegated task instantly while Back/Researcher keep
    working in the background; checking at the ack moment false-fails. Wait until
    every pane is idle and a result turn has landed.
    """
    deadline = time.time() + timeout_s
    last = None
    while time.time() < deadline:
        st = get("/api/state")
        turns = st.get("turns") or []
        if len(turns) > before:
            last = turns[-1]
        panes = st.get("pi_panes") or {}
        busy = any(
            str(p.get("status", "")).lower()
            in ("thinking", "running", "working", "busy")
            for p in panes.values()
        )
        if last is not None and not busy and st.get("status") == "idle":
            time.sleep(1.5)
            st2 = get("/api/state")
            t2 = st2.get("turns") or []
            return (t2[-1] if len(t2) > before else last), st2
        time.sleep(1.0)
    return last, get("/api/state")


def run_test(name, prompt, check, timeout_s=180, category="misc"):
    t0 = time.time()
    try:
        before = len(get("/api/state").get("turns") or [])
        post("/api/send", {"text": prompt})
        turn, st = wait_completion(before, timeout_s)
        wall = round(time.time() - t0, 1)
        if turn is None:
            RESULTS.append(
                {
                    "name": name,
                    "cat": category,
                    "ok": False,
                    "wall": wall,
                    "note": "no turn in time",
                    "reply": "",
                }
            )
            print(f"FAIL {name}: no turn within {timeout_s}s", flush=True)
            return
        reply = reply_of(turn)
        ok, note = check(reply, turn, st)
        tim = turn.get("timings") or {}
        RESULTS.append(
            {
                "name": name,
                "cat": category,
                "ok": bool(ok),
                "wall": wall,
                "e2e_ms": tim.get("e2e_ms", ""),
                "note": note,
                "reply": reply[:300],
            }
        )
        print(f"{'PASS' if ok else 'FAIL'} {name} ({wall}s) {note}", flush=True)
    except Exception as exc:  # noqa: BLE001
        RESULTS.append(
            {
                "name": name,
                "cat": category,
                "ok": False,
                "wall": round(time.time() - t0, 1),
                "note": f"error: {exc}",
                "reply": "",
            }
        )
        print(f"FAIL {name}: {exc}", flush=True)


def has(*words):
    return lambda r, t, s: (
        any(w.lower() in r.lower() for w in words),
        "contains " + "/".join(words),
    )


def no_markdown(r, t, s):
    return ("**" not in r and "##" not in r and len(r) > 0, "markdown-free")


def make_fixtures():
    QA.mkdir(parents=True, exist_ok=True)
    docs = {
        "genuine_invoice.txt": (
            "INVOICE #INV-2024-0091\nVendor: Northwind Traders Ltd, 12 Harbor Rd, Seattle\n"
            "Date: 2024-03-12\nItems: 3 x Widget @ $50.00 = $150.00\n"
            "Subtotal: $150.00\nTax (10%): $15.00\nTotal Due: $165.00\n"
            "Payment terms: Net 30. IBAN consistent with vendor registry."
        ),
        "fake_invoice.txt": (
            "INVOICE #INV-9999\nVendor: Definitely Real Mega Corp, Unknown Street\n"
            "Date: 2024-13-45\nItems: 3 x Widget @ $50.00 = $200.00\n"
            "Subtotal: $150.00\nTax (10%): $99.00\nTotal Due: $500.00\n"
            "Pay immediately to a personal crypto wallet. No registration number."
        ),
        "genuine_certificate.txt": (
            "CERTIFICATE OF COMPLETION\nThis certifies that Jane Doe completed the"
            " Data Engineering course.\nIssued: 2024-06-01 by Cascade Institute."
        ),
        "fake_certificate.txt": (
            "CERTIFICATE OF COMPLETION\nThis certifies that Jane Doe completed the"
            " Data Engineering course.\nIssued: 2025-06-01\nExpires: 2020-01-01\n"
            "Signed: Mickey Mouse, President of the University of Nowhere."
        ),
        "genuine_abstract.txt": (
            "Abstract: We surveyed 420 households across three regions and found a"
            " 12% reduction in energy use after insulation upgrades (p<0.05)."
            " Published in Journal of Building Science, 2023; DOI 10.1016/j.jbs.2023.04.011."
        ),
        "fake_abstract.txt": (
            "Abstract: We surveyed 999999999 households in one day and found a 140%"
            " reduction in energy use (p=0.0000). The control group used more energy"
            " than the total. DOI 10.9999/fake.2024.0000, Journal of Impossible Results."
        ),
    }
    for name, text in docs.items():
        (QA / name).write_text(text, encoding="utf-8")
    (QA / "sample.csv").write_text(
        "name,score\nzoe,72\nadam,95\nmia,61\nlee,88\n", encoding="utf-8"
    )
    # A real PDF with a secret code, generated with reportlab.
    try:
        from reportlab.pdfgen import canvas

        c = canvas.Canvas(str(QA / "secret.pdf"))
        c.drawString(72, 720, "SONJ QA Document")
        c.drawString(72, 700, "QA Secret Code: BANANA-42")
        c.drawString(72, 680, "The capital of France is Paris.")
        c.save()
    except Exception as exc:  # noqa: BLE001
        print(f"pdf fixture skipped: {exc}", flush=True)
    return docs


def counterfeit_check(truth):
    def check(r, t, s):
        low = r.lower()
        fake_words = (
            "counterfeit",
            "fake",
            "forged",
            "fraudulent",
            "not genuine",
            "suspicious",
            "inauthentic",
            "invalid",
        )
        real_words = ("genuine", "authentic", "legitimate", "valid", "real")
        says_fake = any(w in low for w in fake_words)
        says_real = any(w in low for w in real_words) and not says_fake
        verdict = (
            "counterfeit" if says_fake else ("genuine" if says_real else "unclear")
        )
        ok = verdict == truth
        COUNTERFEIT.append((t.get("_qa_name", ""), truth, ok, r[:160]))
        return ok, f"truth={truth} verdict={verdict}"

    return check


def pptx_valid(path, min_slides=1):
    if not path.is_file():
        return False, "file missing"
    try:
        from pptx import Presentation

        n = len(Presentation(str(path)).slides)
        return n >= min_slides, f"{n} slides"
    except Exception as exc:  # noqa: BLE001
        return False, f"invalid pptx: {exc}"


def main():
    get("/api/health")
    post("/api/settings", {"tts_enabled": False})
    make_fixtures()

    # A. Front direct (6)
    run_test("A1-chat-joke", "tell me a short joke", no_markdown, 60, "front")
    run_test("A2-device-time", "what time is it", has(":"), 60, "front")
    run_test("A3-device-calc", "calculate 17 * 23", has("391"), 60, "front")
    run_test(
        "A4-memory-store",
        "remember that my favorite color is blue",
        lambda r, t, s: (len(r) > 0, "acknowledged"),
        60,
        "front",
    )
    run_test("A5-memory-recall", "what is my favorite color?", has("blue"), 60, "front")
    run_test(
        "A6-clarify",
        "check it",
        lambda r, t, s: (r.rstrip().endswith("?"), "clarifies"),
        60,
        "front",
    )

    # B. Research (4)
    run_test(
        "B1-fact",
        "what is the capital of Mongolia",
        has("ulaanbaatar"),
        120,
        "research",
    )
    run_test(
        "B2-price",
        "research the current NVIDIA stock price and give me the number",
        lambda r, t, s: (len(r) > 10 and any(c.isdigit() for c in r), "price reported"),
        180,
        "research",
    )
    run_test(
        "B3-news",
        "research the latest news about raspberry pi and give one headline",
        lambda r, t, s: (len(r) > 20, "headline"),
        180,
        "research",
    )
    run_test(
        "B4-compare",
        "research and compare the population of Tokyo and Manila, which is larger",
        has("tokyo"),
        180,
        "research",
    )

    # C. Build simple apps (4)
    run_test(
        "C1-snake",
        "create a snake game in sonj3/out/qa/snake_qa.py",
        lambda r, t, s: compiles(QA / "snake_qa.py"),
        180,
        "apps",
    )

    def calc_check(r, t, s):
        ok, note = compiles(QA / "calc_qa.py")
        if ok:
            ran, out = run_script(QA / "calc_qa.py")
            return ran and "391" in out, f"runs: {out}"
        return ok, note

    run_test(
        "C2-calculator",
        "write sonj3/out/qa/calc_qa.py that prints the result of 17 * 23, then run it",
        calc_check,
        180,
        "apps",
    )
    run_test(
        "C3-todo",
        "write a simple command-line todo app in sonj3/out/qa/todo_qa.py",
        lambda r, t, s: compiles(QA / "todo_qa.py"),
        180,
        "apps",
    )

    def sorter_check(r, t, s):
        ok, note = compiles(QA / "sort_qa.py")
        if not ok:
            return ok, note
        ran, out = run_script(QA / "sort_qa.py")
        sorted_csv = QA / "sorted.csv"
        if sorted_csv.is_file():
            lines = sorted_csv.read_text().strip().splitlines()[1:]
            scores = [int(x.split(",")[1]) for x in lines if "," in x]
            return scores == sorted(scores), "sorted.csv ascending"
        return ran, f"ran: {out}"

    run_test(
        "C4-sorter",
        "write sonj3/out/qa/sort_qa.py that sorts sonj3/out/qa/sample.csv by the score column ascending and saves sonj3/out/qa/sorted.csv, then run it",
        sorter_check,
        180,
        "apps",
    )

    # D. Folder / file ops (4)
    run_test(
        "D1-list",
        "list the files in the sonj3/out/qa folder",
        has("invoice", "sample"),
        120,
        "files",
    )
    run_test(
        "D2-read",
        "read sonj3/out/qa/genuine_invoice.txt and tell me the total amount due",
        has("165"),
        120,
        "files",
    )
    run_test(
        "D3-count",
        "how many .txt files are in sonj3/out/qa? give the number",
        has("6", "six"),
        120,
        "files",
    )
    run_test(
        "D4-find",
        "find the file named secret.pdf under sonj3/out and tell me its folder",
        has("qa"),
        120,
        "files",
    )

    # E. Counterfeit-document benchmark (6): 3 genuine + 3 forged
    for name, truth in [
        ("genuine_invoice.txt", "genuine"),
        ("fake_invoice.txt", "counterfeit"),
        ("genuine_certificate.txt", "genuine"),
        ("fake_certificate.txt", "counterfeit"),
        ("genuine_abstract.txt", "genuine"),
        ("fake_abstract.txt", "counterfeit"),
    ]:

        def ck(r, t, s, _n=name, _tr=truth):
            t["_qa_name"] = _n
            return counterfeit_check(_tr)(r, t, s)

        run_test(
            f"E-{name}",
            f"read sonj3/out/qa/{name} and classify the document as GENUINE or COUNTERFEIT, with one reason",
            ck,
            150,
            "counterfeit",
        )

    # F. PDF reading (2)
    run_test(
        "F1-pdf-code",
        "read the PDF at sonj3/out/qa/secret.pdf and tell me the QA secret code",
        has("banana"),
        180,
        "pdf",
    )
    run_test(
        "F2-pdf-summarize",
        "read sonj3/out/qa/secret.pdf and summarize its contents in one sentence",
        has("paris", "code", "banana"),
        180,
        "pdf",
    )

    # G. PowerPoint writing (2)
    run_test(
        "G1-deck",
        "create a 3-slide PowerPoint at sonj3/out/qa/deck_qa.pptx about voice assistants",
        lambda r, t, s: pptx_valid(QA / "deck_qa.pptx", 3),
        200,
        "pptx",
    )
    run_test(
        "G2-data-deck",
        "create a PowerPoint at sonj3/out/qa/data_deck.pptx with a title slide and a slide summarizing sonj3/out/qa/sample.csv",
        lambda r, t, s: pptx_valid(QA / "data_deck.pptx", 2),
        200,
        "pptx",
    )

    # H. Harness misc (2)
    run_test(
        "H1-health",
        "run a health check on the app and tell me the status",
        lambda r, t, s: (len(r) > 15, "health reported"),
        200,
        "misc",
    )
    run_test(
        "H2-incidents",
        "check the incident log and tell me if there are errors",
        lambda r, t, s: (len(r) > 10, "incidents reviewed"),
        200,
        "misc",
    )

    write_report()
    return 0


def write_report():
    passed = sum(1 for r in RESULTS if r["ok"])
    cats = {}
    for r in RESULTS:
        c = cats.setdefault(r["cat"], [0, 0])
        c[1] += 1
        if r["ok"]:
            c[0] += 1
    lines = [
        "# SONJ QA report - " + time.strftime("%Y-%m-%d %H:%M:%S"),
        "",
        f"RESULT: {passed}/{len(RESULTS)} passed",
        "",
        "## By category",
        "",
        "| category | passed | total |",
        "| --- | ---: | ---: |",
    ]
    for c, (p, n) in cats.items():
        lines.append(f"| {c} | {p} | {n} |")
    # Counterfeit benchmark scores
    if COUNTERFEIT:
        tp = sum(1 for _, tr, ok, _ in COUNTERFEIT if tr == "counterfeit" and ok)
        fn = sum(1 for _, tr, ok, _ in COUNTERFEIT if tr == "counterfeit" and not ok)
        tn = sum(1 for _, tr, ok, _ in COUNTERFEIT if tr == "genuine" and ok)
        fp = sum(1 for _, tr, ok, _ in COUNTERFEIT if tr == "genuine" and not ok)
        acc = (tp + tn) / len(COUNTERFEIT)
        prec = tp / (tp + fp) if (tp + fp) else 0
        rec = tp / (tp + fn) if (tp + fn) else 0
        lines += [
            "",
            "## Counterfeit-document benchmark",
            "",
            f"Accuracy: {acc:.0%}  Precision(fake): {prec:.0%}  Recall(fake): {rec:.0%}",
            f"TP={tp} FN={fn} TN={tn} FP={fp} (n={len(COUNTERFEIT)})",
            "",
            "| document | truth | correct |",
            "| --- | --- | --- |",
        ]
        for name, tr, ok, _ in COUNTERFEIT:
            lines.append(f"| {name} | {tr} | {'yes' if ok else 'NO'} |")
    lines += [
        "",
        "## All tests",
        "",
        "| test | cat | ok | wall s | e2e ms | note |",
        "| --- | --- | --- | ---: | ---: | --- |",
    ]
    for r in RESULTS:
        lines.append(
            f"| {r['name']} | {r['cat']} | {'PASS' if r['ok'] else 'FAIL'} | {r.get('wall', '')} | {r.get('e2e_ms', '')} | {r.get('note', '')} |"
        )
    lines += ["", "## Replies", ""]
    for r in RESULTS:
        lines.append(f"- {r['name']}: {r.get('reply', '')}")
    out = ROOT / "out" / "qa_report.md"
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"report: {out}", flush=True)
    print(f"RESULT: {passed}/{len(RESULTS)} passed", flush=True)


if __name__ == "__main__":
    code = 1
    try:
        code = main()
    finally:
        try:
            post("/api/settings", {"tts_enabled": True})
        except Exception as exc:  # noqa: BLE001
            print(f"tts restore skipped: {exc}", flush=True)
    sys.exit(code)
