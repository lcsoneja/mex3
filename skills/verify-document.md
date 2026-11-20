# Document Verification & Forensics

When asked to deduce whether a document is real or fake, or to verify claims/files:
1. Inspect the file structure and metadata directly using `pi.bash` / Python (`pymupdf` for PDF producer/creator/dates/fonts, `openpyxl` for Excel formulas vs cached values, `stat`/SHA-256 hash, hidden/extra files) and `pi.read` / `pi.grep`.
2. Check internal consistency: dates, totals/arithmetic, citations, signatures/watermarks, mismatched formatting or editing traces.
3. Cross-check factual claims, statutes, official gazettes, or entities against live web sources (`extensions.web_search` / `extensions.fetch_content`) or local reference files.
4. Consult the `critic` role pane via `ask_agent` to challenge weak assumptions and verify your verdict before answering.
5. State a clear verdict (**VERIFIED**, **SUSPICIOUS**, or **LIKELY FAKE**) backed by exact file paths, page/line citations, metadata, and external evidence.
