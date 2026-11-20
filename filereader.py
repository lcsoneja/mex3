"""Read documents into text the agents can cite.

Every extraction returns text plus citations (page, slide, sheet or heading), so an answer about
a document can point back to where it came from. This layer is read-only by design: nothing here
writes to the filesystem, and only folders on the allowlist can be read at all.

Formats: PDF, DOCX (if python-docx is installed), XLSX/CSV, PPTX, images, and any text format
(markdown, source code, Obsidian notes). A PDF without a text layer needs OCR, which is reported
as a clear error rather than silently returning nothing.
"""

from __future__ import annotations

import csv
import importlib
import os
from dataclasses import dataclass, field
from pathlib import Path

MAX_CHARS = 400_000  # per document, after extraction
MAX_FILE_BYTES = 64_000_000
TEXT_SUFFIXES = {
    ".txt",
    ".md",
    ".markdown",
    ".rst",
    ".log",
    ".json",
    ".yaml",
    ".yml",
    ".toml",
    ".ini",
    ".cfg",
    ".py",
    ".js",
    ".ts",
    ".tsx",
    ".jsx",
    ".html",
    ".css",
    ".sh",
    ".ps1",
    ".bat",
    ".sql",
    ".c",
    ".h",
    ".cpp",
    ".java",
    ".cs",
    ".go",
    ".rs",
    ".rb",
    ".php",
    ".r",
    ".m",
    ".swift",
    ".kt",
}
SUPPORTED = ("pdf", "docx", "xlsx", "xls", "csv", "tsv", "pptx", "image", "text")

# Folders the agents may read. The project and the user's Desktop are allowed by default;
# anything else has to be granted explicitly with allow_folder().
_ALLOWED: list[Path] = []


class FileError(RuntimeError):
    """Raised when a file cannot be read or is not allowed."""


@dataclass
class Citation:
    """A labelled span of a document: where a piece of the text came from."""

    label: str
    text: str
    start: int = 0
    end: int = 0


@dataclass
class Document:
    """Extracted text plus the citations that make it verifiable."""

    path: Path
    kind: str
    text: str
    citations: list[Citation] = field(default_factory=list)
    meta: dict[str, str] = field(default_factory=dict)

    @property
    def chars(self) -> int:
        return len(self.text)

    def excerpt(self, limit: int = 4000) -> str:
        """The head of the document, for a first look."""
        if self.chars <= limit:
            return self.text
        return self.text[:limit] + f"\n\n[... {self.chars - limit} more characters]"


# Only the project itself is readable out of the box. Anything else has to be granted with
# allow_folder() or listed in ME3_ALLOWED_FOLDERS (os.pathsep separated).
#
# Anchored to this file, NOT to the working directory: an audit pointed out that Path.cwd()
# makes the allowlist depend on wherever the app happens to be launched from, which is not a
# safety boundary at all.
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_ALLOWED = (PROJECT_ROOT,)
ALLOWED_ENV = "ME3_ALLOWED_FOLDERS"
LINES_PER_CITATION = 60


def allow_folder(folder: str | Path) -> Path:
    """Grant read access to a folder for this session."""
    path = Path(folder).expanduser().resolve()
    if not path.is_dir():
        raise FileError(f"not a folder: {path}")
    if path not in _ALLOWED:
        _ALLOWED.append(path)
    return path


def allowed_folders() -> list[Path]:
    """Folders the agents may read: the project, plus anything explicitly granted."""
    if not _ALLOWED:
        candidates = list(DEFAULT_ALLOWED)
        extra = os.environ.get(ALLOWED_ENV, "")
        candidates.extend(
            Path(part) for part in extra.split(os.pathsep) if part.strip()
        )
        for folder in candidates:
            if folder.is_dir() and folder.resolve() not in _ALLOWED:
                _ALLOWED.append(folder.resolve())
    return list(_ALLOWED)


def is_allowed(path: Path) -> bool:
    resolved = path.resolve()
    for folder in allowed_folders():
        try:
            resolved.relative_to(folder)
            return True
        except ValueError:
            continue
    return False


def resolve(path: str | Path, base_dir: Path | None = None) -> Path:
    """Turn a caller-supplied path into a readable file, or explain why not."""
    candidate = Path(path).expanduser()
    if not candidate.is_absolute():
        anchor = Path(base_dir).resolve() if base_dir is not None else Path.cwd()
        candidate = anchor / candidate
    candidate = candidate.resolve()
    if not candidate.exists():
        raise FileError(f"file not found: {candidate}")
    if candidate.is_dir():
        raise FileError(f"{candidate} is a folder - use list_folder for folders")
    if not is_allowed(candidate):
        raise FileError(
            f"not allowed to read {candidate}; allowed folders: "
            + ", ".join(str(f) for f in allowed_folders())
        )
    size = candidate.stat().st_size
    if size > MAX_FILE_BYTES:
        raise FileError(
            f"{candidate.name} is {size / 1e6:.0f} MB, over the {MAX_FILE_BYTES / 1e6:.0f} MB limit"
        )
    return candidate


def list_folder(
    folder: str | Path,
    pattern: str = "*",
    limit: int = 200,
    base_dir: Path | None = None,
    include_dirs: bool = False,
) -> list[dict[str, object]]:
    """List readable files (and optionally subfolders) in a folder, newest first."""
    candidate = Path(folder).expanduser()
    if not candidate.is_absolute():
        anchor = Path(base_dir).resolve() if base_dir is not None else Path.cwd()
        candidate = anchor / candidate
    path = candidate.resolve()
    if not path.is_dir():
        raise FileError(f"not a folder: {path}")
    if not is_allowed(path):
        raise FileError(f"not allowed to read {path}")
    rows: list[dict[str, object]] = []
    entries = sorted(
        path.glob(pattern), key=lambda p: (not p.is_dir(), -p.stat().st_mtime)
    )
    for item in entries[:limit]:
        if item.is_dir() and include_dirs and not item.name.startswith("."):
            rows.append(
                {"name": item.name + "/", "suffix": "", "mb": 0.0, "kind": "dir"}
            )
        elif item.is_file():
            rows.append(
                {
                    "name": item.name,
                    "suffix": item.suffix.lower(),
                    "mb": round(item.stat().st_size / 1e6, 2),
                    "kind": kind_of(item),
                }
            )
    return rows


def kind_of(path: Path) -> str:
    suffix = path.suffix.lower()
    if suffix == ".pdf":
        return "pdf"
    if suffix == ".docx":
        return "docx"
    if suffix in {".xlsx", ".xlsm"}:
        return "xlsx"
    if suffix in {".csv", ".tsv"}:
        return suffix.lstrip(".")
    if suffix == ".pptx":
        return "pptx"
    if suffix in {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tif", ".tiff"}:
        return "image"
    if suffix in TEXT_SUFFIXES or not suffix:
        return "text"
    return "unknown"


def read_document(path: str | Path, base_dir: Path | None = None) -> Document:
    """Extract a document, dispatching on its kind."""
    target = resolve(path, base_dir=base_dir)
    kind = kind_of(target)
    if kind == "pdf":
        return _read_pdf(target)
    if kind == "docx":
        return _read_docx(target)
    if kind == "xlsx":
        return _read_xlsx(target)
    if kind in {"csv", "tsv"}:
        return _read_csv(target, kind)
    if kind == "pptx":
        return _read_pptx(target)
    if kind == "image":
        return _read_image(target)
    if kind == "text":
        return _read_text(target)
    raise FileError(f"unsupported file type: {target.name}")


def search(document: Document, query: str, limit: int = 5) -> list[Citation]:
    """Find the passages that mention a query, with their citations."""
    words = [w for w in query.lower().split() if len(w) > 2]
    if not words:
        return []
    scored: list[tuple[int, Citation]] = []
    for citation in document.citations or [Citation(document.path.name, document.text)]:
        haystack = citation.text.lower()
        score = sum(haystack.count(word) for word in words)
        if score:
            scored.append((score, citation))
    scored.sort(key=lambda pair: -pair[0])
    return [citation for _score, citation in scored[:limit]]


# --------------------------------------------------------------------- readers


def _clip(text: str) -> str:
    return text if len(text) <= MAX_CHARS else text[:MAX_CHARS] + "\n\n[truncated]"


def _read_text(path: Path) -> Document:
    """Plain text with line-range citations, so a quote can actually be checked.

    An audit flagged that text files only carried a filename: chunking into line ranges makes
    the citation as verifiable as a PDF page number.
    """
    text = path.read_text(encoding="utf-8", errors="replace")
    lines = text.split(chr(10))
    blocks = [
        chr(10).join(lines[start : start + LINES_PER_CITATION])
        for start in range(0, max(1, len(lines)), LINES_PER_CITATION)
    ]
    citations = [
        Citation(
            f"{path.name}#lines {index * LINES_PER_CITATION + 1}-"
            f"{min(len(lines), (index + 1) * LINES_PER_CITATION)}",
            block,
        )
        for index, block in enumerate(blocks)
    ]
    meta = {"lines": str(len(lines)), "citations": str(len(citations))}
    return Document(path, "text", _clip(text), citations, meta)


def _read_pdf(path: Path) -> Document:
    citations: list[Citation] = []
    parts: list[str] = []
    try:
        import pymupdf  # optional dependency, imported on use

        with pymupdf.open(str(path)) as pdf:
            # Index by page number instead of iterating: PyMuPDF's Document is iterable at
            # runtime but is not typed as Iterable, and its type is also called Document.
            count = int(pdf.page_count)
            for index in range(count):
                # get_text() is overloaded (str | list | dict) in PyMuPDF's stubs; with no
                # argument it returns plain text, and str() makes that explicit.
                page_text = str(pdf.load_page(index).get_text())
                if page_text.strip():
                    parts.append(page_text)
                    citations.append(
                        Citation(f"{path.name}#page {index + 1}", page_text)
                    )
            meta = {"pages": str(count)}
    except ImportError:
        from pypdf import PdfReader

        reader = PdfReader(str(path))
        for index, page in enumerate(reader.pages, start=1):
            page_text = page.extract_text() or ""
            if page_text.strip():
                parts.append(page_text)
                citations.append(Citation(f"{path.name}#page {index}", page_text))
        meta = {"pages": str(len(reader.pages))}
    text = _clip("\n\n".join(parts))
    if not text.strip():
        raise FileError(
            f"{path.name} has no extractable text layer - it is probably a scan and needs OCR"
        )
    return Document(path, "pdf", text, citations, meta)


def _read_docx(path: Path) -> Document:
    # importlib keeps the dependency optional and the type checker happy when it is absent.
    try:
        docx = importlib.import_module("docx")
    except ImportError as exc:
        raise FileError(
            "reading .docx needs python-docx: pip install python-docx"
        ) from exc
    document = docx.Document(str(path))
    parts: list[str] = []
    citations: list[Citation] = []
    heading = "start"
    buffer: list[str] = []
    for paragraph in document.paragraphs:
        style = (paragraph.style.name or "").lower()
        if style.startswith("heading") and buffer:
            text = "\n".join(buffer)
            parts.append(text)
            citations.append(Citation(f"{path.name}#{heading}", text))
            buffer = []
        if style.startswith("heading"):
            heading = paragraph.text.strip() or heading
        buffer.append(paragraph.text)
    if buffer:
        text = "\n".join(buffer)
        parts.append(text)
        citations.append(Citation(f"{path.name}#{heading}", text))
    return Document(
        path,
        "docx",
        _clip("\n\n".join(parts)),
        citations,
        {"paragraphs": str(len(document.paragraphs))},
    )


def _read_xlsx(path: Path) -> Document:
    try:
        openpyxl = importlib.import_module("openpyxl")
    except ImportError as exc:
        raise FileError("reading .xlsx needs openpyxl: pip install openpyxl") from exc
    book = openpyxl.load_workbook(str(path), read_only=True, data_only=True)
    parts: list[str] = []
    citations: list[Citation] = []
    for sheet in book.worksheets:
        rows: list[str] = []
        for row in sheet.iter_rows(values_only=True):
            cells = ["" if cell is None else str(cell) for cell in row]
            if any(cell.strip() for cell in cells):
                rows.append(" | ".join(cells).rstrip(" |"))
            if len(rows) >= 400:
                rows.append("[... more rows]")
                break
        text = f"sheet: {sheet.title}\n" + "\n".join(rows)
        parts.append(text)
        citations.append(Citation(f"{path.name}#sheet {sheet.title}", text))
    book.close()
    return Document(
        path,
        "xlsx",
        _clip("\n\n".join(parts)),
        citations,
        {"sheets": str(len(citations))},
    )


def _read_csv(path: Path, kind: str) -> Document:
    delimiter = "\t" if kind == "tsv" else ","
    rows: list[str] = []
    with path.open(encoding="utf-8", errors="replace", newline="") as handle:
        reader = csv.reader(handle, delimiter=delimiter)
        for index, row in enumerate(reader):
            rows.append(" | ".join(row))
            if index >= 500:
                rows.append("[... more rows]")
                break
    text = "\n".join(rows)
    return Document(
        path,
        kind,
        _clip(text),
        [Citation(path.name, _clip(text))],
        {"rows": str(len(rows))},
    )


def _read_pptx(path: Path) -> Document:
    try:
        pptx = importlib.import_module("pptx")
    except ImportError as exc:
        raise FileError(
            "reading .pptx needs python-pptx: pip install python-pptx"
        ) from exc
    presentation = pptx.Presentation(str(path))
    parts: list[str] = []
    citations: list[Citation] = []
    for index, slide in enumerate(presentation.slides, start=1):
        chunks: list[str] = []
        for shape in slide.shapes:
            # Not every shape has text: guard with getattr rather than assuming the type.
            frame = getattr(shape, "text_frame", None)
            if frame is not None:
                chunks.append(str(frame.text))
        if slide.has_notes_slide and slide.notes_slide.notes_text_frame is not None:
            notes = slide.notes_slide.notes_text_frame.text
            if notes.strip():
                chunks.append(f"notes: {notes}")
        text = "\n".join(chunk for chunk in chunks if chunk.strip())
        if text:
            parts.append(f"slide {index}:\n{text}")
            citations.append(Citation(f"{path.name}#slide {index}", text))
    return Document(
        path,
        "pptx",
        _clip("\n\n".join(parts)),
        citations,
        {"slides": str(len(presentation.slides))},
    )


def _read_image(path: Path) -> Document:
    try:
        pil = importlib.import_module("PIL.Image")
    except ImportError as exc:
        raise FileError("reading images needs Pillow: pip install pillow") from exc
    with pil.open(str(path)) as image:
        meta = {"size": f"{image.width}x{image.height}", "mode": image.mode}
    ocr = os.environ.get("ME3_OCR", "")
    text = f"image {path.name}: {meta['size']} ({meta['mode']})"
    if ocr:
        text += f"\n\nOCR:\n{_run_ocr(path)}"
    else:
        text += "\n\n(no OCR configured: set ME3_OCR=1 after installing Tesseract to read text in images)"
    return Document(path, "image", text, [Citation(path.name, text)], meta)


def _run_ocr(path: Path) -> str:
    try:
        pytesseract = importlib.import_module("pytesseract")
        pil = importlib.import_module("PIL.Image")
    except ImportError:
        return "(OCR unavailable: pip install pytesseract and Tesseract)"
    with pil.open(str(path)) as image:
        return str(pytesseract.image_to_string(image))[:20_000]
