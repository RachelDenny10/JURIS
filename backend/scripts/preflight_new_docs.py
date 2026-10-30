r"""
Pre-flight check for new PDFs (read-only: uploads and changes nothing).

For each PDF in a folder it reports: size, page count, extractable text
(PyMuPDF only, no OCR), whether the file is identical (SHA-256) to a
stored document, and the closest existing / in-folder document by
TF-IDF cosine similarity of the raw text. A near-duplicate (for example
the same rules published on two different sites) is flagged so it can be
skipped instead of becoming a second "independent" sample.

Run from the backend folder (PostgreSQL must be running):
    python scripts\preflight_new_docs.py --folder "C:\path\to\folder"
"""

import argparse
import hashlib
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import fitz  # noqa: E402  (PyMuPDF)
from sklearn.feature_extraction.text import TfidfVectorizer  # noqa: E402
from sklearn.metrics.pairwise import cosine_similarity  # noqa: E402

from app.core.config import get_settings  # noqa: E402
from app.services import document_service  # noqa: E402

TOKEN_PATTERN = r"(?u)\b[a-zA-Z][a-zA-Z]+\b"
MIN_TEXT_CHARS = 200


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_pdf(path: Path) -> tuple[int, str, str | None]:
    try:
        with fitz.open(path) as doc:
            return doc.page_count, "\n".join(page.get_text("text") for page in doc), None
    except Exception as exc:
        return 0, "", str(exc)[:100]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--folder", required=True)
    parser.add_argument("--dup", type=float, default=0.85, help="similarity treated as near-duplicate")
    parser.add_argument("--near", type=float, default=0.60, help="similarity worth a manual look")
    args = parser.parse_args()

    folder = Path(args.folder)
    if not folder.is_dir():
        print(f"Folder not found: {folder}")
        sys.exit(1)

    settings = get_settings()
    max_bytes = settings.max_upload_size_bytes

    # ---- existing corpus ----
    existing = []
    for doc in document_service.list_documents():
        pdf = settings.project_root / doc.storage_path
        text_file = settings.project_root / doc.text_path if doc.text_path else None
        existing.append(
            {
                "label": f"{doc.category.value} '{doc.title}' [{doc.document_id[:8]}] ({doc.original_filename})",
                "hash": sha256(pdf) if pdf.exists() else None,
                "text": text_file.read_text(encoding="utf-8") if text_file and text_file.exists() else "",
            }
        )

    # ---- new files ----
    files = sorted(p for p in folder.iterdir() if p.is_file() and p.suffix.lower() == ".pdf")
    others = sorted(p for p in folder.iterdir() if p.is_file() and p.suffix.lower() != ".pdf")
    for p in others:
        print(f"NOT A PDF (skipped): {p.name}")
    if not files:
        print("No .pdf files found in that folder.")
        return

    items = []
    for path in files:
        with path.open("rb") as handle:
            header_ok = handle.read(5) == b"%PDF-"
        pages, text, error = read_pdf(path) if header_ok else (0, "", "does not start with %PDF-")
        items.append(
            {
                "path": path,
                "size": path.stat().st_size,
                "hash": sha256(path),
                "pages": pages,
                "text": text,
                "error": error,
            }
        )

    # ---- similarity over raw text (existing + new) ----
    pool = [("existing", i, e["text"]) for i, e in enumerate(existing) if len(e["text"]) >= MIN_TEXT_CHARS]
    pool += [("new", i, n["text"]) for i, n in enumerate(items) if len(n["text"]) >= MIN_TEXT_CHARS]
    sim = None
    index_of: dict = {}
    if len(pool) >= 2:
        matrix = TfidfVectorizer(token_pattern=TOKEN_PATTERN, lowercase=True, sublinear_tf=True).fit_transform(
            [p[2] for p in pool]
        )
        sim = cosine_similarity(matrix)
        index_of = {(kind, i): n for n, (kind, i, _) in enumerate(pool)}

    print(f"\nChecking {len(items)} PDF(s) against {len(existing)} stored document(s)\n")
    for number, item in enumerate(items, start=1):
        chars = len(item["text"])
        per_page = chars / item["pages"] if item["pages"] else 0
        same_existing = [e["label"] for e in existing if e["hash"] == item["hash"]]
        same_new = [
            o["path"].name for o in items if o is not item and o["hash"] == item["hash"] and o["path"].name < item["path"].name
        ]

        best_existing = best_new = None
        if sim is not None and ("new", number - 1) in index_of:
            row = sim[index_of[("new", number - 1)]]
            for (kind, i), n in index_of.items():
                if kind == "new" and i == number - 1:
                    continue
                score = float(row[n])
                if kind == "existing" and (best_existing is None or score > best_existing[0]):
                    best_existing = (score, existing[i]["label"])
                if kind == "new" and (best_new is None or score > best_new[0]):
                    best_new = (score, items[i]["path"].name)

        if item["error"]:
            verdict = f"UNREADABLE ({item['error']})"
        elif item["size"] > max_bytes:
            verdict = f"TOO LARGE for upload limit ({settings.max_upload_size_mb} MB)"
        elif same_existing:
            verdict = "DUPLICATE: identical file already stored. Skip it."
        elif same_new:
            verdict = f"DUPLICATE inside the folder (same as {same_new[0]}). Upload only one."
        elif chars < MIN_TEXT_CHARS or per_page < 100:
            verdict = "NO / LITTLE TEXT LAYER (scanned). OCR will be needed."
        elif best_existing and best_existing[0] >= args.dup:
            verdict = "NEAR-DUPLICATE of a stored document. Probably skip it."
        elif best_new and best_new[0] >= args.dup:
            verdict = "NEAR-DUPLICATE of another file in this folder. Upload only one."
        elif (best_existing and best_existing[0] >= args.near) or (best_new and best_new[0] >= args.near):
            verdict = "similar to another document: check by hand."
        else:
            verdict = "looks new."

        print(f"[{number}] {item['path'].name}")
        print(f"    size {item['size'] / 1_000_000:.2f} MB | pages {item['pages']} | text chars {chars} ({per_page:.0f}/page)")
        print(f"    verdict: {verdict}")
        if best_existing:
            print(f"    closest stored : {best_existing[0]:.3f}  {best_existing[1]}")
        if best_new:
            print(f"    closest in folder: {best_new[0]:.3f}  {best_new[1]}")
        print()

    print("Similarity is a guide, not a verdict: open any flagged pair and compare the titles and dates.")


if __name__ == "__main__":
    main()