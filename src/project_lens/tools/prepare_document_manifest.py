"""Create a non-destructive manifest for files in documents/."""
from __future__ import annotations

import json
import re
from pathlib import Path

from project_lens.paths import DOCUMENTS_DIR, PROCESSED_DIR

DOCUMENTS = DOCUMENTS_DIR
CHUNKS = PROCESSED_DIR / "chunks.jsonl"
OUTPUT = PROCESSED_DIR / "document_manifest.json"
QA_TEMPLATE = PROCESSED_DIR / "qa_template.jsonl"


def project_from_name(name: str) -> str:
    patterns = [
        r"th[ạa]nh\s*l[ộo]c",
        r"ph[ướu]c\s*l[ýy]",
        r"tr[àa]\s*[ôo]n",
        r"ph[ướu]c\s*t[âa]n",
        r"r[ạa]ch\s*gi[áa]",
        r"[đd][ăa]k\s*r['’]?l[ấa]p",
    ]
    for pattern in patterns:
        match = re.search(pattern, name, re.IGNORECASE)
        if match:
            return match.group(0)
    return "unknown"


def main() -> None:
    if not DOCUMENTS.exists():
        raise FileNotFoundError(DOCUMENTS)

    chunk_counts: dict[str, int] = {}
    if CHUNKS.exists():
        with CHUNKS.open(encoding="utf-8") as file:
            for line in file:
                record = json.loads(line)
                source = record.get("metadata", {}).get("source", "")
                chunk_counts[source] = chunk_counts.get(source, 0) + 1

    manifest = []
    for path in sorted(DOCUMENTS.rglob("*")):
        if not path.is_file() or path.name.startswith("~$"):
            continue
        relative = str(path.relative_to(DOCUMENTS))
        manifest.append(
            {
                "file": relative,
                "extension": path.suffix.lower(),
                "project": project_from_name(path.stem),
                "size_bytes": path.stat().st_size,
                "chunk_count": chunk_counts.get(str(path), 0),
                "document_type": "spreadsheet" if path.suffix.lower() in {".xlsx", ".xls", ".csv"} else "document",
            }
        )

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    with QA_TEMPLATE.open("w", encoding="utf-8") as file:
        for item in manifest:
            file.write(json.dumps({"question": "", "expected_file": item["file"], "expected_answer": ""}, ensure_ascii=False) + "\n")
    print(f"Created manifest for {len(manifest)} files: {OUTPUT}")
    print(f"Created QA template: {QA_TEMPLATE}")


if __name__ == "__main__":
    main()
