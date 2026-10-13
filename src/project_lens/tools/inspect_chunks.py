"""Inspect chunk quality and document coverage without changing data."""
import json
from collections import defaultdict
from pathlib import Path

from project_lens.paths import PROCESSED_DIR

CHUNKS = PROCESSED_DIR / "chunks.jsonl"


def main():
    records = [json.loads(line) for line in CHUNKS.read_text(encoding="utf-8").splitlines() if line.strip()]
    grouped = defaultdict(list)
    for record in records:
        grouped[record.get("metadata", {}).get("source", "unknown")].append(record)

    print(f"TOTAL CHUNKS: {len(records)}")
    print(f"TOTAL SOURCES: {len(grouped)}\n")
    for source, items in sorted(grouped.items(), key=lambda pair: str(pair[0]).lower()):
        lengths = [len(item.get("page_content", "")) for item in items]
        metadata = items[0].get("metadata", {})
        sheets = sorted({str(item.get("metadata", {}).get("sheet_name")) for item in items if item.get("metadata", {}).get("sheet_name")})
        print(f"FILE: {Path(source).name}")
        print(f"  chunks={len(items)}, min={min(lengths)}, max={max(lengths)}, avg={sum(lengths)//len(lengths)}")
        if sheets:
            print(f"  sheets={', '.join(sheets)}")
        sample = items[0].get("page_content", "").replace("\n", " ")
        print(f"  sample={sample[:220]}\n")


if __name__ == "__main__":
    main()
