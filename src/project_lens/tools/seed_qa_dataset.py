"""Generate starter evaluation questions from the document manifest."""
import json
from pathlib import Path

from project_lens.paths import PROCESSED_DIR

MANIFEST = PROCESSED_DIR / "document_manifest.json"
OUTPUT = PROCESSED_DIR / "qa_dataset.jsonl"


def main():
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    rows = []
    for item in manifest:
        file_name = item["file"]
        project = item["project"]
        rows.extend(
            [
                {"question": f"Tài liệu {file_name} dùng để làm gì?", "expected_files": [file_name], "expected_answer": ""},
                {"question": f"Tài liệu {file_name} có những đầu mục nào?", "expected_files": [file_name], "expected_answer": ""},
                {"question": f"Tài liệu của dự án {project} chứa nội dung gì?", "expected_files": [file_name], "expected_answer": ""},
            ]
        )
    with OUTPUT.open("w", encoding="utf-8") as file:
        for row in rows:
            file.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(f"Created {len(rows)} evaluation questions: {OUTPUT}")


if __name__ == "__main__":
    main()
