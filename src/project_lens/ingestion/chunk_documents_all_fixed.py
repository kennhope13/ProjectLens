"""Fixed multi-format chunker that skips Office temporary files."""

from __future__ import annotations

import json

from project_lens.ingestion import chunk_documents_all as base


def load_text_directly(path):
    content = path.read_text(encoding="utf-8", errors="replace")
    return [base.Document(page_content=content, metadata={"source": str(path)})]


def main() -> None:
    if not base.DOCUMENTS_DIR.exists():
        raise FileNotFoundError(f"Missing documents directory: {base.DOCUMENTS_DIR}")

    # Replace the incompatible automatic encoding detection used by TextLoader.
    base.load_text = load_text_directly
    documents = []
    skipped = []
    failed = []
    for path in base.DOCUMENTS_DIR.rglob("*"):
        if not path.is_file() or path.name.startswith("~$"):
            continue
        if path.suffix.lower() not in base.SUPPORTED_EXTENSIONS:
            skipped.append(path)
            continue
        try:
            documents.extend(base.load_file(path))
        except Exception as error:
            failed.append((path, str(error)))

    if not documents:
        raise ValueError("No readable documents found.")

    splitter = base.RecursiveCharacterTextSplitter(
        chunk_size=base.CHUNK_SIZE,
        chunk_overlap=base.CHUNK_OVERLAP,
        add_start_index=True,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    chunks = splitter.split_documents(documents)
    base.OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    with base.OUTPUT_FILE.open("w", encoding="utf-8") as file:
        for chunk in chunks:
            record = {
                "id": base.create_chunk_id(chunk),
                "page_content": chunk.page_content,
                "metadata": chunk.metadata,
            }
            file.write(json.dumps(record, ensure_ascii=False, default=str) + "\n")

    print(f"Loaded {len(documents)} document sections.")
    print(f"Created {len(chunks)} chunks: {base.OUTPUT_FILE}")
    if skipped:
        print(f"Skipped {len(skipped)} unsupported files.")
    if failed:
        print(f"Failed to read {len(failed)} files:")
        for path, message in failed:
            print(f"- {path.name}: {message}")


if __name__ == "__main__":
    main()
