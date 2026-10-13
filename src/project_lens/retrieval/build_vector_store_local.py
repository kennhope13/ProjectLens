"""Build a local FAISS index without embedding API quota."""
import json
from pathlib import Path

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

from project_lens.paths import PROCESSED_DIR, VECTOR_STORE_DIR


CHUNKS = PROCESSED_DIR / "chunks.jsonl"
STORE = VECTOR_STORE_DIR
MODEL_NAME = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"


def main():
    with CHUNKS.open(encoding="utf-8") as file:
        records = [json.loads(line) for line in file if line.strip()]
    if not records:
        raise ValueError("No chunks found. Run chunk_documents_all_fixed.py first.")

    model = SentenceTransformer(MODEL_NAME)
    texts = [record["page_content"] for record in records]
    vectors = model.encode(texts, batch_size=32, show_progress_bar=True, normalize_embeddings=True)
    vectors = np.asarray(vectors, dtype=np.float32)

    STORE.mkdir(parents=True, exist_ok=True)
    index = faiss.IndexFlatIP(vectors.shape[1])
    index.add(vectors)
    faiss.write_index(index, str(STORE / "index.faiss"))
    (STORE / "records.json").write_text(json.dumps(records, ensure_ascii=False), encoding="utf-8")
    (STORE / "model_name.txt").write_text(MODEL_NAME, encoding="utf-8")
    print(f"Saved {len(records)} local vectors to: {STORE}")


if __name__ == "__main__":
    main()
