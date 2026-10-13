"""Embed prepared chunks and store them in PostgreSQL with pgvector."""
import json

import numpy as np
from sentence_transformers import SentenceTransformer

from project_lens.paths import PROCESSED_DIR
from project_lens.storage.postgres import EMBEDDING_DIMENSIONS, replace_chunks


CHUNKS = PROCESSED_DIR / "chunks.jsonl"
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
    if vectors.shape[1] != EMBEDDING_DIMENSIONS:
        raise ValueError(
            f"Embedding dimension {vectors.shape[1]} does not match PostgreSQL schema "
            f"dimension {EMBEDDING_DIMENSIONS}. Update the schema before changing embedding models."
        )

    replace_chunks(records, vectors, MODEL_NAME)
    print(f"Saved {len(records)} chunks and vectors to PostgreSQL using pgvector.")


if __name__ == "__main__":
    main()
