"""PostgreSQL storage for RAG chunks and pgvector embeddings."""

from __future__ import annotations

import json
import os
from typing import Any, Iterable

import psycopg
from pgvector import Vector
from pgvector.psycopg import register_vector
from psycopg.types.json import Jsonb


DATABASE_URL = os.getenv(
    "DATABASE_URL",
    "postgresql://projectlens:projectlens@localhost:5432/projectlens",
)
EMBEDDING_DIMENSIONS = 384


def connect(initialize: bool = False) -> psycopg.Connection:
    connection = psycopg.connect(DATABASE_URL)
    if initialize:
        connection.execute("CREATE EXTENSION IF NOT EXISTS vector")
        connection.commit()
    register_vector(connection)
    return connection


def ensure_schema(connection: psycopg.Connection) -> None:
    connection.execute(
        f"""
        CREATE TABLE IF NOT EXISTS rag_chunks (
            id TEXT PRIMARY KEY,
            content TEXT NOT NULL,
            metadata JSONB NOT NULL DEFAULT '{{}}'::jsonb,
            embedding VECTOR({EMBEDDING_DIMENSIONS}) NOT NULL,
            created_at TIMESTAMPTZ NOT NULL DEFAULT now()
        )
        """
    )
    connection.execute(
        """
        CREATE TABLE IF NOT EXISTS rag_settings (
            key TEXT PRIMARY KEY,
            value TEXT NOT NULL
        )
        """
    )
    connection.execute(
        """
        CREATE INDEX IF NOT EXISTS rag_chunks_embedding_hnsw_idx
        ON rag_chunks USING hnsw (embedding vector_cosine_ops)
        """
    )


def replace_chunks(
    records: list[dict[str, Any]], embeddings: Iterable[Any], model_name: str
) -> None:
    with connect(initialize=True) as connection:
        ensure_schema(connection)
        connection.commit()
        with connection.transaction():
            connection.execute("DELETE FROM rag_chunks")
            with connection.cursor() as cursor:
                cursor.executemany(
                    """
                    INSERT INTO rag_chunks (id, content, metadata, embedding)
                    VALUES (%s, %s, %s, %s)
                    """,
                    [
                        (
                            record["id"],
                            record["page_content"],
                            Jsonb(json.loads(json.dumps(record.get("metadata", {}), default=str))),
                            Vector(embedding),
                        )
                        for record, embedding in zip(records, embeddings, strict=True)
                    ],
                )
            connection.execute(
                """
                INSERT INTO rag_settings (key, value)
                VALUES ('embedding_model', %s)
                ON CONFLICT (key) DO UPDATE SET value = EXCLUDED.value
                """,
                (model_name,),
            )


def load_chunk_records() -> list[dict[str, Any]]:
    with connect() as connection:
        rows = connection.execute(
            "SELECT id, content, metadata FROM rag_chunks ORDER BY id"
        ).fetchall()
    return [
        {"id": chunk_id, "page_content": content, "metadata": metadata}
        for chunk_id, content, metadata in rows
    ]


def load_embedding_model(default: str) -> str:
    with connect() as connection:
        row = connection.execute(
            "SELECT value FROM rag_settings WHERE key = 'embedding_model'"
        ).fetchone()
    return row[0] if row else default


def search_chunk_ids(embedding: Any, limit: int = 8) -> list[str]:
    vector = Vector(embedding)
    with connect() as connection:
        rows = connection.execute(
            """
            SELECT id
            FROM rag_chunks
            ORDER BY embedding <=> %s
            LIMIT %s
            """,
            (vector, limit),
        ).fetchall()
    return [row[0] for row in rows]
