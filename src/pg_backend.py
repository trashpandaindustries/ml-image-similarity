"""Supabase/Postgres pgvector storage backend.

The backend uses a direct Postgres connection (not the Supabase REST API), which
keeps vector writes and nearest-neighbour searches in the database.  The schema
is created by ``001_art_schema.sql`` before this backend is used.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from pgvector.psycopg import register_vector
from psycopg import sql
from psycopg.types.json import Jsonb
from psycopg_pool import ConnectionPool

from .backend import ArtistVote, SearchResult, split_path
from .config import Config

class PgVectorBackend:
    """Read and write a single embedding model in Postgres/pgvector."""

    backend_name = "pgvector"

    def __init__(self, dsn: str, model_name: str, pretrained: str, dim: int, dataset_root: Path) -> None:
        self.dataset_root = Path(dataset_root)
        self._pool = ConnectionPool(dsn, configure=register_vector, min_size=1, max_size=8)
        self.dim = int(dim)
        if self.dim < 1:
            raise ValueError("Embedding dimension must be positive.")
        self.model_name = model_name
        self.pretrained = pretrained
        self.model_id = self._ensure_model()

    @classmethod
    def from_config(cls, config: Config) -> "PgVectorBackend":
        dsn = config.database_url
        if not dsn:
            raise ValueError(
                "Postgres backend requires DATABASE_URL (or SUPABASE_DB_URL) to be set."
            )
        return cls(dsn, config.model_name, config.pretrained, config.output_dim, config.dataset_root)

    def close(self) -> None:
        self._pool.close()

    def _ensure_model(self) -> int:
        with self._pool.connection() as conn, conn.cursor() as cur:
            cur.execute(
                """
                insert into art.embedding_models (name, pretrained, dim)
                values (%s, %s, %s)
                on conflict (name, pretrained) do update set name = excluded.name
                returning id, dim
                """,
                (self.model_name, self.pretrained, self.dim),
            )
            model_id, stored_dim = cur.fetchone()
            if stored_dim != self.dim:
                raise ValueError("Stored model dimension does not match the requested dimension.")
            cur.execute("select art.ensure_hnsw(%s, %s)", (model_id, self.dim))
            return int(model_id)

    @property
    def size(self) -> int:
        with self._pool.connection() as conn, conn.cursor() as cur:
            cur.execute("select count(*) from art.artwork_embeddings where model_id = %s", (self.model_id,))
            return int(cur.fetchone()[0])

    @property
    def meta(self) -> dict[str, Any]:
        with self._pool.connection() as conn, conn.cursor() as cur:
            cur.execute(
                "select name, pretrained, dim, meta from art.embedding_models where id = %s",
                (self.model_id,),
            )
            name, pretrained, dim, meta = cur.fetchone()
        return {"model_name": name, "pretrained": pretrained, "dim": dim, **(meta or {})}

    @staticmethod
    def _vector(vector: np.ndarray) -> np.ndarray:
        return np.ascontiguousarray(vector, dtype=np.float32).reshape(-1)

    def search(self, vector: np.ndarray, k: int, *, style: str | None = None,
               artist: str | None = None, exclude_path: str | None = None) -> list[SearchResult]:
        if k <= 0:
            return []
        query = sql.SQL("""
            select a.path, coalesce(a.style, ''), coalesce(ar.name, ''), coalesce(a.title, ''),
                   1 - (e.embedding::vector({dim}) <=> %(query)s::vector({dim})) as score
            from art.artwork_embeddings e
            join art.artworks a on a.id = e.artwork_id
            left join art.artists ar on ar.id = a.artist_id
            where e.model_id = %(model_id)s
              and (%(style)s::text is null or a.style = %(style)s)
              and (%(artist)s::text is null or ar.slug = %(artist)s)
              and (%(exclude_path)s::text is null or a.path <> %(exclude_path)s)
            order by e.embedding::vector({dim}) <=> %(query)s::vector({dim})
            limit %(limit)s
        """).format(dim=sql.Literal(self.dim))
        params = {"query": self._vector(vector), "model_id": self.model_id, "style": style,
                  "artist": artist.lower() if artist else None, "exclude_path": exclude_path, "limit": k}
        with self._pool.connection() as conn, conn.cursor() as cur:
            cur.execute("set local hnsw.iterative_scan = relaxed_order")
            cur.execute(query, params)
            rows = cur.fetchall()
        return [SearchResult(rank=i, path=row[0], style=row[1], artist=row[2], title=row[3],
                             score=float(row[4]), abspath=str(self.dataset_root / row[0])) for i, row in enumerate(rows, 1)]

    def identify(self, vector: np.ndarray, k: int = 50, top: int = 5) -> list[ArtistVote]:
        votes: dict[str, dict[str, Any]] = {}
        for hit in self.search(vector, k):
            slug = split_path(hit.path)[1]
            if not slug:
                continue
            entry = votes.setdefault(slug, {"name": hit.artist, "weight": 0.0, "votes": 0, "best": -1.0})
            entry["weight"] += hit.score
            entry["votes"] += 1
            entry["best"] = max(entry["best"], hit.score)
        return [ArtistVote(slug=slug, **value) for slug, value in
                sorted(votes.items(), key=lambda item: item[1]["weight"], reverse=True)[:top]]

    def sample_paths(self, n: int, seed: str | int = 0) -> list[str]:
        with self._pool.connection() as conn, conn.cursor() as cur:
            cur.execute("""
                select a.path from art.artwork_embeddings e join art.artworks a on a.id = e.artwork_id
                where e.model_id = %s order by md5(a.path || %s) limit %s
            """, (self.model_id, str(seed), n))
            return [row[0] for row in cur.fetchall()]

    def sync(self, embeddings: np.ndarray, manifest: pd.DataFrame, meta: dict[str, Any]) -> None:
        """Upsert a completed local index and remove stale vectors for this model."""
        if embeddings.shape[0] != len(manifest):
            raise ValueError("embeddings and manifest length mismatch on sync.")
        if embeddings.shape[1] != self.dim:
            raise ValueError(f"Expected {self.dim}-dimensional embeddings, got {embeddings.shape[1]}.")
        paths = manifest["path"].astype(str).tolist()
        with self._pool.connection() as conn, conn.cursor() as cur:
            cur.execute("update art.embedding_models set meta = %s::jsonb where id = %s", (Jsonb(meta), self.model_id))
            for row, vector in zip(manifest.itertuples(index=False), embeddings, strict=True):
                path = str(row.path)
                _, slug, name, derived_title = split_path(path)
                artist_name = str(row.artist) if pd.notna(row.artist) and str(row.artist) else name
                title = str(row.title) if pd.notna(row.title) and str(row.title) else derived_title
                artist_id = None
                if slug:
                    cur.execute("""insert into art.artists (slug, name) values (%s, %s)
                                   on conflict (slug) do update set name = excluded.name returning id""", (slug, artist_name))
                    artist_id = cur.fetchone()[0]
                cur.execute("""insert into art.artworks (path, style, artist_id, title, size, mtime)
                               values (%s, %s, %s, %s, %s, %s)
                               on conflict (path) do update set style=excluded.style, artist_id=excluded.artist_id,
                                   title=excluded.title, size=excluded.size, mtime=excluded.mtime returning id""",
                            (path, str(row.style) if pd.notna(row.style) else None, artist_id, title, int(row.size), int(row.mtime)))
                artwork_id = cur.fetchone()[0]
                cur.execute("""insert into art.artwork_embeddings (artwork_id, model_id, embedding)
                               values (%s, %s, %s) on conflict (artwork_id, model_id)
                               do update set embedding = excluded.embedding""", (artwork_id, self.model_id, self._vector(vector)))
            if paths:
                cur.execute("""delete from art.artwork_embeddings e using art.artworks a
                               where e.artwork_id = a.id and e.model_id = %s and not (a.path = any(%s))""",
                            (self.model_id, paths))
            else:
                cur.execute("delete from art.artwork_embeddings where model_id = %s", (self.model_id,))