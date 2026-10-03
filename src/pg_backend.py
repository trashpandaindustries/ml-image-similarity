# src/pg_backend.py
import psycopg
from psycopg_pool import ConnectionPool
from pgvector.psycopg import register_vector

class PgVectorBackend:
    def __init__(self, dsn: str, model_name: str, pretrained: str, dim: int):
        self.pool = ConnectionPool(dsn, configure=register_vector, min_size=1, max_size=8)
        self.dim = int(dim)               # from DB/config, never user input
        self.model_id = self._ensure_model(model_name, pretrained, dim)

    def search(self, vector, k, *, style=None, artist=None, exclude_path=None):
        d = self.dim
        sql = f"""
          select a.path, a.style, ar.name, a.title,
                 1 - (e.embedding::vector({d}) <=> %(q)s::vector({d})) as score
          from artwork_embeddings e
          join artworks a on a.id = e.artwork_id
          left join artists ar on ar.id = a.artist_id
          where e.model_id = %(m)s
            and (%(style)s::text is null or a.style = %(style)s)
            and (%(artist)s::text is null or ar.slug = %(artist)s)
            and (%(ex)s::text is null or a.path <> %(ex)s)
          order by e.embedding::vector({d}) <=> %(q)s::vector({d})
          limit %(k)s"""
        with self.pool.connection() as conn:
            conn.execute("set local hnsw.iterative_scan = relaxed_order")
            rows = conn.execute(sql, dict(q=vector, m=self.model_id, style=style,
                                          artist=artist, ex=exclude_path, k=k)).fetchall()
        return [...]  # map to SearchResult