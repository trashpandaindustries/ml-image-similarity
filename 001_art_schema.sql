-- Art similarity schema (dedicated `art` schema on the shared Supabase instance).
-- Idempotent: safe to re-run. Target: Postgres 17, pgvector 0.8.x.
--
-- Not exposed through PostgREST on purpose: the engine/API connect to Postgres
-- directly. RLS is enabled with no policies as defence in depth, so even if the
-- schema were exposed later, anon/authenticated get nothing by default.

create schema if not exists art;

-- Supabase installs extensions in `extensions`. If this already exists elsewhere
-- (check: select extnamespace::regnamespace from pg_extension where extname='vector')
-- adjust the search_path lines below and DB_SEARCH_PATH in the app.
create extension if not exists vector with schema extensions;

set search_path = art, extensions, public;

-- --------------------------------------------------------------------------
-- Relational data
-- --------------------------------------------------------------------------
create table if not exists art.artists (
  id    bigserial primary key,
  slug  text not null unique,      -- folder name, lower-cased (the stable key)
  name  text not null              -- display name derived from the folder
);

create table if not exists art.artworks (
  id         bigserial primary key,
  path       text not null unique, -- POSIX path relative to dataset root
  style      text,
  artist_id  bigint references art.artists(id) on delete set null,
  title      text,
  size       bigint,               -- incremental-update fingerprint
  mtime      bigint,               -- (same semantics as manifest.csv)
  created_at timestamptz not null default now()
);
create index if not exists artworks_artist_idx on art.artworks (artist_id);
create index if not exists artworks_style_idx  on art.artworks (style);

-- --------------------------------------------------------------------------
-- Vectors. One row per (artwork, model); dimension lives with the model row so
-- model swaps / dynamic dims never touch artwork metadata.
-- --------------------------------------------------------------------------
create table if not exists art.embedding_models (
  id         serial primary key,
  name       text not null,
  pretrained text not null,
  dim        int  not null check (dim > 0),
  meta       jsonb not null default '{}'::jsonb,
  unique (name, pretrained)
);

create table if not exists art.artwork_embeddings (
  artwork_id bigint not null references art.artworks(id)        on delete cascade,
  model_id   int    not null references art.embedding_models(id) on delete cascade,
  embedding  vector not null,      -- untyped: dim is per model (see ensure_hnsw)
  primary key (artwork_id, model_id)
);
create index if not exists artwork_embeddings_model_idx
  on art.artwork_embeddings (model_id);

-- --------------------------------------------------------------------------
-- Per-model partial HNSW index.
-- HNSW needs a fixed dimension, so we index a typed cast of the untyped column
-- and restrict the index to one model. Queries MUST use the identical
-- expression `embedding::vector(<dim>)` and a literal `model_id = <id>` for the
-- planner to use it (the app does this).
-- --------------------------------------------------------------------------
create or replace function art.ensure_hnsw(p_model_id int, p_dim int)
returns void
language plpgsql
set search_path = art, extensions, public
as $$
begin
  execute format(
    'create index if not exists %I on art.artwork_embeddings '
    'using hnsw ((embedding::vector(%s)) vector_cosine_ops) '
    'with (m = 16, ef_construction = 64) where model_id = %s',
    'ae_hnsw_m' || p_model_id, p_dim, p_model_id);
end
$$;

-- --------------------------------------------------------------------------
-- Lock down
-- --------------------------------------------------------------------------
alter table art.artists            enable row level security;
alter table art.artworks           enable row level security;
alter table art.embedding_models   enable row level security;
alter table art.artwork_embeddings enable row level security;

revoke all on schema art from anon, authenticated;
revoke all on all tables    in schema art from anon, authenticated;
revoke all on all sequences in schema art from anon, authenticated;
revoke all on all functions in schema art from anon, authenticated;
