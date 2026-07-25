"""Generate a self-contained HTML (and optional PNG) similarity demo.

Produces a portable ``demo.html`` in which each query image is shown alongside its
top-5 visually similar matches and their cosine similarity scores. Thumbnails are
embedded as base64 data URIs so the file renders anywhere with no server, no
dataset access and no external requests - ideal for reviewers.

A ``--png`` option additionally renders a flat contact-sheet image. Because code
hosts (e.g. GitHub) render images inline in Markdown but not HTML pages, the PNG
lets a reviewer preview retrieval results directly in the README.

    python demo/generate_demo.py --num-queries 6 --output demo/demo.html
    python demo/generate_demo.py --num-queries 4 --png demo/preview.png
    python demo/generate_demo.py --query "/path/to/image.jpg"
"""

from __future__ import annotations

import argparse
import base64
import html
import io
import logging
import random
import sys
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import Config  # noqa: E402
from src.logging_utils import configure_logging  # noqa: E402
from src.search import InvalidQueryImageError, SearchResult, SimilarityEngine  # noqa: E402

logger = logging.getLogger("src.demo")

_THUMB_PX = 240


def _thumb_data_uri(path: Path, size: int = _THUMB_PX) -> str:
    """Return a base64 JPEG data URI for a downscaled copy of ``path``."""
    try:
        image = Image.open(path)
        image.draft("RGB", (size, size))
        image = image.convert("RGB")
        image.thumbnail((size, size), Image.LANCZOS)
        buffer = io.BytesIO()
        image.save(buffer, format="JPEG", quality=82)
        encoded = base64.b64encode(buffer.getvalue()).decode("ascii")
        return f"data:image/jpeg;base64,{encoded}"
    except Exception as exc:  # noqa: BLE001
        logger.warning("Could not render thumbnail for %s (%s)", path, exc)
        return ""


def _card(uri: str, caption_html: str, is_query: bool) -> str:
    cls = "card query" if is_query else "card"
    return (
        f'<figure class="{cls}">'
        f'<div class="imgwrap"><img loading="lazy" src="{uri}" alt=""></div>'
        f"<figcaption>{caption_html}</figcaption></figure>"
    )


def _result_caption(result: SearchResult) -> str:
    style = html.escape(result.style.replace("_", " "))
    artist = html.escape(result.artist.title()) if result.artist else "Unknown"
    return (
        f'<span class="score">{result.score:.3f}</span>'
        f'<span class="artist">{artist}</span>'
        f'<span class="style">{style}</span>'
    )


def _query_row(engine: SimilarityEngine, query_path: Path, top_k: int) -> str:
    try:
        results = engine.search(query_path, k=top_k)
    except (FileNotFoundError, InvalidQueryImageError) as exc:
        logger.warning("Skipping query %s: %s", query_path, exc)
        return (
            '<section class="row"><div class="query-col">'
            f'{_card(_thumb_data_uri(query_path), "<span class=\'label\'>SKIPPED</span>", True)}'
            f'</div><div class="arrow">&rarr;</div><div class="results-col">'
            f'<p style="color:var(--muted)">{html.escape(str(exc))}</p></div></section>'
        )
    q_style = ""
    try:
        rel = query_path.resolve().relative_to(engine._config.dataset_root.resolve())  # noqa: SLF001
        q_style = rel.parts[0].replace("_", " ")
    except ValueError:
        pass
    query_caption = (
        f'<span class="label">QUERY</span>'
        f'<span class="style">{html.escape(q_style)}</span>'
    )
    query_card = _card(_thumb_data_uri(query_path), query_caption, is_query=True)
    result_cards = "".join(_card(_thumb_data_uri(Path(r.abspath)), _result_caption(r), False)
                           for r in results)
    return (
        '<section class="row">'
        f'<div class="query-col">{query_card}</div>'
        '<div class="arrow">&rarr;</div>'
        f'<div class="results-col">{result_cards}</div>'
        "</section>"
    )


def build_html(engine: SimilarityEngine, query_paths: list[Path], top_k: int) -> str:
    """Render the full HTML document for the given queries."""
    rows = "\n".join(_query_row(engine, p, top_k) for p in query_paths)
    meta = engine.meta
    subtitle = (
        f"{meta.get('model_name')} / {meta.get('pretrained')} &middot; "
        f"{engine.size:,} images &middot; {engine.backend} &middot; "
        "cosine similarity"
    )
    return _TEMPLATE.format(subtitle=subtitle, rows=rows, k=top_k)


_FONT_CANDIDATES: tuple[str, ...] = (
    "DejaVuSans.ttf",
    "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf",
    "C:/Windows/Fonts/arial.ttf",
    "/Library/Fonts/Arial.ttf",
)


def _load_font(size: int):
    """Return a scalable TrueType font, falling back to Pillow's bitmap default.

    A crisp TrueType face is used when one can be located; on minimal systems
    without any of the candidate fonts the bitmap default keeps the demo working
    (at a fixed small size) rather than failing.
    """
    for candidate in _FONT_CANDIDATES:
        try:
            return ImageFont.truetype(candidate, size)
        except OSError:
            continue
    try:
        return ImageFont.load_default(size=size)
    except TypeError:  # older Pillow: load_default() takes no size argument
        return ImageFont.load_default()


def _fit_thumb(path: Path, box: int) -> Image.Image:
    """Decode an image and center-crop it to fill a ``box`` x ``box`` square."""
    image = Image.open(path)
    image.draft("RGB", (box * 2, box * 2))
    image = image.convert("RGB")
    width, height = image.size
    scale = box / min(width, height)
    resized = image.resize((round(width * scale), round(height * scale)), Image.LANCZOS)
    left = (resized.width - box) // 2
    top = (resized.height - box) // 2
    return resized.crop((left, top, left + box, top + box))


def build_png(
    engine: SimilarityEngine, query_paths: list[Path], top_k: int, box: int = 168
) -> Image.Image:
    """Render a flat contact-sheet image: each query row with its top-k + scores.

    Args:
        engine: Loaded similarity engine.
        query_paths: Query images to render (one row each).
        top_k: Number of matches per query.
        box: Thumbnail square size in pixels.

    Returns:
        The rendered RGB image.
    """
    rows: list[tuple[Path, list[SearchResult]]] = []
    for query_path in query_paths:
        try:
            rows.append((query_path, engine.search(query_path, k=top_k)))
        except (FileNotFoundError, InvalidQueryImageError) as exc:
            logger.warning("Skipping query %s: %s", query_path, exc)
    if not rows:
        raise RuntimeError("No valid query produced results for the PNG demo.")

    pad, gap, arrow_w, cap_h, header_h = 16, 12, 48, 30, 60
    row_h = box + cap_h + pad
    width = pad + box + arrow_w + top_k * (box + gap) + pad
    height = header_h + len(rows) * row_h + pad

    bg, ink, muted, accent, score_col, edge = (
        (246, 247, 249), (27, 31, 36), (91, 100, 112),
        (47, 111, 237), (26, 127, 55), (210, 214, 219),
    )
    canvas = Image.new("RGB", (width, height), bg)
    draw = ImageDraw.Draw(canvas)
    f_title, f_score, f_cap = _load_font(26), _load_font(19), _load_font(15)

    draw.text(
        (pad, 18),
        f"Visual Image Similarity  -  query -> top-{top_k} matches (cosine score)",
        fill=ink, font=f_title,
    )

    def paste(image: Image.Image, cx: int, cy: int, border: tuple[int, int, int]) -> None:
        draw.rectangle([cx, cy, cx + box, cy + box], fill=(0, 0, 0))
        canvas.paste(image, (cx + (box - image.width) // 2, cy + (box - image.height) // 2))
        draw.rectangle([cx, cy, cx + box - 1, cy + box - 1], outline=border, width=3)

    def centered(text: str, cx: int, cy: int, font: ImageFont.ImageFont, fill) -> None:
        draw.text((cx + (box - draw.textlength(text, font=font)) / 2, cy), text, fill=fill, font=font)

    for i, (query_path, results) in enumerate(rows):
        y = header_h + i * row_h
        paste(_fit_thumb(query_path, box), pad, y, accent)
        centered("QUERY", pad, y + box + 7, f_cap, accent)
        draw.text((pad + box + 10, y + box // 2 - 16), "->", fill=muted, font=f_title)
        for j, result in enumerate(results):
            cx = pad + box + arrow_w + j * (box + gap)
            paste(_fit_thumb(Path(result.abspath), box), cx, y, edge)
            centered(f"{result.score:.3f}", cx, y + box + 5, f_score, score_col)
    return canvas


def _choose_queries(engine: SimilarityEngine, query: str | None, n: int) -> list[Path]:
    if query:
        return [Path(query)]
    sample = engine._manifest.sample(n, random_state=random.randint(0, 10_000))  # noqa: SLF001
    root = engine._config.dataset_root  # noqa: SLF001
    return [root / p for p in sample["path"]]


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Generate a static HTML similarity demo.")
    parser.add_argument("--query", type=str, default=None, help="A specific query image.")
    parser.add_argument("--num-queries", type=int, default=6, help="Random queries if none given.")
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--output", type=str, default=None, help="Output HTML path.")
    parser.add_argument("--png", type=str, default=None, help="Also render a PNG contact sheet here.")
    parser.add_argument("--dataset-root", type=str, default=None)
    parser.add_argument("--embeddings-dir", type=str, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--log-level", type=str, default=None)
    return parser


def main(argv: list[str] | None = None) -> None:
    """CLI entry point for the static demo generator."""
    args = _build_parser().parse_args(argv)
    configure_logging(args.log_level)
    if args.seed is not None:
        random.seed(args.seed)
    config = Config.from_env(args.dataset_root).merged_with(
        embeddings_dir=Path(args.embeddings_dir) if args.embeddings_dir else None,
    )
    engine = SimilarityEngine(config)
    queries = _choose_queries(engine, args.query, args.num_queries)

    wrote_png = False
    if args.png:
        png_path = Path(args.png)
        png_path.parent.mkdir(parents=True, exist_ok=True)
        build_png(engine, queries, args.top_k).save(png_path, optimize=True)
        logger.info("Wrote PNG demo with %d queries to %s", len(queries), png_path)
        print(png_path)
        wrote_png = True

    # Write HTML when explicitly requested, or by default when no PNG was asked for.
    if args.output or not wrote_png:
        output = Path(args.output) if args.output else Path(__file__).parent / "output" / "demo.html"
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(build_html(engine, queries, args.top_k), encoding="utf-8")
        logger.info("Wrote HTML demo with %d queries to %s", len(queries), output)
        print(output)


_TEMPLATE = """<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Visual Image Similarity - Top {k} Results</title>
<style>
  :root {{
    --bg: #0f1115; --panel: #171a21; --edge: #262b36;
    --fg: #e8eaed; --muted: #9aa4b2; --accent: #6ea8fe; --score: #7ee787;
  }}
  @media (prefers-color-scheme: light) {{
    :root {{ --bg:#f6f7f9; --panel:#fff; --edge:#e3e6ea; --fg:#1b1f24;
             --muted:#5b6470; --accent:#2f6fed; --score:#1a7f37; }}
  }}
  * {{ box-sizing: border-box; }}
  body {{ margin:0; background:var(--bg); color:var(--fg);
         font-family:-apple-system,Segoe UI,Roboto,Helvetica,Arial,sans-serif; }}
  header {{ padding:28px 24px 8px; }}
  h1 {{ margin:0 0 6px; font-size:20px; font-weight:650; letter-spacing:.2px; }}
  .subtitle {{ color:var(--muted); font-size:13px; }}
  main {{ padding:16px 24px 40px; display:flex; flex-direction:column; gap:16px; }}
  .row {{ display:flex; align-items:center; gap:14px; background:var(--panel);
          border:1px solid var(--edge); border-radius:14px; padding:14px; overflow-x:auto; }}
  .query-col {{ flex:0 0 auto; }}
  .results-col {{ display:flex; gap:12px; flex:1 1 auto; }}
  .arrow {{ color:var(--muted); font-size:26px; flex:0 0 auto; }}
  .card {{ margin:0; width:170px; flex:0 0 auto; text-align:center; }}
  .card.query .imgwrap {{ outline:2px solid var(--accent); }}
  .imgwrap {{ height:170px; border-radius:10px; overflow:hidden; background:#000;
              display:flex; align-items:center; justify-content:center; }}
  .imgwrap img {{ width:100%; height:100%; object-fit:cover; display:block; }}
  figcaption {{ display:flex; flex-direction:column; gap:1px; margin-top:7px;
                font-size:12px; line-height:1.3; }}
  .label {{ color:var(--accent); font-weight:700; letter-spacing:.6px; font-size:11px; }}
  .score {{ color:var(--score); font-weight:700; font-variant-numeric:tabular-nums; }}
  .artist {{ color:var(--fg); font-weight:550; white-space:nowrap; overflow:hidden;
             text-overflow:ellipsis; }}
  .style {{ color:var(--muted); }}
</style>
</head>
<body>
  <header>
    <h1>Visual Image Similarity &mdash; Top {k} Results</h1>
    <div class="subtitle">{subtitle}</div>
  </header>
  <main>
{rows}
  </main>
</body>
</html>"""


if __name__ == "__main__":
    main()
