"""Interactive Streamlit demo for the image similarity engine.

Launch from the repository root::

    streamlit run demo/app.py

Reviewers can either upload an image or draw a random one from the corpus, then
inspect the top-5 visually similar paintings with their cosine similarity scores.
The heavy objects (index + model) are cached across reruns so interaction is
instant after the first load.
"""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path

import streamlit as st
from PIL import Image, UnidentifiedImageError

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import Config  # noqa: E402
from src.logging_utils import configure_logging  # noqa: E402
from src.search import InvalidQueryImageError, SimilarityEngine  # noqa: E402

st.set_page_config(page_title="WikiArt Visual Similarity", page_icon="🖼️", layout="wide")


@st.cache_resource(show_spinner="Loading index and model…")
def load_engine() -> SimilarityEngine:
    """Load the similarity engine once and cache it for the session."""
    configure_logging("WARNING")
    config = Config.from_env()
    return SimilarityEngine(config)


def _render_results(engine: SimilarityEngine, query_path: Path, top_k: int) -> None:
    try:
        results = engine.search(query_path, k=top_k)
    except (FileNotFoundError, InvalidQueryImageError) as exc:
        st.error(str(exc))
        return

    left, right = st.columns([1, 3], gap="large")
    with left:
        st.markdown("#### Query")
        st.image(str(query_path), width='stretch')
    with right:
        st.markdown(f"#### Top {top_k} similar")
        cols = st.columns(top_k)
        for col, result in zip(cols, results):
            with col:
                st.image(result.abspath, width='stretch')
                st.markdown(
                    f"**{result.score:.3f}**  \n"
                    f"{result.artist.title() or 'Unknown'}  \n"
                    f"<span style='color:#888'>{result.style.replace('_', ' ')}</span>",
                    unsafe_allow_html=True,
                )


def main() -> None:
    """Render the Streamlit application."""
    st.title("WikiArt Visual Similarity Search")

    try:
        engine = load_engine()
    except FileNotFoundError:
        st.error("No embedding index found. Run `python -m src.embed` first.")
        st.stop()

    meta = engine.meta
    st.caption(
        f"{meta.get('model_name')} / {meta.get('pretrained')} · "
        f"{engine.size:,} images · {engine.backend} · cosine similarity"
    )

    top_k = st.sidebar.slider("Number of results", 3, 12, 5)
    mode = st.sidebar.radio("Query source", ["Random from corpus", "Upload an image"])

    if mode == "Random from corpus":
        if "query_rel" not in st.session_state or st.sidebar.button("🎲 New random query"):
            st.session_state["query_rel"] = engine._manifest.sample(1).iloc[0]["path"]  # noqa: SLF001
        query_path = engine._config.dataset_root / st.session_state["query_rel"]  # noqa: SLF001
        _render_results(engine, query_path, top_k)
    else:
        uploaded = st.file_uploader("Upload a query image", type=["jpg", "jpeg", "png", "webp"])
        if uploaded is None:
            st.info("Upload an image to search the corpus.")
            return
        suffix = Path(uploaded.name).suffix or ".jpg"
        try:
            with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
                Image.open(uploaded).convert("RGB").save(tmp, format="JPEG")
                tmp_path = Path(tmp.name)
        except (UnidentifiedImageError, OSError) as exc:
            st.error(f"Could not read the uploaded file as an image ({exc}).")
            return
        _render_results(engine, tmp_path, top_k)


if __name__ == "__main__":
    main()
