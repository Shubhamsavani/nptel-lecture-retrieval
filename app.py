"""
app.py  —  NPTEL Lecture Retrieval System — Production Demo
============================================================
Clean search interface for the NPTEL lecture retrieval system.

Features:
  - Natural language query input
  - Intent dropdown (Conceptual / Theoretical / Code)
  - Result cards with YouTube deep-link buttons
  - Transcript snippets and OCR text

Running
-------
    streamlit run app.py
    streamlit run app.py --server.port 8502

Environment variables (from .env):
    PROJECT_ROOT      — project root directory
    EMBEDDING_DEVICE  — "cuda" or "cpu"
"""

import os
import sys
import time
from pathlib import Path

import streamlit as st

# ── path setup ────────────────────────────────────────────────────────────────
_here = Path(__file__).resolve().parent
for _candidate in [_here, _here / "src" / "retrieval", _here / "src"]:
    if (_candidate / "retriever.py").exists():
        sys.path.insert(0, str(_candidate))
        break

# ── load .env ─────────────────────────────────────────────────────────────────
try:
    from dotenv import load_dotenv
    for _c in [_here, _here.parent]:
        if (_c / ".env").exists():
            load_dotenv(_c / ".env")
            break
except ImportError:
    pass

# ── page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="NPTEL Lecture Search",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ── CSS ───────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
  @import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;500&family=IBM+Plex+Sans:wght@300;400;500;600&display=swap');

  html, body, [class*="css"] { font-family: 'IBM Plex Sans', sans-serif; }

  .app-header {
    padding: 1.8rem 0 1.2rem 0;
    border-bottom: 2px solid #1a1a2e;
    margin-bottom: 1.8rem;
  }
  .app-title {
    font-size: 2rem;
    font-weight: 700;
    color: #ffffff;
    letter-spacing: -0.5px;
    margin: 0;
  }
  .app-subtitle {
    font-size: 0.82rem;
    color: #94a3b8;
    margin-top: 5px;
    font-family: 'IBM Plex Mono', monospace;
  }

  .result-card {
    background: #ffffff;
    border: 1px solid #e5e7eb;
    border-left: 4px solid #1a1a2e;
    border-radius: 8px;
    padding: 1.2rem 1.4rem;
    margin-bottom: 1rem;
    transition: box-shadow 0.15s;
  }
  .result-card:hover { box-shadow: 0 4px 16px rgba(0,0,0,0.09); }
  .result-card.code-card        { border-left-color: #f59e0b; }
  .result-card.theory-card      { border-left-color: #6366f1; }
  .result-card.conceptual-card  { border-left-color: #10b981; }

  .rank-badge {
    display: inline-block;
    background: #1a1a2e;
    color: white;
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.68rem;
    padding: 2px 8px;
    border-radius: 3px;
    margin-right: 8px;
    font-weight: 500;
  }
  .course-tag {
    display: inline-block;
    background: #f3f4f6;
    color: #374151;
    font-size: 0.72rem;
    padding: 2px 9px;
    border-radius: 12px;
    margin-right: 6px;
    font-weight: 500;
  }
  .type-tag {
    display: inline-block;
    font-size: 0.72rem;
    padding: 2px 9px;
    border-radius: 12px;
    font-weight: 500;
  }
  .type-code        { background: #fef3c7; color: #92400e; }
  .type-theoretical { background: #ede9fe; color: #4c1d95; }
  .type-conceptual  { background: #d1fae5; color: #065f46; }

  .lecture-title {
    font-size: 1rem;
    font-weight: 600;
    color: #111827;
    margin: 0.5rem 0 0.2rem 0;
  }
  .timestamp-line {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.76rem;
    color: #6b7280;
    margin-bottom: 0.6rem;
  }
  .snippet {
    font-size: 0.86rem;
    color: #374151;
    line-height: 1.65;
    border-left: 2px solid #e5e7eb;
    padding-left: 10px;
    margin: 0.5rem 0 0.7rem 0;
    font-style: italic;
  }
  .play-btn {
    display: inline-block;
    background: #dc2626;
    color: white !important;
    text-decoration: none !important;
    padding: 6px 16px;
    border-radius: 5px;
    font-size: 0.8rem;
    font-weight: 600;
    letter-spacing: 0.2px;
  }
  .play-btn:hover { background: #b91c1c; }

  .score-line {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.68rem;
    color: #9ca3af;
    margin-top: 0.5rem;
  }

  .meta-bar {
    font-size: 0.8rem;
    color: #6b7280;
    margin: 0.8rem 0 1.2rem 0;
  }

  .no-results {
    text-align: center;
    padding: 4rem 2rem;
    color: #9ca3af;
    font-size: 0.9rem;
  }

  .stTextInput > div > div > input {
    font-size: 1rem;
    border-radius: 6px;
  }
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# Retriever loader — cached so models load once per session
# ─────────────────────────────────────────────────────────────────────────────

@st.cache_resource(show_spinner="Loading retrieval models…")
def load_retriever():
    try:
        import retriever as _ret
        _ret._get_embed_model()   # warm up embedding model
        return _ret
    except ImportError as e:
        st.error(f"Could not import retriever.py: {e}")
        st.stop()
    except FileNotFoundError as e:
        st.error(
            f"Index file not found: {e}\n\n"
            "Build the indexes first:\n"
            "  python embedder.py --strategy c3\n"
            "  python bm25_builder.py --strategy c3"
        )
        st.stop()


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

INTENT_OPTIONS = {
    "Conceptual  — how things work, intuition, examples": "conceptual",
    "Theoretical — definitions, proofs, complexity, analysis": "theoretical",
    "Code  — implementation, syntax, programming examples": "code",
}

TYPE_CSS = {
    "code":        ("code-card",       "type-tag type-code",        "💻 code"),
    "theoretical": ("theory-card",     "type-tag type-theoretical", "📐 theory"),
    "conceptual":  ("conceptual-card", "type-tag type-conceptual",  "💡 concept"),
}


def _fmt_time(seconds: float) -> str:
    m = int(seconds) // 60
    s = int(seconds) % 60
    return f"{m}:{s:02d}"


def render_card(result: dict) -> None:
    ctype              = result.get("content_type", "conceptual")
    card_cls, tag_cls, tag_label = TYPE_CSS.get(ctype, TYPE_CSS["conceptual"])

    transcript = result.get("transcript", "").replace("\n", " ").strip()
    snippet    = transcript[:240] + ("…" if len(transcript) > 240 else "")
    link       = result.get("youtube_deep_link") or result.get("youtube_url", "")
    start      = result.get("start_sec", 0)
    end        = result.get("end_sec", 0)
    score      = result.get("retrieval_score", 0)
    code_badge = "<span class='course-tag'>🎬 code segment</span>" \
                 if result.get("is_code_segment") else ""

    st.markdown(f"""
    <div class="result-card {card_cls}">
      <div>
        <span class="rank-badge">#{result['rank']}</span>
        <span class="course-tag">{result.get('course_name', '')}</span>
        <span class="{tag_cls}">{tag_label}</span>
        {code_badge}
      </div>
      <div class="lecture-title">{result.get('lecture_title', 'Unknown lecture')}</div>
      <div class="timestamp-line">
        ⏱ {_fmt_time(start)} → {_fmt_time(end)}
        &nbsp;·&nbsp; Lecture {result.get('lecture_number', '?')}
        &nbsp;·&nbsp; {result.get('instructor', '')}
      </div>
      <div class="snippet">"{snippet}"</div>
      {"<a class='play-btn' href='" + link + "' target='_blank'>▶ Play from here</a>" if link else ""}
      <div class="score-line">retrieval score: {score:.5f}</div>
    </div>
    """, unsafe_allow_html=True)

    ocr     = result.get("ocr_text", "").strip()
    full_tx = result.get("transcript", "").strip()
    if ocr or full_tx:
        with st.expander("Show slide text & full transcript"):
            if ocr and not result.get("ocr_failed"):
                st.markdown("**Slide OCR text:**")
                st.code(ocr.replace("\n---\n", "\n──────────\n"), language=None)
            if full_tx:
                st.markdown("**Full transcript:**")
                st.markdown(f"_{full_tx}_")


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    ret = load_retriever()

    # Header
    st.markdown("""
    <div class="app-header">
      <div class="app-title">🎓 NPTEL Lecture Search</div>
      <div class="app-subtitle">
        dense · sparse · reciprocal rank fusion · cross-encoder reranking · c3 slide-boundary index
      </div>
    </div>
    """, unsafe_allow_html=True)

    # Search controls
    col_query, col_intent, col_btn = st.columns([4, 2, 1])

    with col_query:
        query = st.text_input(
            "Query",
            placeholder="e.g.  how does binary search tree insertion work",
            label_visibility="collapsed",
        )

    with col_intent:
        intent_label = st.selectbox(
            "Intent",
            options=list(INTENT_OPTIONS.keys()),
            label_visibility="collapsed",
        )

    with col_btn:
        search_clicked = st.button("Search", type="primary", use_container_width=True)

    st.markdown(
        "<div style='font-size:0.76rem;color:#9ca3af;margin-top:3px'>"
        "Try: &nbsp;"
        "<code>BST insertion</code> &nbsp;·&nbsp;"
        "<code>how does virtual memory work</code> &nbsp;·&nbsp;"
        "<code>python code for bubble sort</code> &nbsp;·&nbsp;"
        "<code>proof of Dijkstra correctness</code>"
        "</div>",
        unsafe_allow_html=True,
    )

    st.divider()

    # Run search
    if (search_clicked or query) and query.strip():
        intent = INTENT_OPTIONS[intent_label]

        with st.spinner("Searching…"):
            t0 = time.time()
            try:
                results = ret.search(
                    query  = query.strip(),
                    intent = intent,
                    top_k  = 5,
                )
            except FileNotFoundError as e:
                st.error(str(e))
                return
            latency = time.time() - t0

        if results:
            st.markdown(
                f"<div class='meta-bar'>"
                f"<b>{len(results)}</b> results &nbsp;·&nbsp; "
                f"<b>{latency:.2f}s</b> &nbsp;·&nbsp; "
                f"intent: <b>{intent}</b>"
                f"</div>",
                unsafe_allow_html=True,
            )

            result_col, stat_col = st.columns([3, 1])

            with result_col:
                for r in results:
                    render_card(r)

            with stat_col:
                st.markdown("#### Stats")
                st.metric("Latency", f"{latency:.2f}s")
                st.metric("Results", len(results))
                st.metric("Intent", intent.capitalize())
                st.markdown("---")
                st.markdown(
                    "<div style='font-size:0.74rem;color:#6b7280;line-height:1.7'>"
                    "Pipeline:<br>"
                    "① FAISS dense<br>"
                    "② BM25 sparse<br>"
                    "③ RRF fusion<br>"
                    "④ Content boost<br>"
                    "⑤ Cross-encoder rerank<br>"
                    "⑥ Deduplication"
                    "</div>",
                    unsafe_allow_html=True,
                )
        else:
            st.markdown(
                "<div class='no-results'>"
                "No results found — try different keywords or a broader query."
                "</div>",
                unsafe_allow_html=True,
            )
    else:
        # Landing state
        st.markdown(
            "<div style='text-align:center;padding:2rem 0;color:#9ca3af;font-size:0.88rem'>"
            "Enter a query above to search across NPTEL lecture videos."
            "</div>",
            unsafe_allow_html=True,
        )


if __name__ == "__main__":
    main()