import streamlit as st
import json
import time
import sys
from pathlib import Path

# ─────────────────────────────────────────────
# Import retriever + queries
# ─────────────────────────────────────────────
_here = Path(__file__).resolve().parent
for _candidate in [_here, _here / "src", _here / "src/retrieval"]:
    if (_candidate / "retriever.py").exists():
        sys.path.insert(0, str(_candidate))
        break

import retriever
from evaluator import EVAL_QUERIES

# ─────────────────────────────────────────────
# Config
# ─────────────────────────────────────────────
ANNOT_FILE = Path("data/eval/annotations.jsonl")
TOP_K = 5

# ─────────────────────────────────────────────
# Session state
# ─────────────────────────────────────────────
if "idx" not in st.session_state:
    st.session_state.idx = 0

if "done" not in st.session_state:
    st.session_state.done = False

# ─────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────
def save_record(record):
    ANNOT_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(ANNOT_FILE, "a", encoding="utf-8") as f:
        f.write(json.dumps(record) + "\n")


def format_time(sec):
    m = int(sec) // 60
    s = int(sec) % 60
    return f"{m}:{s:02d}"


def next_query():
    st.session_state.idx += 1
    if st.session_state.idx >= len(EVAL_QUERIES):
        st.session_state.done = True
    st.rerun()


# ─────────────────────────────────────────────
# UI
# ─────────────────────────────────────────────
st.set_page_config(page_title="Annotation UI", layout="wide")

st.title("🧪 Evaluation Annotation Tool")

# Progress
total = len(EVAL_QUERIES)
current = st.session_state.idx

st.progress(current / total)
st.write(f"Query {current + 1} / {total}")

if st.session_state.done:
    st.success("✅ All queries completed!")
    st.stop()

# Current query
q = EVAL_QUERIES[current]

st.markdown(f"### Query ID: `{q['id']}`")
st.markdown(f"**Course:** `{q['expected_course']}`")
st.markdown(f"**Query:** {q['query']}")

# ─────────────────────────────────────────────
# Run retrieval
# ─────────────────────────────────────────────
with st.spinner("Retrieving results..."):
    t0 = time.time()
    results = retriever.search(
        query=q["query"],
        strategy="c3",  # use best strategy
        top_k=TOP_K,
        use_llm=False,
        use_rerank=True,
    )
    latency = time.time() - t0

st.caption(f"Retrieved in {latency:.2f}s")

# ─────────────────────────────────────────────
# Show results
# ─────────────────────────────────────────────
for i, r in enumerate(results):

    st.markdown("---")

    st.markdown(f"### Result {i+1}")

    st.write(f"**Course:** {r.get('course_name')}")
    st.write(f"**Lecture:** {r.get('lecture_title')}")
    st.write(f"**Lecture #**: {r.get('lecture_number')}")

    start = r.get("start_sec", 0)
    end   = r.get("end_sec", 0)

    st.write(f"⏱ {format_time(start)} → {format_time(end)}")

    snippet = r.get("transcript", "")[:200]
    st.write(f"> {snippet}...")

    link = r.get("youtube_deep_link", "")
    if link:
        st.markdown(f"[▶ Watch]({link})")

    # SELECT BUTTON
    if st.button(f"✅ Select THIS (Result {i+1})", key=f"select_{i}"):

        record = {
            "id": q["id"],
            "query": q["query"],
            "expected_course": q["expected_course"],
            "expected_lecture": r.get("lecture_number"),
            "expected_start_sec": r.get("start_sec"),
            "status": "selected"
        }

        save_record(record)

        st.success("Saved!")
        next_query()


# ─────────────────────────────────────────────
# NONE OPTION
# ─────────────────────────────────────────────
st.markdown("---")

if st.button("❌ None of these are correct"):

    record = {
        "id": q["id"],
        "query": q["query"],
        "expected_course": q["expected_course"],
        "status": "skipped"
    }

    save_record(record)

    st.warning("Marked as skipped")
    next_query()