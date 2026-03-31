import streamlit as st
import requests

# 🔌 Flask API endpoint
API_URL = "http://127.0.0.1:5000/search"

# 🎨 Page config
st.set_page_config(
    page_title="Lecture Retrieval",
    layout="wide"
)

# 🏷 Title
st.title("🎓 Multimodal Lecture Retrieval System")
st.markdown("Search lecture content using transcript + slides + AI")

# 🔍 Input section
query = st.text_input("🔎 Enter your query")

col1, col2, col3 = st.columns(3)

with col1:
    strategy = st.selectbox(
        "Chunking Strategy",
        ["c1", "c2", "c3"],
        index=2
    )

with col2:
    use_llm = st.checkbox("Use LLM (Ollama)", value=True)

with col3:
    top_k = st.slider("Top K Results", 1, 5, 5)

# 🚀 Search button
if st.button("Search"):

    if not query.strip():
        st.warning("Please enter a query")
    else:
        with st.spinner("🔍 Searching..."):

            try:
                response = requests.post(
                    API_URL,
                    json={
                        "query": query,
                        "strategy": strategy,
                        "use_llm": use_llm
                    },
                    timeout=60
                )

                if response.status_code != 200:
                    st.error(f"API Error: {response.text}")
                else:
                    data = response.json()
                    results = data.get("results", [])

                    # 🎉 Header
                    st.success(f"Found {len(results)} results")

                    # 📊 Display results
                    for i, r in enumerate(results[:top_k]):

                        with st.container():
                            st.markdown(f"## 🔹 Result {i+1}")

                            # 🎥 YouTube link
                            if r.get("youtube_link"):
                                st.markdown(
                                    f"🎥 [Watch Lecture]({r['youtube_link']})"
                                )

                            # 🧠 Transcript
                            # st.markdown("**🧠 Transcript:**")
                            # st.write(r.get("transcript", "")[:400] + "...")

                            # 🖼 OCR text
                            # if r.get("ocr_text"):
                            #     st.markdown("**🖼 Slide Text:**")
                            #     st.write(r["ocr_text"][:300])

                            # 📊 Score
                            # st.markdown(
                            #     f"📊 **Score:** `{r.get('score', 0):.4f}`"
                            # )

                            st.divider()

            except Exception as e:
                st.error(f"❌ Error: {str(e)}")