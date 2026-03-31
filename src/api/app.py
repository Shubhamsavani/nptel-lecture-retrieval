from flask import Flask, request, jsonify
import sys
import os

# Add project path
sys.path.append(os.path.abspath("D:/sk/Phase 2/project/nptel-lecture-retrieval/src/retrieval"))

from retriever import api_search

app = Flask(__name__)


@app.route("/")
def home():
    return {"message": "Lecture Retrieval API running 🚀"}


@app.route("/search", methods=["POST"])
def search():
    try:
        data = request.json

        query = data.get("query", "")
        strategy = data.get("strategy", "c3")
        use_llm = data.get("use_llm", False)

        if not query:
            return jsonify({"error": "Query is required"}), 400

        results = api_search(
            query=query,
            strategy=strategy,
            use_llm=use_llm
        )

        return jsonify({
            "query": query,
            "results": results
        })

    except Exception as e:
        print("🔥 BACKEND ERROR:", str(e))   # IMPORTANT
        return jsonify({
            "error": str(e)
        }), 500

if __name__ == "__main__":
    app.run(debug=True, port=5000)