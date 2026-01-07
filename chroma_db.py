# ==========================================
# ChromaDB API Server for WhatsApp RAG Bot
# (Single collection, In-Memory, New API)
# ==========================================

import os
from flask import Flask, request, jsonify
import chromadb

# ---------------- APP ----------------
app = Flask(__name__)

# ---------------- CHROMA CLIENT ----------------
# New Chroma API (auto in-memory)
chroma_client = chromadb.Client()

# ---------------- COLLECTION ----------------
COLLECTION_NAME = "default"


def get_collection():
    try:
        return chroma_client.get_collection(name=COLLECTION_NAME)
    except Exception:
        return chroma_client.create_collection(name=COLLECTION_NAME)


# ---------------- ADD DOCUMENT ----------------
@app.route("/add_doc", methods=["POST"])
def add_doc():
    data = request.get_json(force=True)

    text = data.get("text")
    embedding = data.get("embedding")

    if text is None or embedding is None:
        return jsonify({"error": "text and embedding are required"}), 400

    try:
        collection = get_collection()

        collection.add(
            documents=[text],
            embeddings=[embedding],
            ids=[str(abs(hash(text)))]
        )

        return jsonify({"status": "success"}), 200

    except Exception as e:
        print("Add doc error:", e)
        return jsonify({"error": str(e)}), 500


# ---------------- QUERY DOCUMENTS ----------------
@app.route("/query_docs", methods=["POST"])
def query_docs():
    data = request.get_json(force=True)

    vector = data.get("vector")
    top_k = int(data.get("top_k", 3))

    if vector is None:
        return jsonify({"error": "vector is required"}), 400

    try:
        collection = get_collection()

        results = collection.query(
            query_embeddings=[vector],
            n_results=top_k
        )

        documents = results.get("documents", [[]])[0]

        return jsonify({"results": documents}), 200

    except Exception as e:
        print("Query error:", e)
        return jsonify({"error": str(e)}), 500


# ---------------- HEALTH ----------------
@app.route("/")
def health():
    return "ChromaDB API Server Running (In-Memory)", 200


# ---------------- RUN ----------------
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
