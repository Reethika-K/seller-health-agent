"""
The knowledge base for the diagnostic agent (the "retrieval" part of RAG).

  python kb.py build                                  read knowledge_base/*.md, embed them, store in kb_index/
  python kb.py search "late shipments rising"         show the 3 closest documents
  python kb.py search "late shipments rising" --category Logistics

What happens:
  1. Each document is turned into a list of numbers (an embedding) that captures its meaning.
  2. The embeddings are stored in Chroma, a small local vector database (folder kb_index/).
  3. A search query is embedded the same way, and Chroma returns the documents whose
     embeddings are closest in meaning. The diagnostic agent will call search_knowledge().
"""
import os
import re
import sys
import time
from pathlib import Path

import chromadb
from dotenv import load_dotenv
from google import genai
from google.genai import types

load_dotenv()
KB_DIR = Path("knowledge_base")
INDEX_DIR = "kb_index"
COLLECTION = "seller_kb"
EMBED_MODEL = "gemini-embedding-001"
CATEGORIES = ["Marketing", "Logistics", "Customer Experience", "Seller support",
              "Finance", "Inventory", "General"]

client = genai.Client(api_key=os.environ.get("GEMINI_API_KEY", "missing"))


def embed(texts, kind):
    """Turn a list of texts into embeddings with Gemini. kind = 'doc' or 'query'."""
    task = "RETRIEVAL_DOCUMENT" if kind == "doc" else "RETRIEVAL_QUERY"
    for attempt in range(5):
        try:
            result = client.models.embed_content(
                model=EMBED_MODEL, contents=texts,
                config=types.EmbedContentConfig(task_type=task))
            return [e.values for e in result.embeddings]
        except Exception as e:
            temporary = any(c in str(e) for c in ("429", "RESOURCE_EXHAUSTED", "503", "UNAVAILABLE"))
            if temporary and attempt < 4:
                print(f"  Gemini busy, waiting 20s (attempt {attempt + 1})...")
                time.sleep(20)
            else:
                raise


def load_documents():
    """Read knowledge_base/*.md into a list of dicts (id, title, category, metrics, text)."""
    docs = []
    for path in sorted(KB_DIR.glob("*.md")):
        text = path.read_text(encoding="utf-8")
        title_line = text.splitlines()[0].lstrip("# ").strip()          # "LOG-01: Late shipment rate rising"
        doc_id, _, title = title_line.partition(":")
        category = re.search(r"^Category:\s*(.+)$", text, re.M).group(1).strip()
        metrics = re.search(r"^Metrics:\s*(.+)$", text, re.M).group(1).strip()
        docs.append({"id": doc_id.strip(), "title": title.strip(), "category": category,
                     "metrics": metrics, "text": text})
    return docs


def _collection():
    return chromadb.PersistentClient(path=INDEX_DIR).get_or_create_collection(
        COLLECTION, embedding_function=None, metadata={"hnsw:space": "cosine"})


def build_index():
    """Embed every document and store it. Safe to rerun: it replaces the old index."""
    docs = load_documents()
    if not docs:
        sys.exit(f"No .md files found in {KB_DIR}/")
    try:
        chromadb.PersistentClient(path=INDEX_DIR).delete_collection(COLLECTION)
    except Exception:
        pass                                  # no old index yet
    col = _collection()
    embeddings = embed([d["text"] for d in docs], "doc")
    col.add(ids=[d["id"] for d in docs], embeddings=embeddings,
            documents=[d["text"] for d in docs],
            metadatas=[{"title": d["title"], "category": d["category"], "metrics": d["metrics"]}
                       for d in docs])
    print(f"Indexed {len(docs)} documents into {INDEX_DIR}/")


def search_knowledge(query: str, category: str = "", top_k: int = 3) -> list:
    """Search the failure-pattern knowledge base for documents that explain a symptom.

    Args:
        query: Describe what you see in plain words, naming the metrics and their direction,
            for example "late_shipment_rate rising, stockout_rate flat, order_volume stable".
        category: Optional. Restrict the search to one category: Marketing, Logistics,
            Customer Experience, Seller support, Finance or Inventory. Leave empty to search all.
        top_k: How many documents to return (default 3).

    Returns:
        A list of documents, best match first, each with id, title, category, metrics, text and score.
    """
    col = _collection()
    where = None
    if category:
        # always allow the general documents (sudden vs gradual, dips, knock-on paths)
        where = {"category": {"$in": [category, "General"]}}
    res = col.query(query_embeddings=embed([query], "query"), n_results=top_k, where=where)
    out = []
    for i in range(len(res["ids"][0])):
        meta = res["metadatas"][0][i]
        out.append({"id": res["ids"][0][i], "title": meta["title"], "category": meta["category"],
                    "metrics": meta["metrics"], "text": res["documents"][0][i],
                    "score": round(1 - res["distances"][0][i], 3)})   # cosine similarity
    return out


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else ""
    if cmd == "build":
        build_index()
    elif cmd == "search" and len(sys.argv) > 2:
        cat = sys.argv[sys.argv.index("--category") + 1] if "--category" in sys.argv else ""
        for r in search_knowledge(sys.argv[2], category=cat):
            print(f"{r['score']:.3f}  {r['id']:7s} [{r['category']}]  {r['title']}")
    else:
        print('Usage:\n  python kb.py build\n  python kb.py search "your symptom" [--category Logistics]')