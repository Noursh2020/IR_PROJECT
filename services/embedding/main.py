"""
Embedding Service
=====================================================================
IR Concepts Applied (Lecture 2 & 3 + Project Spec):

    Dense Embedding / Neural Representation (Lecture 3):
        Unlike TF-IDF (sparse, keyword-based), embeddings map text to a
        dense vector in a continuous semantic space.
        Lecture: "Neural models capture semantic meaning and context,
                  improving recall and precision."
        Example:  "car" and "automobile" → close vectors (high cosine sim)
                  even though they share no common tokens.

    Sentence Transformers (Project Spec - Embedding):
        Pre-trained BERT-based model that encodes full sentences/passages.
        Model used: all-MiniLM-L6-v2
          - 384-dimensional vectors
          - Fast inference, good quality for retrieval
          - Trained with contrastive learning on (query, passage) pairs

    Word2Vec (Project Spec - Word2Vec):
        Older word-level embedding.
        We represent a document as the mean of its word vectors.
        Trained on the indexed corpus using gensim.
        Captures word-level semantics but misses sentence structure.

    Vector Index - FAISS (Lecture 2 - Index Types):
        Lecture: "Vector Index stores embedding vectors and supports
                  semantic similarity search."
        FAISS (Facebook AI Similarity Search):
          - Stores all document vectors as a matrix
          - Efficient Approximate Nearest Neighbor (ANN) search
          - Much faster than brute-force cosine over all documents
        Index type used: IndexFlatIP (exact inner product / cosine on normalized vecs)

    Cosine Similarity for Embeddings (Lecture 3):
        After L2-normalizing vectors:
          cosine(q, d) = q · d  (dot product = cosine on unit vectors)
        Lecture: "Cosine similarity disregards magnitude; considers only angle."

    Hybrid Role (Lecture 3 - Serial & Parallel):
        Serial:   BM25 top-1000 → Embedding re-ranks (slow model on small set)
        Parallel: BM25 list + Embedding list → RRF / Weighted fusion

SOA Role (Project Spec):
    خدمة مستقلة للـ Embedding — تُبنى وتُخزّن الفهارس المتجهية،
    وتُجيب على استعلامات البحث الدلالي.
"""

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Dict, Optional, Tuple
from pathlib import Path
import numpy as np
import pickle
import httpx
import faiss

app = FastAPI(
    title="Embedding Service",
    description="Sentence-Transformers + Word2Vec + FAISS Vector Index for Semantic Retrieval",
    version="1.0.0",
)

PREPROCESSING_URL = "http://localhost:8001"
INDEX_DIR = Path("indexes/embeddings")
INDEX_DIR.mkdir(parents=True, exist_ok=True)

# ──────────────────────────────────────────────
# Lazy-loaded models (loaded once on first use)
# ──────────────────────────────────────────────
_sbert_model = None       # sentence-transformers
_w2v_model = None         # gensim Word2Vec
_faiss_sbert: Dict = {}   # dataset_id → {index, doc_ids}
_faiss_w2v: Dict = {}     # dataset_id → {index, doc_ids}
_doc_store: Dict = {}     # dataset_id → {doc_id: text}

SBERT_MODEL_NAME = "all-MiniLM-L6-v2"
EMBEDDING_DIM_SBERT = 384


def get_sbert():
    """Load sentence-transformers model (once)."""
    global _sbert_model
    if _sbert_model is None:
        from sentence_transformers import SentenceTransformer
        print(f"[Embedding] Loading SBERT model: {SBERT_MODEL_NAME}")
        _sbert_model = SentenceTransformer(SBERT_MODEL_NAME)
        print("[Embedding] SBERT model loaded.")
    return _sbert_model


def get_faiss():
    """Import FAISS (lazy)."""
    import faiss
    return faiss


# ──────────────────────────────────────────────
# Schemas
# ──────────────────────────────────────────────

class EmbedRequest(BaseModel):
    texts: List[str]
    model: str = "sbert"     # "sbert" | "word2vec"
    batch_size: int = 64


class EmbedResponse(BaseModel):
    vectors: List[List[float]]
    model: str
    dim: int


class IndexEmbeddingRequest(BaseModel):
    dataset_id: str
    doc_ids: List[str]
    texts: List[str]
    model: str = "sbert"     # "sbert" | "word2vec" | "both"


class VectorSearchRequest(BaseModel):
    query: str
    dataset_id: str
    model: str = "sbert"
    top_k: int = 100


class VectorSearchResult(BaseModel):
    doc_id: str
    score: float
    rank: int


class VectorSearchResponse(BaseModel):
    query: str
    model: str
    results: List[VectorSearchResult]


# ──────────────────────────────────────────────
# Encoding Functions
# ──────────────────────────────────────────────

def encode_sbert(texts: List[str], batch_size: int = 64) -> np.ndarray:
    """
    Sentence-BERT Encoding (Project Spec - Embedding):
    Encodes a list of texts into dense 384-dim vectors.
    Uses mean pooling over BERT token embeddings.

    L2-normalize so that dot product = cosine similarity.
    Lecture: "Cosine similarity = dot product on unit vectors."
    """
    model = get_sbert()
    # encode() returns numpy array shape (N, 384)
    vectors = model.encode(
        texts,
        batch_size=batch_size,
        show_progress_bar=len(texts) > 100,
        convert_to_numpy=True,
        normalize_embeddings=True,   # L2-normalize → cosine via dot product
    )
    return vectors.astype(np.float32)


def encode_word2vec(texts: List[str], dataset_id: str = "") -> np.ndarray:
    """
    Word2Vec Encoding (Project Spec - Word2Vec):
    Represents each document as the mean of its word vectors.

    Doc vector = (1/N) × Σ word_vector(w)  for w in doc tokens

    Words not in vocabulary are skipped.
    L2-normalized at the end for cosine similarity.
    """
    global _w2v_model
    if _w2v_model is None:
        raise HTTPException(503, "Word2Vec model not trained yet. Call /train/word2vec first.")

    vectors = []
    for text in texts:
        tokens = text.lower().split()
        word_vecs = [
            _w2v_model.wv[w]
            for w in tokens
            if w in _w2v_model.wv
        ]
        if word_vecs:
            doc_vec = np.mean(word_vecs, axis=0).astype(np.float32)
        else:
            # Zero vector for empty/OOV documents
            doc_vec = np.zeros(_w2v_model.vector_size, dtype=np.float32)

        # L2-normalize
        norm = np.linalg.norm(doc_vec)
        if norm > 0:
            doc_vec = doc_vec / norm
        vectors.append(doc_vec)

    return np.array(vectors, dtype=np.float32)


# ──────────────────────────────────────────────
# FAISS Index Management (Lecture 2 - Vector Index)
# ──────────────────────────────────────────────

def build_faiss_index(vectors: np.ndarray) -> "faiss.Index":
    """
    Build a FAISS flat inner-product index.
    IndexFlatIP = exact search using dot product (= cosine on L2-normalized vectors).
    Lecture: "Vector Index stores embedding vectors and supports semantic similarity."
    """
    faiss = get_faiss()
    dim = vectors.shape[1]
    # IndexFlatIP: exact nearest-neighbor by inner product
    index = faiss.IndexFlatIP(dim)
    index.add(vectors)
    print(f"[Embedding] FAISS index built: {index.ntotal} vectors, dim={dim}")
    return index


def save_faiss_index(dataset_id: str, model_name: str, index, doc_ids: List[str]):
    """Persist FAISS index and doc_ids mapping to disk."""
    faiss = get_faiss()
    path = INDEX_DIR / dataset_id / model_name
    path.mkdir(parents=True, exist_ok=True)

    faiss.write_index(index, str(path / "faiss.index"))
    with open(path / "doc_ids.pkl", "wb") as f:
        pickle.dump(doc_ids, f)
    print(f"[Embedding] Saved FAISS index for dataset='{dataset_id}' model='{model_name}'")


def load_faiss_index(dataset_id: str, model_name: str) -> Tuple[Optional[object], List[str]]:
    """Load FAISS index from disk."""
    faiss = get_faiss()
    path = INDEX_DIR / dataset_id / model_name

    if not (path / "faiss.index").exists():
        return None, []

    index = faiss.read_index(str(path / "faiss.index"))
    with open(path / "doc_ids.pkl", "rb") as f:
        doc_ids = pickle.load(f)

    print(f"[Embedding] Loaded FAISS index: {index.ntotal} vectors for '{dataset_id}/{model_name}'")
    return index, doc_ids


def search_faiss(
    query_vec: np.ndarray,
    dataset_id: str,
    model_name: str,
    top_k: int,
    candidate_ids: Optional[List[str]] = None,
) -> List[Tuple[str, float]]:
    """
    FAISS nearest-neighbor search.
    Returns top_k (doc_id, score) pairs sorted by cosine similarity.

    If candidate_ids is given (serial hybrid), search only among those docs.
    """
    store_key = f"{dataset_id}_{model_name}"

    # Load from memory or disk
    if store_key not in _faiss_sbert:
        index, doc_ids = load_faiss_index(dataset_id, model_name)
        if index is None:
            return []
        _faiss_sbert[store_key] = {"index": index, "doc_ids": doc_ids}

    index = _faiss_sbert[store_key]["index"]
    doc_ids = _faiss_sbert[store_key]["doc_ids"]

    if index.ntotal == 0:
        return []

    query_vec = query_vec.reshape(1, -1).astype(np.float32)
    k = min(top_k, index.ntotal)

    # FAISS search: returns (scores, indices)
    scores, indices = index.search(query_vec, k)

    results = []
    doc_id_set = set(candidate_ids) if candidate_ids else None

    for score, idx in zip(scores[0], indices[0]):
        if idx < 0 or idx >= len(doc_ids):
            continue
        doc_id = doc_ids[idx]
        if doc_id_set and doc_id not in doc_id_set:
            continue
        results.append((doc_id, float(score)))

    return results


# ──────────────────────────────────────────────
# API Endpoints
# ──────────────────────────────────────────────

@app.get("/health")
def health():
    sbert_loaded = _sbert_model is not None
    w2v_loaded = _w2v_model is not None
    return {
        "status": "ok",
        "service": "embedding",
        "sbert_loaded": sbert_loaded,
        "word2vec_loaded": w2v_loaded,
        "indexed_datasets": list(set(k.split("_")[0] for k in _faiss_sbert.keys())),
    }


@app.post("/embed", response_model=EmbedResponse)
def embed_texts(req: EmbedRequest):
    """
    Encode a list of texts into dense vectors.
    Used by other services to get query/document embeddings.
    """
    if req.model == "sbert":
        vectors = encode_sbert(req.texts, req.batch_size)
        dim = vectors.shape[1]
    elif req.model == "word2vec":
        vectors = encode_word2vec(req.texts)
        dim = vectors.shape[1]
    else:
        raise HTTPException(400, f"Unknown model: {req.model}. Use 'sbert' or 'word2vec'.")

    return EmbedResponse(
        vectors=vectors.tolist(),
        model=req.model,
        dim=dim,
    )


@app.post("/index")
async def index_documents(req: IndexEmbeddingRequest):
    """
    Build FAISS vector index for a dataset.
    Pipeline:
      1. Encode all documents with SBERT / Word2Vec
      2. Build FAISS IndexFlatIP
      3. Save to disk

    Called by the Indexing Service after building the Inverted Index.
    """
    if not req.doc_ids or not req.texts:
        raise HTTPException(400, "doc_ids and texts must not be empty")

    _doc_store[req.dataset_id] = dict(zip(req.doc_ids, req.texts))
    models_to_build = []

    if req.model in ("sbert", "both"):
        models_to_build.append("sbert")
    if req.model in ("word2vec", "both"):
        models_to_build.append("word2vec")

    results = {}

    for model_name in models_to_build:
        print(f"[Embedding] Encoding {len(req.texts)} docs with {model_name}...")

        if model_name == "sbert":
            # Encode in batches
            all_vectors = []
            batch_size = 64
            for i in range(0, len(req.texts), batch_size):
                batch = req.texts[i:i+batch_size]
                vecs = encode_sbert(batch, batch_size)
                all_vectors.append(vecs)
            vectors = np.vstack(all_vectors)
        else:
            # Train Word2Vec on this corpus first
            await train_word2vec_internal(req.texts)
            vectors = encode_word2vec(req.texts)

        # Build and save FAISS index
        index = build_faiss_index(vectors)
        save_faiss_index(req.dataset_id, model_name, index, req.doc_ids)

        # Cache in memory
        store_key = f"{req.dataset_id}_{model_name}"
        _faiss_sbert[store_key] = {"index": index, "doc_ids": req.doc_ids}

        results[model_name] = {
            "indexed": len(req.doc_ids),
            "dim": vectors.shape[1],
        }

    return {"dataset_id": req.dataset_id, "results": results}


@app.post("/search", response_model=VectorSearchResponse)
async def vector_search(req: VectorSearchRequest):
    """
    Semantic search using FAISS.
    Encodes the query with SBERT/Word2Vec, then finds nearest document vectors.

    Lecture: "Queries must be represented the same way as documents."
    """
    # Encode query
    if req.model == "sbert":
        query_vec = encode_sbert([req.query])[0]
    elif req.model == "word2vec":
        query_vec = encode_word2vec([req.query])[0]
    else:
        raise HTTPException(400, f"Unknown model: {req.model}")

    # Search FAISS
    raw_results = search_faiss(query_vec, req.dataset_id, req.model, req.top_k)

    results = [
        VectorSearchResult(doc_id=doc_id, score=round(score, 6), rank=rank)
        for rank, (doc_id, score) in enumerate(raw_results, start=1)
    ]

    return VectorSearchResponse(query=req.query, model=req.model, results=results)


@app.post("/search/rerank")
async def rerank_with_embedding(
    query: str,
    candidate_ids: List[str],
    dataset_id: str,
    model: str = "sbert",
    top_k: int = 10,
):
    """
    Serial Hybrid — Stage 2 (Lecture 3):
    Given a list of candidate doc IDs (from BM25),
    re-rank them using embedding cosine similarity.

    Lecture: "Use a lightweight model to filter candidates,
              then apply a complex model to re-rank."
    """
    if model == "sbert":
        query_vec = encode_sbert([query])[0]
    else:
        query_vec = encode_word2vec([query])[0]

    # Search within candidates only
    raw_results = search_faiss(
        query_vec, dataset_id, model, top_k,
        candidate_ids=candidate_ids,
    )

    return [
        {"doc_id": doc_id, "score": round(score, 6), "rank": rank}
        for rank, (doc_id, score) in enumerate(raw_results, start=1)
    ]


@app.post("/train/word2vec")
async def train_word2vec(dataset_id: str, texts: List[str]):
    """
    Train Word2Vec on the current corpus (Project Spec).
    Word2Vec learns word vectors from co-occurrence patterns in the corpus.
    Lecture: "Word2Vec: word-level embeddings trained on the corpus."
    """
    await train_word2vec_internal(texts)
    return {"trained": True, "vocab_size": len(_w2v_model.wv)}


async def train_word2vec_internal(texts: List[str]):
    """Train Word2Vec model on provided texts."""
    global _w2v_model
    from gensim.models import Word2Vec

    tokenized = [text.lower().split() for text in texts]
    print(f"[Embedding] Training Word2Vec on {len(tokenized)} documents...")
    _w2v_model = Word2Vec(
        sentences=tokenized,
        vector_size=300,     # 300-dim word vectors
        window=5,            # context window
        min_count=2,         # ignore rare words
        workers=4,
        epochs=10,
    )
    print(f"[Embedding] Word2Vec trained. Vocab: {len(_w2v_model.wv)} words.")


@app.post("/load/{dataset_id}")
def load_indexes(dataset_id: str, model: str = "sbert"):
    """Load saved FAISS index from disk into memory."""
    index, doc_ids = load_faiss_index(dataset_id, model)
    if index is None:
        raise HTTPException(404, f"No saved embedding index for dataset='{dataset_id}' model='{model}'")
    store_key = f"{dataset_id}_{model}"
    _faiss_sbert[store_key] = {"index": index, "doc_ids": doc_ids}
    return {"loaded": True, "dataset_id": dataset_id, "model": model, "vectors": index.ntotal}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8006)