"""
Retrieval Service — Updated with Real Embedding
=====================================================================
IR Concepts Applied:

    VSM / TF-IDF with Cosine Similarity (Lecture 2, 3):
        Sparse vector model. Similarity = cosine angle between vectors.
        Formula: cos(q, d) = (q · d) / (|q| × |d|)

    BM25 - Best Matching 25 (Lecture 3):
        Probabilistic model with TF saturation (k1) and length norm (b).
        Formula: Σ IDF(t) × TF_norm(t,d)

    Embedding / Semantic Retrieval (Project Spec + Lecture 3):
        Dense SBERT or Word2Vec vectors → FAISS nearest-neighbor search.
        "Neural models capture semantic meaning and context."
        Captures synonymy: "car" ≈ "automobile" in vector space.

    Hybrid Serial — Cascade (Lecture 3):
        Stage 1: BM25 → top-1000 candidates (fast keyword filter)
        Stage 2: Embedding re-ranks the 1000 candidates (semantic precision)
        "Use a lightweight model to filter, then a complex model to re-rank."

    Hybrid Parallel (Lecture 3):
        BM25 list + Embedding list run simultaneously → Fusion.
        Fusion Methods:
          - RRF:      score(d) = Σ 1/(k + rank(d))
          - Weighted: score(d) = α × BM25_norm + (1-α) × Emb_norm

SOA Role (Project Spec):
    خدمة البحث والاسترجاع — تستدعي Embedding Service و Indexing Service.
"""

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Dict, Optional, Tuple
from collections import defaultdict
import math
import httpx

app = FastAPI(
    title="Retrieval Service",
    description="VSM/TF-IDF, BM25, Embedding (SBERT/W2V), Hybrid Serial & Parallel",
    version="2.0.0",
)

PREPROCESSING_URL  = "http://localhost:8001"
INDEXING_URL       = "http://localhost:8002"
EMBEDDING_URL      = "http://localhost:8006"
DOCUMENT_STORE_URL = "http://localhost:8009"


# ──────────────────────────────────────────────
# Schemas
# ──────────────────────────────────────────────
class RetrievalRequest(BaseModel):
    query: str
    dataset_id: str
    model: str = "bm25"
    # "tfidf" | "bm25" | "sbert" | "word2vec" |
    # "hybrid_serial" | "hybrid_parallel"
    top_k: int = 10
    bm25_k1: float = 1.5
    bm25_b:  float = 0.75
    fusion_method: str = "rrf"          # "rrf" | "weighted"
    fusion_weights: Dict[str, float] = {"bm25": 0.5, "embedding": 0.5}
    embedding_model: str = "sbert"      # "sbert" | "word2vec"
    similarity_threshold: float = 0.0
    hybrid_candidates: int = 1000       # BM25 candidates for serial stage-1


class RetrievalResult(BaseModel):
    doc_id: str
    score: float
    title: Optional[str] = ""
    snippet: Optional[str] = ""
    rank: int


class RetrievalResponse(BaseModel):
    query: str
    model: str
    results: List[RetrievalResult]
    total_candidates: int
    retrieved: int


# ──────────────────────────────────────────────
# Helper: Preprocess Query
# ──────────────────────────────────────────────
async def preprocess_query(query: str) -> Tuple[List[str], str]:
    """
    Query must be preprocessed identically to documents (Project Spec §4).
    Calls Preprocessing Service.
    """
    async with httpx.AsyncClient(timeout=30.0) as client:
        r = await client.post(
            f"{PREPROCESSING_URL}/preprocess",
            json={"text": query, "use_lemmatization": True, "remove_stopwords": True},
        )
        data = r.json()
        return data["processed_tokens"], data["processed_text"]


# ──────────────────────────────────────────────
# BM25 Scoring (Lecture 3)
# ──────────────────────────────────────────────
def bm25_score(
    query_tokens: List[str],
    doc_id: str,
    inverted_index: Dict,
    doc_lengths: Dict,
    avg_dl: float,
    N: int,
    k1: float = 1.5,
    b: float = 0.75,
) -> float:
    """
    BM25 Formula (Lecture 3):
    Σ IDF(t) × [TF(t,d)×(k1+1)] / [TF(t,d) + k1×(1 - b + b×|d|/avgdl)]
    k1: TF saturation  |  b: length normalization
    """
    score = 0.0
    dl = doc_lengths.get(doc_id, 1)
    for term in query_tokens:
        postings = inverted_index.get(term, {})
        df = len(postings)
        tf = postings.get(doc_id, 0)
        if df == 0 or tf == 0:
            continue
        idf = math.log((N - df + 0.5) / (df + 0.5) + 1)
        tf_norm = (tf * (k1 + 1)) / (tf + k1 * (1 - b + b * (dl / avg_dl)))
        score += idf * tf_norm
    return score


async def _bm25_retrieve(
    query_tokens: List[str],
    req: RetrievalRequest,
    top_k_override: int = None,
) -> List[Tuple[str, float]]:
    """Run BM25 retrieval via Indexing Service."""
    top_k = top_k_override or req.top_k
    async with httpx.AsyncClient(timeout=60.0) as client:
        search_r = await client.post(
            f"{INDEXING_URL}/search/inverted",
            json={"terms": query_tokens, "top_k": top_k * 5},
        )
        stats_r = await client.get(f"{INDEXING_URL}/stats")

    candidates_data = search_r.json()
    stats = stats_r.json()
    N = stats.get("total_documents", 1)
    avg_dl = stats.get("avg_doc_length", 1)

    # Reconstruct inverted index from candidate results
    inverted_index: Dict = defaultdict(dict)
    doc_lengths: Dict = {}
    for cand in candidates_data.get("candidates", []):
        doc_id = cand["doc_id"]
        for term, tf in cand["term_freqs"].items():
            inverted_index[term][doc_id] = tf
        doc_lengths[doc_id] = 1

    scores = []
    for cand in candidates_data.get("candidates", []):
        doc_id = cand["doc_id"]
        score = bm25_score(
            query_tokens, doc_id, inverted_index, doc_lengths,
            avg_dl, N, req.bm25_k1, req.bm25_b,
        )
        if score >= req.similarity_threshold:
            scores.append((doc_id, score))

    scores.sort(key=lambda x: x[1], reverse=True)
    return scores[:top_k]


# ──────────────────────────────────────────────
# TF-IDF Retrieval (Lecture 3)
# ──────────────────────────────────────────────
async def _tfidf_retrieve(
    query_tokens: List[str],
    req: RetrievalRequest,
) -> List[Tuple[str, float]]:
    """TF-IDF cosine similarity retrieval."""
    async with httpx.AsyncClient(timeout=60.0) as client:
        search_r = await client.post(
            f"{INDEXING_URL}/search/inverted",
            json={"terms": query_tokens, "top_k": req.top_k * 10},
        )
        stats_r = await client.get(f"{INDEXING_URL}/stats")

    candidates_data = search_r.json()
    N = stats_r.json().get("total_documents", 1)

    inverted_index: Dict = defaultdict(dict)
    for cand in candidates_data.get("candidates", []):
        for term, tf in cand["term_freqs"].items():
            inverted_index[term][cand["doc_id"]] = tf

    # Build query vector
    query_vec = {}
    for term in query_tokens:
        df = len(inverted_index.get(term, {}))
        if df > 0:
            idf = math.log((N + 1) / (df + 1)) + 1
            query_vec[term] = idf

    q_norm = math.sqrt(sum(v**2 for v in query_vec.values()))
    if q_norm == 0:
        return []
    query_vec = {t: v / q_norm for t, v in query_vec.items()}

    scores = []
    for cand in candidates_data.get("candidates", []):
        doc_id = cand["doc_id"]
        tf_map = cand["term_freqs"]
        doc_vec = {}
        for term, tf_raw in tf_map.items():
            df = len(inverted_index.get(term, {}))
            if df > 0:
                tf = 1 + math.log(tf_raw) if tf_raw > 0 else 0
                idf = math.log((N + 1) / (df + 1)) + 1
                doc_vec[term] = tf * idf
        d_norm = math.sqrt(sum(v**2 for v in doc_vec.values()))
        if d_norm > 0:
            doc_vec = {t: v / d_norm for t, v in doc_vec.items()}
        score = sum(query_vec.get(t, 0) * doc_vec.get(t, 0) for t in query_vec)
        if score >= req.similarity_threshold:
            scores.append((doc_id, score))

    scores.sort(key=lambda x: x[1], reverse=True)
    return scores[:req.top_k]


# ──────────────────────────────────────────────
# Embedding Retrieval — REAL (Project Spec)
# ──────────────────────────────────────────────
async def _embedding_retrieve(
    query: str,
    req: RetrievalRequest,
    candidate_ids: Optional[List[str]] = None,
) -> List[Tuple[str, float]]:
    """
    Semantic retrieval via Embedding Service + FAISS (Lecture 2 & 3).
    Calls /search on the Embedding Service which:
      1. Encodes query with SBERT or Word2Vec
      2. Searches FAISS for nearest document vectors
      3. Returns (doc_id, cosine_score) pairs

    If candidate_ids is given (serial hybrid), only those docs are re-ranked.
    """
    payload = {
        "query": query,
        "dataset_id": req.dataset_id,
        "model": req.embedding_model,
        "top_k": req.top_k,
    }

    endpoint = "/search"
    if candidate_ids:
        # Serial hybrid: re-rank only BM25 candidates
        endpoint = "/search/rerank"
        payload = {
            "query": query,
            "candidate_ids": candidate_ids,
            "dataset_id": req.dataset_id,
            "model": req.embedding_model,
            "top_k": req.top_k,
        }

    async with httpx.AsyncClient(timeout=60.0) as client:
        if candidate_ids:
            r = await client.post(f"{EMBEDDING_URL}{endpoint}", json=payload)
        else:
            r = await client.post(f"{EMBEDDING_URL}{endpoint}", json=payload)

    if r.status_code != 200:
        raise HTTPException(502, f"Embedding service error: {r.text}")

    data = r.json()
    if isinstance(data, list):
        results = data
    else:
        results = data.get("results", [])

    return [(item["doc_id"], item["score"]) for item in results]


# ──────────────────────────────────────────────
# Fusion Methods (Lecture 3)
# ──────────────────────────────────────────────
def rrf_fusion(
    ranked_lists: List[List[Tuple[str, float]]],
    k: int = 60,
) -> List[Tuple[str, float]]:
    """
    RRF — Reciprocal Rank Fusion (Lecture 3):
    score(d) = Σ 1 / (k + rank(d))
    Fairly merges multiple ranked lists without score normalization.
    """
    doc_scores: Dict[str, float] = defaultdict(float)
    for ranked_list in ranked_lists:
        for rank, (doc_id, _) in enumerate(ranked_list, start=1):
            doc_scores[doc_id] += 1.0 / (k + rank)
    return sorted(doc_scores.items(), key=lambda x: x[1], reverse=True)


def weighted_fusion(
    ranked_lists: List[List[Tuple[str, float]]],
    weights: List[float],
) -> List[Tuple[str, float]]:
    """
    Weighted Score Fusion (Lecture 3):
    final_score = α × norm_score_BM25 + (1-α) × norm_score_Embedding
    """
    doc_scores: Dict[str, float] = defaultdict(float)
    for ranked_list, w in zip(ranked_lists, weights):
        if not ranked_list:
            continue
        max_score = ranked_list[0][1] + 1e-9
        for doc_id, score in ranked_list:
            doc_scores[doc_id] += w * (score / max_score)
    return sorted(doc_scores.items(), key=lambda x: x[1], reverse=True)


# ──────────────────────────────────────────────
# Hybrid — Serial / Cascade (Lecture 3)
# ──────────────────────────────────────────────
async def _hybrid_serial(
    query: str,
    query_tokens: List[str],
    req: RetrievalRequest,
) -> List[Tuple[str, float]]:
    """
    Serial Hybrid (Lecture 3 — Cascade):
    Stage 1: BM25 retrieves top-N candidates (fast, keyword)
    Stage 2: Embedding re-ranks the candidates (slow, semantic)

    Why serial? Embedding inference is expensive — run only on a
    small candidate set, not the full corpus.
    Lecture: "Use a lightweight model to filter candidates,
              then apply a complex model to re-rank."
    """
    # Stage 1 — BM25 wide retrieval
    bm25_results = await _bm25_retrieve(
        query_tokens, req,
        top_k_override=req.hybrid_candidates,
    )
    if not bm25_results:
        return []

    candidate_ids = [doc_id for doc_id, _ in bm25_results]

    # Stage 2 — Embedding re-ranks candidates
    try:
        reranked = await _embedding_retrieve(query, req, candidate_ids=candidate_ids)
        return reranked[:req.top_k]
    except Exception:
        # Fallback: return BM25 results if embedding unavailable
        return bm25_results[:req.top_k]


# ──────────────────────────────────────────────
# Hybrid — Parallel (Lecture 3)
# ──────────────────────────────────────────────
async def _hybrid_parallel(
    query: str,
    query_tokens: List[str],
    req: RetrievalRequest,
) -> List[Tuple[str, float]]:
    """
    Parallel Hybrid (Lecture 3):
    BM25 and Embedding run simultaneously → Fusion merges results.

    Why parallel? Both models contribute independently;
    fusion combines their strengths.
    "Run multiple models in parallel (e.g., BM25, BERT), then fuse results."
    """
    import asyncio

    # Run both retrievals concurrently
    bm25_task = _bm25_retrieve(query_tokens, req)
    emb_task  = _embedding_retrieve(query, req)

    try:
        bm25_results, emb_results = await asyncio.gather(bm25_task, emb_task)
    except Exception:
        # If embedding fails, fallback to BM25 only
        bm25_results = await _bm25_retrieve(query_tokens, req)
        return bm25_results

    # Fuse results
    if req.fusion_method == "weighted":
        w_bm25 = req.fusion_weights.get("bm25", 0.5)
        w_emb  = req.fusion_weights.get("embedding", 0.5)
        fused  = weighted_fusion([bm25_results, emb_results], [w_bm25, w_emb])
    else:
        # Default: RRF
        fused = rrf_fusion([bm25_results, emb_results])

    return fused[:req.top_k]


# ──────────────────────────────────────────────
# Main Endpoint
# ──────────────────────────────────────────────
@app.get("/health")
def health():
    return {"status": "ok", "service": "retrieval", "version": "2.0"}


@app.post("/retrieve", response_model=RetrievalResponse)
async def retrieve(req: RetrievalRequest):
    """
    Unified retrieval endpoint.
    Supported models:
      tfidf           → VSM with cosine similarity
      bm25            → Probabilistic BM25
      sbert           → Sentence-BERT semantic search (FAISS)
      word2vec        → Word2Vec mean-pooling search (FAISS)
      hybrid_serial   → BM25 → Embedding re-rank  (Cascade)
      hybrid_parallel → BM25 + Embedding → RRF/Weighted fusion
    """
    query_tokens, _ = await preprocess_query(req.query)

    if not query_tokens and req.model not in ("sbert", "word2vec",
                                               "hybrid_serial", "hybrid_parallel"):
        return RetrievalResponse(
            query=req.query, model=req.model,
            results=[], total_candidates=0, retrieved=0,
        )

    # ── Route to correct model ──
    if req.model == "tfidf":
        raw = await _tfidf_retrieve(query_tokens, req)

    elif req.model == "bm25":
        raw = await _bm25_retrieve(query_tokens, req)

    elif req.model in ("sbert", "word2vec"):
        # Pure semantic retrieval via Embedding Service
        r = req.copy()
        r.embedding_model = req.model
        raw = await _embedding_retrieve(req.query, r)

    elif req.model == "hybrid_serial":
        raw = await _hybrid_serial(req.query, query_tokens, req)

    elif req.model == "hybrid_parallel":
        raw = await _hybrid_parallel(req.query, query_tokens, req)

    else:
        raise HTTPException(
            400,
            f"Unknown model '{req.model}'. "
            "Use: tfidf, bm25, sbert, word2vec, hybrid_serial, hybrid_parallel"
        )

    # ══════════════════════════════════════════════════════════
    # ★ آخر خطوة فقط: استرجاع النصوص الأصلية (raw) من Document Store
    # ══════════════════════════════════════════════════════════
    # حتى هذه النقطة، raw كان عبارة عن (doc_id, score) فقط — جاءت
    # من inverted index / FAISS vectors (بنى بيانات خفيفة بدون نص).
    #
    # الآن، ولـ top_k النتائج فقط، نستعلم دفعة واحدة (batch query)
    # عن النص الأصلي الكامل من Document Store (SQLite، doc_id كـ
    # Primary Key → استرجاع سريع O(log n)).
    #
    # هذا يطابق ملاحظة الأستاذ: "الـ raw text بدو يكون بداتا بيز
    # بلحظة الكويري" — استعلام واحد فقط، على أعلى top_k نتيجة،
    # في آخر خطوة، بدل تحميل النصوص الكاملة أثناء البحث نفسه.
    top_results = raw[:req.top_k]
    doc_ids_to_fetch = [doc_id for doc_id, _ in top_results]

    raw_docs_map: Dict[str, Dict] = {}
    if doc_ids_to_fetch:
        try:
            async with httpx.AsyncClient(timeout=30.0) as client:
                doc_r = await client.post(
                    f"{DOCUMENT_STORE_URL}/get/batch",
                    json={"dataset_id": req.dataset_id, "doc_ids": doc_ids_to_fetch},
                )
                if doc_r.status_code == 200:
                    for item in doc_r.json():
                        raw_docs_map[item["doc_id"]] = item
        except Exception:
            pass  # Document Store غير متاح — نُكمل بدون نصوص (degraded mode)

    results = []
    for rank, (doc_id, score) in enumerate(top_results, start=1):
        doc_data = raw_docs_map.get(doc_id, {})
        text = doc_data.get("raw_text", "")
        results.append(RetrievalResult(
            doc_id=doc_id,
            score=round(score, 6),
            title=doc_data.get("title", ""),
            snippet=text[:200] + "..." if len(text) > 200 else text,
            rank=rank,
        ))

    return RetrievalResponse(
        query=req.query,
        model=req.model,
        results=results,
        total_candidates=len(raw),
        retrieved=len(results),
    )


@app.get("/models")
def list_models():
    """List all available retrieval models."""
    return {
        "models": [
            {"id": "tfidf",            "name": "TF-IDF + Cosine Similarity",        "type": "sparse"},
            {"id": "bm25",             "name": "BM25 (Best Matching 25)",            "type": "sparse"},
            {"id": "sbert",            "name": "Sentence-BERT (FAISS)",              "type": "dense"},
            {"id": "word2vec",         "name": "Word2Vec Mean Pooling (FAISS)",       "type": "dense"},
            {"id": "hybrid_serial",    "name": "Hybrid Serial: BM25 → Embedding",    "type": "hybrid"},
            {"id": "hybrid_parallel",  "name": "Hybrid Parallel: BM25 + Embedding → Fusion", "type": "hybrid"},
        ]
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8003)