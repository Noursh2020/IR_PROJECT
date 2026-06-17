"""
API Gateway Service
=====================================================================
IR Concept (Project Spec - SOA):

    "خدمة الواجهة الأمامية أو الـ API Gateway"

    The Gateway is the single entry point for all client requests.
    It routes requests to the appropriate internal service,
    orchestrates multi-step pipelines, and aggregates results.

    SOA Communication Pattern:
        Client ─► Gateway ─► [Preprocessing | Indexing | Retrieval | Evaluation | Refinement]

    Why a Gateway? (Project Spec - SOA Principles):
        - Loose Coupling: clients don't know internal service URLs
        - Single entry point: easier to add auth, rate limiting, logging
        - Orchestration: multi-step search pipeline in one API call
        - Load balancing and service discovery ready

    Full Search Pipeline (orchestrated by Gateway):
        1. Query → Query Refinement Service (spelling, synonyms, history)
        2. Refined Query → Retrieval Service (BM25 / TF-IDF / Hybrid)
        3. Results → Ranking & Evaluation Service (scoring + metrics)
        4. Final ranked results → Client
"""

from fastapi import FastAPI, HTTPException, Query as QParam
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Dict, Optional, Any
import httpx
import asyncio

app = FastAPI(
    title="IR System API Gateway",
    description="Single entry point — routes to Preprocessing, Indexing, Retrieval, Evaluation, Query Refinement",
    version="1.0.0",
)

# Allow requests from the frontend (UI)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ──────────────────────────────────────────────
# Internal Service URLs (SOA)
# ──────────────────────────────────────────────
SERVICES = {
    "preprocessing":      "http://localhost:8001",
    "indexing":           "http://localhost:8002",
    "retrieval":          "http://localhost:8003",
    "ranking_evaluation": "http://localhost:8004",
    "query_refinement":   "http://localhost:8005",
    "embedding":          "http://localhost:8006",
    "dataset_loader":     "http://localhost:8007",
    "rag":                 "http://localhost:8008",
    "document_store":      "http://localhost:8009",
}


# ──────────────────────────────────────────────
# Schemas
# ──────────────────────────────────────────────

class SearchRequest(BaseModel):
    query: str
    dataset_id: str
    user_id: Optional[str] = "anonymous"

    # Model selection (UI requirement from project spec)
    model: str = "bm25"              # "tfidf" | "bm25" | "hybrid_serial" | "hybrid_parallel"
    top_k: int = 10

    # BM25 params (project spec: UI should expose these)
    bm25_k1: float = 1.5
    bm25_b: float = 0.75

    # Hybrid settings
    fusion_method: str = "rrf"       # "rrf" | "weighted" | "score"
    fusion_weights: Dict[str, float] = {"bm25": 0.5, "embedding": 0.5}

    # Feature toggles (project spec: basic vs basic+extra)
    use_query_refinement: bool = True
    similarity_threshold: float = 0.0


class SearchResponse(BaseModel):
    query: str
    refined_query: Optional[str]
    model: str
    dataset_id: str
    results: List[Dict]
    total_retrieved: int
    refinement_info: Optional[Dict] = None


class IndexRequest(BaseModel):
    dataset_id: str
    documents: List[Dict]   # [{doc_id, text, title?, metadata?}]


class HealthStatus(BaseModel):
    gateway: str
    services: Dict[str, str]


# ──────────────────────────────────────────────
# Health Check (checks all services)
# ──────────────────────────────────────────────

@app.get("/health", response_model=HealthStatus)
async def health():
    """Check health of gateway and all downstream services."""
    service_status = {}

    async def check(name, url):
        try:
            async with httpx.AsyncClient(timeout=3.0) as client:
                r = await client.get(f"{url}/health")
                service_status[name] = "ok" if r.status_code == 200 else "error"
        except Exception:
            service_status[name] = "unreachable"

    await asyncio.gather(*[check(n, u) for n, u in SERVICES.items()])

    return HealthStatus(gateway="ok", services=service_status)


# ──────────────────────────────────────────────
# Main Search Pipeline (orchestrated)
# ──────────────────────────────────────────────

@app.post("/search", response_model=SearchResponse)
async def search(req: SearchRequest):
    """
    Full IR search pipeline:
      Step 1: Query Refinement (optional)
      Step 2: Retrieval (BM25 / TF-IDF / Hybrid)
      Step 3: Return ranked results
    """
    refined_query = req.query
    refinement_info = None

    # ── Step 1: Query Refinement ──
    if req.use_query_refinement:
        try:
            async with httpx.AsyncClient(timeout=15.0) as client:
                refine_r = await client.post(
                    f"{SERVICES['query_refinement']}/refine",
                    json={
                        "query": req.query,
                        "user_id": req.user_id,
                        "dataset_id": req.dataset_id,
                        "use_synonyms": True,
                        "use_spell_correction": True,
                        "use_history": True,
                    },
                )
                if refine_r.status_code == 200:
                    refine_data = refine_r.json()
                    refined_query = refine_data.get("refined_query", req.query)
                    refinement_info = {
                        "added_synonyms": refine_data.get("added_synonyms", []),
                        "spell_corrections": refine_data.get("spell_corrections", {}),
                        "history_boost_terms": refine_data.get("history_boost_terms", []),
                    }
        except Exception as e:
            # Refinement failure is non-fatal — continue with original query
            refinement_info = {"error": str(e)}

    # ── Step 2: Retrieval ──
    async with httpx.AsyncClient(timeout=60.0) as client:
        retrieval_r = await client.post(
            f"{SERVICES['retrieval']}/retrieve",
            json={
                "query": refined_query,
                "dataset_id": req.dataset_id,
                "model": req.model,
                "top_k": req.top_k,
                "bm25_k1": req.bm25_k1,
                "bm25_b": req.bm25_b,
                "fusion_method": req.fusion_method,
                "fusion_weights": req.fusion_weights,
                "similarity_threshold": req.similarity_threshold,
            },
        )

    if retrieval_r.status_code != 200:
        raise HTTPException(502, f"Retrieval service error: {retrieval_r.text}")

    retrieval_data = retrieval_r.json()

    return SearchResponse(
        query=req.query,
        refined_query=refined_query if refined_query != req.query else None,
        model=req.model,
        dataset_id=req.dataset_id,
        results=retrieval_data.get("results", []),
        total_retrieved=retrieval_data.get("retrieved", 0),
        refinement_info=refinement_info,
    )


# ──────────────────────────────────────────────
# Indexing Proxy
# ──────────────────────────────────────────────

@app.post("/index")
async def index_documents(req: IndexRequest):
    """Proxy indexing requests to the Indexing Service."""
    from pydantic import BaseModel as BM
    async with httpx.AsyncClient(timeout=300.0) as client:
        r = await client.post(
            f"{SERVICES['indexing']}/index",
            json=req.dict(),
        )
    return r.json()


@app.post("/index/load/{dataset_id}")
async def load_index(dataset_id: str):
    """Load a saved index for a dataset."""
    async with httpx.AsyncClient(timeout=30.0) as client:
        r = await client.post(f"{SERVICES['indexing']}/load/{dataset_id}")
    return r.json()


@app.get("/index/stats")
async def index_stats():
    """Get indexing statistics."""
    async with httpx.AsyncClient(timeout=10.0) as client:
        r = await client.get(f"{SERVICES['indexing']}/stats")
    return r.json()


@app.post("/index/embeddings")
async def index_embeddings(payload: Dict):
    """
    Build FAISS vector index for a dataset.
    Call AFTER /index to also build the dense vector index.
    Supports: sbert, word2vec, or both.
    """
    async with httpx.AsyncClient(timeout=600.0) as client:
        r = await client.post(f"{SERVICES['embedding']}/index", json=payload)
    return r.json()


# ──────────────────────────────────────────────
# Evaluation Proxy
# ──────────────────────────────────────────────

@app.post("/evaluate")
async def evaluate(payload: Dict):
    """Proxy evaluation requests to the Ranking & Evaluation Service."""
    async with httpx.AsyncClient(timeout=60.0) as client:
        r = await client.post(
            f"{SERVICES['ranking_evaluation']}/evaluate/batch",
            json=payload,
        )
    return r.json()


# ──────────────────────────────────────────────
# Query History & Suggestions
# ──────────────────────────────────────────────

@app.get("/suggest")
async def suggest(query: str = QParam(...), user_id: str = QParam("anonymous")):
    """Get query suggestions for autocomplete."""
    async with httpx.AsyncClient(timeout=10.0) as client:
        r = await client.post(
            f"{SERVICES['query_refinement']}/suggest",
            json={"partial_query": query, "user_id": user_id},
        )
    return r.json()


@app.get("/history/{user_id}")
async def get_user_history(user_id: str):
    """Get a user's search history."""
    async with httpx.AsyncClient(timeout=10.0) as client:
        r = await client.get(f"{SERVICES['query_refinement']}/history/{user_id}")
    return r.json()


# ──────────────────────────────────────────────
# Service Discovery
# ──────────────────────────────────────────────

# ─────────────────────────────────────────────
# RAG — Retrieval-Augmented Generation
# ─────────────────────────────────────────────

@app.post("/rag/ask")
async def rag_ask(payload: Dict):
    """
    RAG Pipeline الكامل:
    سؤال → استرجاع وثائق → Claude يُولّد إجابة مدعومة بالمصادر.

    الميزة الإضافية المطلوبة (5 أفراد).
    """
    async with httpx.AsyncClient(timeout=120.0) as client:
        r = await client.post(f"{SERVICES['rag']}/ask", json=payload)
    return r.json()


@app.get("/rag/explain")
async def rag_explain():
    """شرح RAG للتقرير والمقابلة."""
    async with httpx.AsyncClient(timeout=10.0) as client:
        r = await client.get(f"{SERVICES['rag']}/explain")
    return r.json()


# ─────────────────────────────────────────────
# Dataset Loader Proxy
# ─────────────────────────────────────────────

@app.get("/datasets")
async def list_datasets():
    """List available datasets and their loading status."""
    async with httpx.AsyncClient(timeout=10.0) as client:
        r = await client.get(f"{SERVICES['dataset_loader']}/datasets")
    return r.json()


@app.post("/datasets/load/{dataset_name}")
async def load_dataset(dataset_name: str, payload: Dict, background_tasks=None):
    """Start loading a dataset (msmarco or nq)."""
    async with httpx.AsyncClient(timeout=30.0) as client:
        r = await client.post(
            f"{SERVICES['dataset_loader']}/load/{dataset_name}",
            json=payload,
        )
    return r.json()


@app.get("/datasets/status/{dataset_name}")
async def dataset_status(dataset_name: str):
    """Track loading progress of a dataset."""
    async with httpx.AsyncClient(timeout=10.0) as client:
        r = await client.get(f"{SERVICES['dataset_loader']}/status/{dataset_name}")
    return r.json()


@app.get("/datasets/queries/{dataset_name}")
async def get_queries(dataset_name: str, limit: int = 100):
    """Get sample queries for a loaded dataset."""
    async with httpx.AsyncClient(timeout=10.0) as client:
        r = await client.get(f"{SERVICES['dataset_loader']}/queries/{dataset_name}?limit={limit}")
    return r.json()


@app.get("/services")
def list_services():
    """List all registered services (SOA service registry)."""
    return {
        "services": [
            {"name": name, "url": url, "port": url.split(":")[-1]}
            for name, url in SERVICES.items()
        ]
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)