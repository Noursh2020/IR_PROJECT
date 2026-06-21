from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Dict, Optional, Tuple
import math
import httpx
from collections import defaultdict

app = FastAPI(
    title="Ranking & Evaluation Service",
    description="IR Evaluation: Precision, Recall, P@K, MAP, nDCG",
    version="1.0.0",
)
from fastapi.middleware.cors import CORSMiddleware
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

RETRIEVAL_URL = "http://localhost:8003"


# ──────────────────────────────────────────────
# Schemas
# ──────────────────────────────────────────────

class QrelEntry(BaseModel):
    """
    Qrel = Query Relevance Judgment.
    Standard format from ir-datasets: (query_id, doc_id, relevance_score)
    relevance_score: 0 = not relevant, 1 = relevant, 2+ = highly relevant
    """
    query_id: str
    doc_id: str
    relevance: int  # 0, 1, 2, ...


class RetrievedDoc(BaseModel):
    doc_id: str
    score: float
    rank: int


class EvaluationRequest(BaseModel):
    query_id: str
    retrieved_docs: List[RetrievedDoc]   # Ranked list from retrieval service
    qrels: List[QrelEntry]               # Ground truth relevance judgments
    k: int = 10                          # For P@K, AP@K, nDCG@K


class BatchEvaluationRequest(BaseModel):
    """Evaluate across multiple queries — used for MAP computation."""
    evaluations: List[EvaluationRequest]
    k: int = 10


class MetricsResult(BaseModel):
    query_id: str
    precision: float
    precision_at_k: float
    recall: float
    average_precision: float
    ndcg_at_k: float
    relevant_retrieved: int
    total_relevant: int
    total_retrieved: int


class BatchMetricsResult(BaseModel):
    per_query: List[MetricsResult]
    map_score: float           # Mean Average Precision
    mean_ndcg: float           # Mean nDCG@K
    mean_precision_at_k: float # Mean P@K
    mean_recall: float


# ──────────────────────────────────────────────
# Core Metric Functions (Lecture 4)
# ──────────────────────────────────────────────

def compute_precision(retrieved: List[str], relevant: set) -> float:
    """
    Precision (Lecture 4):
    Proportion of retrieved documents that are relevant.
    Formula: |Relevant ∩ Retrieved| / |Retrieved|
    """
    if not retrieved:
        return 0.0
    relevant_retrieved = sum(1 for doc_id in retrieved if doc_id in relevant)
    return relevant_retrieved / len(retrieved)


def compute_precision_at_k(retrieved: List[str], relevant: set, k: int) -> float:
    """
    Precision@K (Lecture 4):
    "Calculates precision for only K documents. Considers the top K recommendations."
    Formula: |Relevant in top-K| / K
    """
    top_k = retrieved[:k]
    if not top_k:
        return 0.0
    relevant_in_k = sum(1 for doc_id in top_k if doc_id in relevant)
    return relevant_in_k / k


def compute_recall(retrieved: List[str], relevant: set) -> float:
    """
    Recall (Lecture 4):
    "Measures how many relevant items were returned against
     how many relevant items exist in the entire dataset."
    Formula: |Relevant ∩ Retrieved| / |All Relevant|
    """
    if not relevant:
        return 0.0
    relevant_retrieved = sum(1 for doc_id in retrieved if doc_id in relevant)
    return relevant_retrieved / len(relevant)


def compute_average_precision(retrieved: List[str], relevant: set, k: int) -> float:
    """
    Average Precision@K (Lecture 4):
    Accounts for the ranking order — rewards relevant docs appearing early.
    Formula: AP@K = (1/R) × Σ P@k × rel(k)
    where R = total relevant docs, rel(k) = 1 if doc at rank k is relevant.

    Lecture: "MAP is the mean of the Average Precision scores for each query."
    """
    if not relevant:
        return 0.0

    precision_sum = 0.0
    relevant_found = 0

    for rank, doc_id in enumerate(retrieved[:k], start=1):
        if doc_id in relevant:
            relevant_found += 1
            precision_at_rank = relevant_found / rank
            precision_sum += precision_at_rank

    total_relevant = len(relevant)
    return precision_sum / total_relevant if total_relevant > 0 else 0.0


def compute_dcg(retrieved: List[str], graded_relevance: Dict[str, int], k: int) -> float:
    """
    DCG - Discounted Cumulative Gain (Lecture 4):
    "nDCG rewards relevant items at higher ranks more."
    Formula: DCG@K = Σ rel(i) / log2(i+1)   for i in 1..K

    Graded relevance: each doc gets a relevance score (0, 1, 2, ...)
    Higher ranked relevant docs contribute more to the score.
    """
    dcg = 0.0
    for rank, doc_id in enumerate(retrieved[:k], start=1):
        rel = graded_relevance.get(doc_id, 0)
        # Discount: 1 / log2(rank + 1)
        dcg += rel / math.log2(rank + 1)
    return dcg


def compute_ndcg(retrieved: List[str], graded_relevance: Dict[str, int], k: int) -> float:
    """
    nDCG - Normalized DCG (Lecture 4):
    Normalizes DCG by the ideal ranking (IDCG).
    Formula: nDCG@K = DCG@K / IDCG@K

    IDCG = DCG of the ideal ranked list (most relevant docs ranked first).

    Lecture: "nDCG rewards relevant items at higher ranks.
              Use nDCG if you have graded relevance and care about top-ranked items."
    Range: 0.0 (worst) to 1.0 (perfect ranking)
    """
    if not graded_relevance:
        return 0.0

    # Actual DCG
    dcg = compute_dcg(retrieved, graded_relevance, k)

    # Ideal DCG: sort all relevant docs by relevance score descending
    ideal_order = sorted(graded_relevance.keys(),
                         key=lambda d: graded_relevance[d], reverse=True)
    idcg = compute_dcg(ideal_order, graded_relevance, k)

    if idcg == 0:
        return 0.0
    return dcg / idcg


# ──────────────────────────────────────────────
# Result Re-ranking (Score-based)
# ──────────────────────────────────────────────

def rerank_by_score(docs: List[RetrievedDoc]) -> List[RetrievedDoc]:
    """
    Re-rank documents by their retrieval score (descending).
    Assigns fresh ranks after sorting.
    Lecture 3: "Rank results based on similarity values; the nearest, the better."
    """
    sorted_docs = sorted(docs, key=lambda d: d.score, reverse=True)
    for i, doc in enumerate(sorted_docs, start=1):
        doc.rank = i
    return sorted_docs


# ──────────────────────────────────────────────
# Main Evaluation Logic
# ──────────────────────────────────────────────

def evaluate_single_query(
    query_id: str,
    retrieved_docs: List[RetrievedDoc],
    qrels: List[QrelEntry],
    k: int = 10,
) -> MetricsResult:
    """
    Compute all IR metrics for a single query.

    Binary relevant set (for Precision, Recall, MAP):
        All docs with relevance >= 1
    Graded relevance dict (for nDCG):
        {doc_id: relevance_score}
    """
    # Build relevant set and graded relevance from qrels
    relevant_set = set()
    graded_relevance: Dict[str, int] = {}

    for qrel in qrels:
        if qrel.query_id == query_id:
            graded_relevance[qrel.doc_id] = qrel.relevance
            if qrel.relevance >= 1:
                relevant_set.add(qrel.doc_id)

    # Sort retrieved docs by rank
    sorted_retrieved = sorted(retrieved_docs, key=lambda d: d.rank)
    retrieved_ids = [d.doc_id for d in sorted_retrieved]

    # Compute all metrics
    precision = compute_precision(retrieved_ids, relevant_set)
    precision_at_k = compute_precision_at_k(retrieved_ids, relevant_set, k)
    recall = compute_recall(retrieved_ids, relevant_set)
    avg_precision = compute_average_precision(retrieved_ids, relevant_set, k)
    ndcg = compute_ndcg(retrieved_ids, graded_relevance, k)

    relevant_retrieved = sum(1 for doc_id in retrieved_ids if doc_id in relevant_set)

    return MetricsResult(
        query_id=query_id,
        precision=round(precision, 4),
        precision_at_k=round(precision_at_k, 4),
        recall=round(recall, 4),
        average_precision=round(avg_precision, 4),
        ndcg_at_k=round(ndcg, 4),
        relevant_retrieved=relevant_retrieved,
        total_relevant=len(relevant_set),
        total_retrieved=len(retrieved_docs),
    )


# ──────────────────────────────────────────────
# API Endpoints
# ──────────────────────────────────────────────

@app.get("/health")
def health():
    return {"status": "ok", "service": "ranking_evaluation"}


@app.post("/evaluate", response_model=MetricsResult)
def evaluate_query(req: EvaluationRequest):
    """
    Evaluate retrieval results for a single query.
    Returns: Precision, P@K, Recall, AP@K, nDCG@K
    """
    return evaluate_single_query(
        query_id=req.query_id,
        retrieved_docs=req.retrieved_docs,
        qrels=req.qrels,
        k=req.k,
    )


@app.post("/evaluate/batch", response_model=BatchMetricsResult)
def evaluate_batch(req: BatchEvaluationRequest):
    """
    Evaluate across multiple queries.
    Computes per-query metrics + aggregate MAP, mean nDCG, mean P@K.

    MAP (Lecture 4):
        "Mean Average Precision for a set of queries is the mean
         of the Average Precision scores for each query."
        Formula: MAP = (1/Q) × Σ AP@K
    """
    per_query_results = []

    for eval_req in req.evaluations:
        result = evaluate_single_query(
            query_id=eval_req.query_id,
            retrieved_docs=eval_req.retrieved_docs,
            qrels=eval_req.qrels,
            k=req.k,
        )
        per_query_results.append(result)

    # Aggregate metrics
    Q = len(per_query_results)
    if Q == 0:
        return BatchMetricsResult(
            per_query=[], map_score=0.0,
            mean_ndcg=0.0, mean_precision_at_k=0.0, mean_recall=0.0,
        )

    map_score = sum(r.average_precision for r in per_query_results) / Q
    mean_ndcg = sum(r.ndcg_at_k for r in per_query_results) / Q
    mean_pk = sum(r.precision_at_k for r in per_query_results) / Q
    mean_recall = sum(r.recall for r in per_query_results) / Q

    return BatchMetricsResult(
        per_query=per_query_results,
        map_score=round(map_score, 4),
        mean_ndcg=round(mean_ndcg, 4),
        mean_precision_at_k=round(mean_pk, 4),
        mean_recall=round(mean_recall, 4),
    )


@app.post("/rerank", response_model=List[RetrievedDoc])
def rerank(docs: List[RetrievedDoc]):
    """
    Re-rank a list of retrieved documents by their score.
    Returns the list sorted by score (highest first) with updated ranks.
    """
    return rerank_by_score(docs)


@app.get("/metrics/description")
def metrics_description():
    """Return description of all evaluation metrics implemented."""
    return {
        "metrics": [
            {
                "name": "Precision",
                "formula": "|Relevant ∩ Retrieved| / |Retrieved|",
                "lecture": "Lecture 4",
                "description": "Proportion of retrieved documents that are relevant",
            },
            {
                "name": "Precision@K",
                "formula": "|Relevant in top-K| / K",
                "lecture": "Lecture 4",
                "description": "Precision computed only on the top K results",
            },
            {
                "name": "Recall",
                "formula": "|Relevant ∩ Retrieved| / |All Relevant|",
                "lecture": "Lecture 4",
                "description": "How many relevant items were found out of all relevant items",
            },
            {
                "name": "MAP",
                "formula": "(1/Q) × Σ AP@K per query",
                "lecture": "Lecture 4",
                "description": "Mean Average Precision across all queries — good for binary relevance",
            },
            {
                "name": "nDCG@K",
                "formula": "DCG@K / IDCG@K",
                "lecture": "Lecture 4",
                "description": "Normalized Discounted Cumulative Gain — rewards relevant docs at top ranks, supports graded relevance",
            },
        ]
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8004)