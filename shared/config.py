"""
Shared configuration for the IR System.
================================================
IR Concept (Lecture 1 - IR Life Cycle):
    The system is designed following the IR Life Cycle:
    Documents → Text Processing → Indexing → Query Processing → Matching → Ranking → Evaluation

IR Concept (Project Spec - SOA):
    Each service is independent and communicates via REST API,
    following Service-Oriented Architecture (SOA).
"""

import os
from pathlib import Path

# ──────────────────────────────────────────────
# Project Paths
# ──────────────────────────────────────────────
BASE_DIR = Path(__file__).parent.parent
DATA_DIR = BASE_DIR / "data"
INDEX_DIR = BASE_DIR / "indexes"
MODELS_DIR = BASE_DIR / "models"

DATA_DIR.mkdir(parents=True, exist_ok=True)
INDEX_DIR.mkdir(parents=True, exist_ok=True)
MODELS_DIR.mkdir(parents=True, exist_ok=True)

# ──────────────────────────────────────────────
# Supported Datasets (from ir-datasets.com)
# ──────────────────────────────────────────────
DATASETS = {
    "msmarco": {
        "name": "MS MARCO Passage Ranking",
        "description": "~8.8M passages, widely used for passage retrieval",
        "dataset_id": "msmarco-passage",
        "has_qrels": True,
    },
    "beir_scifact": {
        "name": "BEIR SciFact",
        "description": "Scientific fact-checking corpus",
        "dataset_id": "beir/scifact",
        "has_qrels": True,
    },
}

# ──────────────────────────────────────────────
# BM25 Parameters (Lecture 3 - Saturation Effect)
# ──────────────────────────────────────────────
# IR Concept: k1 controls term frequency saturation.
#   Higher k1 → TF has more influence (slower saturation)
#   b controls document length normalization.
#   Higher b → long documents are penalized more.
BM25_DEFAULT_K1 = 1.5
BM25_DEFAULT_B = 0.75

# ──────────────────────────────────────────────
# Hybrid Retrieval Config (Lecture 3 - Hybrid Models)
# ──────────────────────────────────────────────
# IR Concept: Hybrid = BM25 (symbolic) + Embeddings (neural).
#   Serial: BM25 filters candidates → Neural re-ranks
#   Parallel: Both run simultaneously → Fusion merges results
HYBRID_MODE = "serial"          # "serial" | "parallel"
HYBRID_SERIAL_CANDIDATES = 1000  # BM25 top-k before re-ranking
HYBRID_PARALLEL_WEIGHTS = {     # for weighted fusion
    "bm25": 0.5,
    "embedding": 0.5,
}
RRF_K = 60                       # RRF constant (Lecture 3 - RRF)

# ──────────────────────────────────────────────
# Embedding Model Config (Project Spec)
# ──────────────────────────────────────────────
EMBEDDING_MODEL = "all-MiniLM-L6-v2"   # sentence-transformers
EMBEDDING_BATCH_SIZE = 256

# ──────────────────────────────────────────────
# Retrieval Defaults
# ──────────────────────────────────────────────
DEFAULT_TOP_K = 10
SIMILARITY_THRESHOLD = 0.0       # Lecture 3: filter by cosine threshold

# ──────────────────────────────────────────────
# Service Ports (SOA - each service on its own port)
# ──────────────────────────────────────────────
PORTS = {
    "gateway":           8000,
    "preprocessing":     8001,
    "indexing":          8002,
    "retrieval":         8003,
    "ranking_evaluation": 8004,
    "query_refinement":  8005,
}