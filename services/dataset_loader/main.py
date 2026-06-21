"""
Dataset Loader Service
=====================================================================
المسؤولية:
    - تحميل Datasets من ir-datasets
    - Sampling منها بعدد مناسب (200K+ docs حسب المشروع)
    - استدعاء Preprocessing → Indexing → Embedding بالترتيب
    - حفظ الـ qrels لاستخدامها في التقييم

الـ Datasets المختارة (حسب شروط المشروع):
    ✓ أكثر من 200K document
    ✓ تحتوي على qrels (ground truth للتقييم)
    ✓ ليست Antique (ممنوعة)

    Dataset 1: msmarco-passage/dev/small
        - 8.8M passages (نأخذ 200K sample)
        - 6,980 queries مع qrels
        - الأشهر في مجال IR

    Dataset 2: beir/nq  (Natural Questions)
        - 2.6M docs (نأخذ 200K sample)
        - Wikipedia-based QA
        - تحتوي على qrels

SOA Role: خدمة مستقلة تُنسّق بين جميع خدمات النظام لتحميل وفهرسة البيانات.
"""

from fastapi import FastAPI, BackgroundTasks, HTTPException
from pydantic import BaseModel
from typing import Optional, List, Dict
import httpx
import asyncio
import json
import os
from pathlib import Path
from datetime import datetime

app = FastAPI(
    title="Dataset Loader Service",
    description="Loads ir-datasets, preprocesses, and indexes them through all services",
    version="1.0.0",
)
from fastapi.middleware.cors import CORSMiddleware
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])
# ── Service URLs ──
GATEWAY_URL        = "http://localhost:8000"
INDEXING_URL       = "http://localhost:8002"
EMBEDDING_URL      = "http://localhost:8006"
DOCUMENT_STORE_URL = "http://localhost:8009"

DATA_DIR = Path("data")
DATA_DIR.mkdir(exist_ok=True)

# ── Loading progress tracker ──
loading_status: Dict[str, Dict] = {}


# ──────────────────────────────────────────────
# Supported Datasets
# ──────────────────────────────────────────────
SUPPORTED_DATASETS = {
  "touche": {
        "ir_datasets_id": "beir/webis-touche2020",
        "display_name": "Webis-Touché 2020 (BEIR — Argument Retrieval)",
        "description": "382K argumentative documents. BEIR benchmark task for argument retrieval.",
        "doc_field": "text",
        "min_docs": 200_000,
        "has_qrels": True,
    },
}


# ──────────────────────────────────────────────
# Schemas
# ──────────────────────────────────────────────
class LoadRequest(BaseModel):
    dataset_name: str           # "msmarco" | "nq"
    max_docs: int = 200_000     # how many docs to index (sample)
    max_queries: int = 500      # how many queries for evaluation
    build_embeddings: bool = True
    embedding_model: str = "sbert"   # "sbert" | "word2vec" | "both"
    batch_size: int = 500       # docs per indexing batch


class LoadStatus(BaseModel):
    dataset_name: str
    status: str        # "idle" | "loading" | "indexing" | "embedding" | "done" | "error"
    progress: float    # 0.0 - 1.0
    message: str
    docs_loaded: int
    docs_indexed: int
    queries_loaded: int
    qrels_loaded: int
    started_at: Optional[str]
    finished_at: Optional[str]
    error: Optional[str]


# ──────────────────────────────────────────────
# Core Loading Pipeline
# ──────────────────────────────────────────────

async def load_and_index_dataset(dataset_name: str, req: LoadRequest):
    """
    Full pipeline:
    1. Load docs from ir-datasets
    2. Batch-send to Indexing Service (Inverted Index + TF-IDF)
    3. Build Embedding Index (SBERT/Word2Vec + FAISS)
    4. Save qrels to disk for evaluation
    """
    status = loading_status[dataset_name]
    cfg = SUPPORTED_DATASETS[dataset_name]

    try:
        import ir_datasets

        # ── Step 1: Load dataset ──
        status.update({"status": "loading", "message": "Connecting to ir-datasets...", "progress": 0.02})
        ds = ir_datasets.load(cfg["ir_datasets_id"])

        # ── Step 2: Load documents in batches ──
        status.update({"message": f"Loading up to {req.max_docs:,} documents...", "progress": 0.05})

        all_docs = []        # نسخة مُقتطَعة (2000 حرف) — للفهرسة وembedding فقط
        all_doc_ids = []
        all_texts = []        # نفس المُقتطَعة — تُستخدَم للـ embedding
        all_raw_docs = []     # ★ النص الكامل (raw) — يُخزَّن في Document Store

        for i, doc in enumerate(ds.docs_iter()):
            if i >= req.max_docs:
                break

            raw_text = getattr(doc, cfg["doc_field"], "") or ""
            title = getattr(doc, "title", "") or ""

            if len(raw_text.strip()) < 10:   # skip very short docs
                continue

            doc_id = str(doc.doc_id)

        
            # نسخة مُقتطَعة (2000 حرف) — تكفي للفهرسة (TF-IDF/BM25) وembedding
            # ملاحظة: هذه النسخة لا تُعرَض للمستخدم أبداً، فقط تُستخدَم
            # لبناء الـ indexes الخفيفة (Lecture 2 - فلسفة الفهرسة)
            capped_text = raw_text
            all_docs.append({
                "doc_id": doc_id,
                "text": capped_text,
                "title": title[:200],
                "metadata": {},
            }) 
            all_doc_ids.append(doc_id)
            all_texts.append(capped_text)

            if (i + 1) % 10_000 == 0:
                progress = 0.05 + (i / req.max_docs) * 0.30
                status.update({
                    "docs_loaded": i + 1,
                    "progress": round(progress, 3),
                    "message": f"Loaded {i+1:,} / {req.max_docs:,} documents...",
                })

        status["docs_loaded"] = len(all_docs)

        # ── Step 2.5: ★ تخزين النصوص الأصلية (raw) في Document Store ──
        # هذا منفصل عن الفهرسة — قاعدة بيانات SQLite بـ doc_id كـ Primary Key.
        # سيُستعلَم منها فقط في "آخر خطوة" عند عرض top-K نتائج البحث.

        status.update({"message": f"Loaded {len(all_docs):,} docs. Starting indexing...", "progress": 0.40})

        # ── Step 3: Index in batches (Inverted Index + TF-IDF) ──
        status["status"] = "indexing"
        batch_size = req.batch_size
        total_indexed = 0

        async with httpx.AsyncClient(timeout=300.0) as client:
            for start in range(0, len(all_docs), batch_size):
                batch = all_docs[start:start + batch_size]

                r = await client.post(f"{INDEXING_URL}/index", json={"dataset_id": dataset_name, "documents": batch})

                if r.status_code != 200:
                  raise Exception(f"Indexing failed at batch {start}: {r.text[:200]}")

                result = r.json()
                actual_indexed = result.get("indexed", 0)
                if actual_indexed != len(batch):
                 raise Exception(f"⚠️ توقّعنا فهرسة {len(batch)} وثيقة لكن فعلياً فُهرس {actual_indexed} فقط عند batch {start}")
                total_indexed += actual_indexed
                
                progress = 0.40 + (total_indexed / len(all_docs)) * 0.30
                status.update({
                    "docs_indexed": total_indexed,
                    "progress": round(progress, 3),
                    "message": f"Indexed {total_indexed:,} / {len(all_docs):,} docs...",
                })

        status.update({"message": "Inverted index built. Building embedding index...", "progress": 0.70})

        # ── Step 4: Build Embedding Index (SBERT/Word2Vec + FAISS) ──
        if req.build_embeddings:
            status["status"] = "embedding"
            status.update({"message": f"Encoding {len(all_docs):,} docs with {req.embedding_model}...", "progress": 0.72})

            # Send in batches to avoid timeout
            emb_batch = 5_000
            for start in range(0, len(all_docs), emb_batch):
                batch_ids   = all_doc_ids[start:start + emb_batch]
                batch_texts = all_texts[start:start + emb_batch]

                async with httpx.AsyncClient(timeout=600.0) as client:
                    r = await client.post(
                        f"{EMBEDDING_URL}/index",
                        json={
                            "dataset_id": dataset_name,
                            "doc_ids": batch_ids,
                            "texts": batch_texts,
                            "model": req.embedding_model,
                        },
                    )
                    if r.status_code != 200:
                        raise Exception(f"Embedding indexing failed: {r.text[:200]}")

                progress = 0.72 + ((start + emb_batch) / len(all_docs)) * 0.18
                status.update({
                    "progress": min(round(progress, 3), 0.90),
                    "message": f"Embedded {min(start+emb_batch, len(all_docs)):,} / {len(all_docs):,} docs...",
                })

        # ── Step 5: Load queries ──
        status.update({"message": "Loading queries...", "progress": 0.91})
        queries = {}
        for i, q in enumerate(ds.queries_iter()):
            if i >= req.max_queries:
                break
            queries[str(q.query_id)] = q.text
        status["queries_loaded"] = len(queries)

        # ── Step 6: Load qrels ──
        status.update({"message": "Loading qrels...", "progress": 0.94})
        qrels = []
        query_ids_set = set(queries.keys())
        for qrel in ds.qrels_iter():
            if str(qrel.query_id) in query_ids_set:
                qrels.append({
                    "query_id": str(qrel.query_id),
                    "doc_id": str(qrel.doc_id),
                    "relevance": int(qrel.relevance),
                })
        status["qrels_loaded"] = len(qrels)

        # ── Step 7: Save queries + qrels to disk ──
        save_path = DATA_DIR / dataset_name
        save_path.mkdir(exist_ok=True)

        with open(save_path / "queries.json", "w") as f:
            json.dump(queries, f, ensure_ascii=False, indent=2)

        with open(save_path / "qrels.json", "w") as f:
            json.dump(qrels, f, ensure_ascii=False, indent=2)

        # Save dataset metadata
        meta = {
            "dataset_name": dataset_name,
            "ir_datasets_id": cfg["ir_datasets_id"],
            "display_name": cfg["display_name"],
            "docs_indexed": total_indexed,
            "queries_count": len(queries),
            "qrels_count": len(qrels),
            "embedding_model": req.embedding_model if req.build_embeddings else None,
            "loaded_at": datetime.now().isoformat(),
        }
        with open(save_path / "meta.json", "w") as f:
            json.dump(meta, f, indent=2)

        # ── Done ──
        status.update({
            "status": "done",
            "progress": 1.0,
            "message": f"✅ Dataset '{dataset_name}' fully loaded and indexed!",
            "finished_at": datetime.now().isoformat(),
        })

    except Exception as e:
        import traceback
        status.update({
            "status": "error",
            "progress": status.get("progress", 0),
            "message": "Loading failed.",
            "error": f"{str(e)}\n{traceback.format_exc()}",
            "finished_at": datetime.now().isoformat(),
        })


# ──────────────────────────────────────────────
# API Endpoints
# ──────────────────────────────────────────────

@app.get("/health")
def health():
    return {"status": "ok", "service": "dataset_loader"}


@app.get("/datasets")
def list_datasets():
    """List all supported datasets with their metadata."""
    result = []
    for name, cfg in SUPPORTED_DATASETS.items():
        save_path = DATA_DIR / name / "meta.json"
        is_loaded = save_path.exists()
        meta = {}
        if is_loaded:
            with open(save_path) as f:
                meta = json.load(f)

        result.append({
            "name": name,
            "display_name": cfg["display_name"],
            "description": cfg["description"],
            "ir_datasets_id": cfg["ir_datasets_id"],
            "min_docs": cfg["min_docs"],
            "has_qrels": cfg["has_qrels"],
            "is_loaded": is_loaded,
            "loaded_meta": meta,
        })
    return {"datasets": result}


@app.post("/load/{dataset_name}")
async def load_dataset(dataset_name: str, req: LoadRequest, background_tasks: BackgroundTasks):
    """
    Start loading and indexing a dataset in the background.
    Returns immediately — use GET /status/{dataset_name} to track progress.

    Pipeline: ir-datasets → Preprocessing → Indexing → Embedding → qrels saved
    """
    if dataset_name not in SUPPORTED_DATASETS:
        raise HTTPException(400, f"Unknown dataset '{dataset_name}'. Available: {list(SUPPORTED_DATASETS.keys())}")

    if dataset_name in loading_status and loading_status[dataset_name]["status"] in ("loading", "indexing", "embedding"):
        return {"message": f"Dataset '{dataset_name}' is already loading.", "status": loading_status[dataset_name]}

    req.dataset_name = dataset_name

    loading_status[dataset_name] = {
        "dataset_name": dataset_name,
        "status": "loading",
        "progress": 0.0,
        "message": "Starting...",
        "docs_loaded": 0,
        "docs_indexed": 0,
        "queries_loaded": 0,
        "qrels_loaded": 0,
        "started_at": datetime.now().isoformat(),
        "finished_at": None,
        "error": None,
    }

    background_tasks.add_task(load_and_index_dataset, dataset_name, req)

    return {
        "message": f"Loading '{dataset_name}' started in background.",
        "track_progress": f"GET /status/{dataset_name}",
    }


@app.get("/status/{dataset_name}", response_model=LoadStatus)
def get_status(dataset_name: str):
    """Get the loading progress for a dataset."""
    if dataset_name not in loading_status:
        # Check if already saved to disk
        save_path = DATA_DIR / dataset_name / "meta.json"
        if save_path.exists():
            with open(save_path) as f:
                meta = json.load(f)
            return LoadStatus(
                dataset_name=dataset_name,
                status="done",
                progress=1.0,
                message="Dataset already loaded.",
                docs_loaded=meta.get("docs_indexed", 0),
                docs_indexed=meta.get("docs_indexed", 0),
                queries_loaded=meta.get("queries_count", 0),
                qrels_loaded=meta.get("qrels_count", 0),
                started_at=meta.get("loaded_at"),
                finished_at=meta.get("loaded_at"),
                error=None,
            )
        raise HTTPException(404, f"No loading initiated for '{dataset_name}'.")

    s = loading_status[dataset_name]
    return LoadStatus(**s)


@app.get("/qrels/{dataset_name}")
def get_qrels(dataset_name: str, limit: int = 1000):
    """Return qrels for a dataset (used by Evaluation Service)."""
    path = DATA_DIR / dataset_name / "qrels.json"
    if not path.exists():
        raise HTTPException(404, f"Qrels not found for '{dataset_name}'. Load the dataset first.")
    with open(path) as f:
        qrels = json.load(f)
    return {"dataset": dataset_name, "total": len(qrels), "qrels": qrels[:limit]}


@app.get("/queries/{dataset_name}")
def get_queries(dataset_name: str, limit: int = 100):
    """Return queries for a dataset."""
    path = DATA_DIR / dataset_name / "queries.json"
    if not path.exists():
        raise HTTPException(404, f"Queries not found for '{dataset_name}'. Load the dataset first.")
    with open(path) as f:
        queries = json.load(f)
    items = list(queries.items())[:limit]
    return {"dataset": dataset_name, "total": len(queries), "queries": dict(items)}


@app.delete("/reset/{dataset_name}")
def reset_dataset(dataset_name: str):
    """Remove loaded dataset data (qrels, queries, meta) from disk."""
    import shutil
    path = DATA_DIR / dataset_name
    if path.exists():
        shutil.rmtree(path)
    loading_status.pop(dataset_name, None)
    return {"reset": True, "dataset": dataset_name}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8007)