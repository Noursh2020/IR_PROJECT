"""
Indexing Service
=====================================================================
يستخدم asyncpg مباشرة للأداء العالي عند فهرسة ملايين الوثائق.

جداول PostgreSQL المستخدمة:
    documents  — doc_id (PK), title, content, metadata
    terms      — id (serial PK), term (UNIQUE)
    postings   — term_id (FK), doc_id (FK), tf
"""

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Dict, Optional
import asyncpg
import httpx

app = FastAPI(title="Indexing Service", version="1.0.0")

PREPROCESSING_URL = "http://localhost:8001"
DB_POOL = None

# ══════════════════════════════════════════════
# DB Connection
# ══════════════════════════════════════════════

async def init_db():
    global DB_POOL
    DB_POOL = await asyncpg.create_pool(
        user="postgres",
        password="root",
        database="ir_db",
        host="localhost",
        port=5432,
        min_size=2,
        max_size=10,
    )


@app.on_event("startup")
async def startup():
    await init_db()


@app.on_event("shutdown")
async def shutdown():
    if DB_POOL:
        await DB_POOL.close()


# ══════════════════════════════════════════════
# Schemas
# ══════════════════════════════════════════════

class Document(BaseModel):
    doc_id: str
    text: str
    title: Optional[str] = ""
    metadata: Optional[Dict] = {}


class IndexRequest(BaseModel):
    dataset_id: str
    documents: List[Document]


class SearchRequest(BaseModel):
    terms: List[str]
    top_k: int = 50


# ══════════════════════════════════════════════
# DB Helpers
# ══════════════════════════════════════════════

async def save_doc(conn, doc: Document, dataset_id: str):
    import json
    await conn.execute("""
        INSERT INTO documents (doc_id, title, content, metadata)
        VALUES ($1, $2, $3, $4)
        ON CONFLICT (doc_id) DO NOTHING
    """, doc.doc_id, doc.title or "", doc.text, json.dumps(doc.metadata or {}))


async def get_or_create_term_id(conn, term: str) -> int:
    """
    أرجع term_id من جدول terms.
    إذا المصطلح غير موجود → أضفه وأرجع الـ ID الجديد.
    """
    row = await conn.fetchrow(
        "SELECT id FROM terms WHERE term = $1", term
    )
    if row:
        return row["id"]

    row = await conn.fetchrow(
        "INSERT INTO terms (term) VALUES ($1) ON CONFLICT (term) DO UPDATE SET term = EXCLUDED.term RETURNING id",
        term
    )
    return row["id"]


async def save_postings(conn, term_tf_map: Dict[str, int], doc_id: str):
    """حفظ كل مصطلحات وثيقة واحدة دفعة واحدة."""
    for term, tf in term_tf_map.items():
        term_id = await get_or_create_term_id(conn, term)
        await conn.execute("""
            INSERT INTO postings (term_id, doc_id, tf)
            VALUES ($1, $2, $3)
            ON CONFLICT (term_id, doc_id)
            DO UPDATE SET tf = EXCLUDED.tf
        """, term_id, doc_id, tf)


async def save_progress(conn, dataset_id: str, last_index: int):
    """حفظ checkpoint للفهرسة."""
    await conn.execute("""
        INSERT INTO indexing_progress (dataset_id, last_index)
        VALUES ($1, $2)
        ON CONFLICT (dataset_id)
        DO UPDATE SET last_index = EXCLUDED.last_index
    """, dataset_id, last_index)


async def get_progress(dataset_id: str) -> int:
    """أرجع آخر index تمت فهرسته (0 إذا لا يوجد)."""
    async with DB_POOL.acquire() as conn:
        row = await conn.fetchrow(
            "SELECT last_index FROM indexing_progress WHERE dataset_id = $1",
            dataset_id
        )
        return row["last_index"] if row else 0


# ══════════════════════════════════════════════
# Indexing Endpoint
# ══════════════════════════════════════════════

@app.post("/index")
async def index_documents(req: IndexRequest):
    """
    فهرسة مجموعة وثائق:
    1. Preprocess النص → tokens
    2. احسب TF لكل مصطلح
    3. احفظ في documents + terms + postings
    """
    BATCH = 20
    start = await get_progress(req.dataset_id)
    total_indexed = 0

    async with httpx.AsyncClient(timeout=300) as client:
        for i in range(start, len(req.documents), BATCH):
            batch = req.documents[i:i + BATCH]
            texts = [d.text for d in batch]

            # ── Preprocessing ──
            res = await client.post(
                f"{PREPROCESSING_URL}/preprocess/batch",
                json={
                    "texts": texts,
                    "use_lemmatization": True,
                    "remove_stopwords": True,
                }
            )
            if res.status_code != 200:
                raise HTTPException(500, f"Preprocessing failed: {res.text}")

            processed = res.json()["results"]

            # ── Save to DB ──
            async with DB_POOL.acquire() as conn:
                async with conn.transaction():
                    for doc, prep in zip(batch, processed):
                        # 1. حفظ الوثيقة
                        await save_doc(conn, doc, req.dataset_id)

                        # 2. احسب TF
                        tokens = prep["processed_tokens"]
                        term_tf: Dict[str, int] = {}
                        for token in tokens:
                            term_tf[token] = term_tf.get(token, 0) + 1

                        # 3. حفظ المصطلحات والـ postings
                        await save_postings(conn, term_tf, doc.doc_id)

                    # 4. حفظ التقدم
                    await save_progress(conn, req.dataset_id, i + len(batch))

            total_indexed += len(batch)
            print(f"[Indexing] Batch {i}–{i+len(batch)} done ({total_indexed} total)")

    return {
        "status": "completed",
        "dataset_id": req.dataset_id,
        "indexed": total_indexed,
    }


# ══════════════════════════════════════════════
# Search Endpoint (للـ Retrieval Service)
# ══════════════════════════════════════════════

@app.post("/search/inverted")
async def search_inverted(req: SearchRequest):
    """
    بحث في الـ inverted index عن الوثائق التي تحتوي على المصطلحات.
    يُستخدم من Retrieval Service لحساب BM25 / TF-IDF.
    """
    async with DB_POOL.acquire() as conn:
        rows = await conn.fetch("""
            SELECT p.doc_id,
                   SUM(p.tf) AS total_tf,
                   json_object_agg(t.term, p.tf) AS term_freqs
            FROM postings p
            JOIN terms t ON t.id = p.term_id
            WHERE t.term = ANY($1::text[])
            GROUP BY p.doc_id
            ORDER BY total_tf DESC
            LIMIT $2
        """, req.terms, req.top_k)

    candidates = []
    for row in rows:
        import json
        candidates.append({
            "doc_id": row["doc_id"],
            "total_tf": row["total_tf"],
            "term_freqs": json.loads(row["term_freqs"]),
        })

    return {"candidates": candidates, "total": len(candidates)}


# ══════════════════════════════════════════════
# Stats Endpoint
# ══════════════════════════════════════════════

@app.get("/stats")
async def get_stats():
    """إحصائيات الفهرس — تُستخدم من Retrieval Service لحساب IDF."""
    async with DB_POOL.acquire() as conn:
        doc_count = await conn.fetchval("SELECT COUNT(*) FROM documents")
        term_count = await conn.fetchval("SELECT COUNT(*) FROM terms")
        posting_count = await conn.fetchval("SELECT COUNT(*) FROM postings")

        # متوسط طول الوثائق (بالكلمات)
        avg_dl = await conn.fetchval("""
            SELECT COALESCE(AVG(tf_sum), 1)
            FROM (
                SELECT doc_id, SUM(tf) AS tf_sum
                FROM postings
                GROUP BY doc_id
            ) t
        """)

    return {
        "total_documents": doc_count,
        "total_terms": term_count,
        "total_postings": posting_count,
        "avg_doc_length": float(avg_dl or 1),
    }


@app.get("/health")
def health():
    return {"status": "ok", "service": "indexing"}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8002)