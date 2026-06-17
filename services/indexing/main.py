from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Dict, Optional
import asyncpg
import httpx

app = FastAPI()

PREPROCESSING_URL = "http://localhost:8001"
DB_POOL = None


# ==========================
# DATABASE
# ==========================

async def init_db():
    global DB_POOL

    DB_POOL = await asyncpg.create_pool(
        user="postgres",
        password="root",
        database="ir_db",
        host="localhost",
        port=5432,
        min_size=1,
        max_size=10
    )


@app.on_event("startup")
async def startup():
    await init_db()


@app.on_event("shutdown")
async def shutdown():
    global DB_POOL

    if DB_POOL:
        await DB_POOL.close()


# ==========================
# SCHEMAS
# ==========================

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

# ─────────────────────────────
# Helpers (DB)
# ─────────────────────────────
async def save_doc(doc: Document, dataset_id: str):
    async with DB_POOL.acquire() as conn:
        await conn.execute("""
            INSERT INTO documents(doc_id, dataset_id, text, title, metadata)
            VALUES($1,$2,$3,$4,$5)
            ON CONFLICT DO NOTHING
        """, doc.doc_id, dataset_id, doc.text, doc.title, doc.metadata)


async def save_posting(term: str, doc_id: str, tf: int):
    async with DB_POOL.acquire() as conn:
        await conn.execute("""
            INSERT INTO inverted_index(term, doc_id, tf)
            VALUES($1,$2,$3)
            ON CONFLICT (term, doc_id)
            DO UPDATE SET tf = EXCLUDED.tf
        """, term, doc_id, tf)


async def save_progress(dataset_id: str, idx: int):
    async with DB_POOL.acquire() as conn:
        await conn.execute("""
            INSERT INTO indexing_progress(dataset_id, last_index)
            VALUES($1,$2)
            ON CONFLICT(dataset_id)
            DO UPDATE SET last_index = $2
        """, dataset_id, idx)


async def get_progress(dataset_id: str):
    async with DB_POOL.acquire() as conn:
        row = await conn.fetchrow("""
            SELECT last_index FROM indexing_progress
            WHERE dataset_id=$1
        """, dataset_id)

        return row["last_index"] if row else 0


# ─────────────────────────────
# INDEXING (MAIN FIX)
# ─────────────────────────────
@app.post("/index")
async def index_documents(req: IndexRequest):

    BATCH = 20
    start = await get_progress(req.dataset_id)

    async with httpx.AsyncClient(timeout=300) as client:

        for i in range(start, len(req.documents), BATCH):

            batch = req.documents[i:i+BATCH]
            texts = [d.text for d in batch]

            # preprocess
            res = await client.post(
                f"{PREPROCESSING_URL}/preprocess/batch",
                json={
                    "texts": texts,
                    "use_lemmatization": True,
                    "remove_stopwords": True
                }
            )

            if res.status_code != 200:
                raise HTTPException(500, res.text)

            processed = res.json()["results"]

            # store docs + index
            for doc, prep in zip(batch, processed):

                await save_doc(doc, req.dataset_id)

                tokens = prep["processed_tokens"]

                freq = {}
                for t in tokens:
                    freq[t] = freq.get(t, 0) + 1

                for term, tf in freq.items():
                    await save_posting(term, doc.doc_id, tf)

            await save_progress(
            req.dataset_id,
            i + len(batch)
      )

            print(f"Batch {i} done")

    return {
        "status": "completed",
        "dataset": req.dataset_id
    }


# ─────────────────────────────
# SEARCH (DB ONLY)
# ─────────────────────────────
@app.post("/search/inverted")
async def search(req: SearchRequest):

    async with DB_POOL.acquire() as conn:

        rows = await conn.fetch("""
            SELECT doc_id, SUM(tf) as score
            FROM inverted_index
            WHERE term = ANY($1)
            GROUP BY doc_id
            ORDER BY score DESC
            LIMIT $2
        """, req.terms, req.top_k)

    return {"results": rows}