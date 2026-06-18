from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Optional
import psycopg2
from psycopg2.extras import execute_values

app = FastAPI(title="Document Store Service", version="1.0.0")

DB_CONFIG = {
    "host": "localhost",
    "port": 5432,
    "dbname": "ir_db",
    "user": "postgres",
    "password": "root"
}

def get_connection():
    return psycopg2.connect(**DB_CONFIG)

class DocumentOut(BaseModel):
    doc_id: str
    raw_text: str
    title: str = ""

class BatchGetRequest(BaseModel):
    dataset_id: str
    doc_ids: List[str]

@app.get("/health")
def health():
    return {"status": "ok", "service": "document_store"}

@app.post("/get/batch", response_model=List[DocumentOut])
def get_batch(req: BatchGetRequest):
    if not req.doc_ids:
        return []
    
    conn = get_connection()
    cur = conn.cursor()
    
    cur.execute(
        "SELECT doc_id, raw_text FROM documents WHERE doc_id = ANY(%s)",
        (req.doc_ids,)
    )
    rows = {row[0]: row for row in cur.fetchall()}
    cur.close()
    conn.close()
    
    results = []
    for doc_id in req.doc_ids:
        if doc_id in rows:
            results.append(DocumentOut(
                doc_id=rows[doc_id][0],
                raw_text=rows[doc_id][1] or "",
                title=""
            ))
        else:
            results.append(DocumentOut(
                doc_id=doc_id,
                raw_text="[Document not found]",
                title=""
            ))
    
    return results

@app.get("/get/{dataset_id}/{doc_id}", response_model=DocumentOut)
def get_single(dataset_id: str, doc_id: str):
    conn = get_connection()
    cur = conn.cursor()
    cur.execute(
        "SELECT doc_id, raw_text, title FROM documents WHERE doc_id = %s",
        (doc_id,)
    )
    row = cur.fetchone()
    cur.close()
    conn.close()
    
    if not row:
        raise HTTPException(404, f"Document '{doc_id}' not found")
    
    return DocumentOut(doc_id=row[0], raw_text=row[1] or "", title=row[2] or "")

@app.get("/stats/{dataset_id}")
def stats(dataset_id: str):
    conn = get_connection()
    cur = conn.cursor()
    cur.execute("SELECT COUNT(*) FROM documents")
    count = cur.fetchone()[0]
    cur.close()
    conn.close()
    return {"dataset_id": dataset_id, "document_count": count}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8009)