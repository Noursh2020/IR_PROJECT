"""
Document Store Service — خدمة تخزين الوثائق الأصلية (Raw Documents)
=====================================================================
ملاحظة الأستاذ (السبب وراء هذه الخدمة):
----------------------------------------------------------------------
    "الـ raw text بدو يكون بداتا بيز بلحظة الكويري. باقي المراحل
     (preprocessing, indexing) أنتو حرّين بـ files أو DB. لكن وقت تجي
     كويري اليوزر وبدي اقرا شي بخص الدوكيومنتات، بقرأه من DB حسب الـ ID."

    الفكرة الجوهرية:
        1. أثناء الفهرسة: نُخزِّن الوثائق الأصلية (raw) في قاعدة بيانات
           مستقلة (SQLite) — كل وثيقة لها doc_id فريد كـ Primary Key.

        2. أثناء البحث: محركات البحث (BM25/TF-IDF/SBERT) تعمل على
           الـ indexes الخفيفة فقط (inverted index, FAISS vectors)
           وتُعيد قائمة doc_ids + scores — بدون أي نص.

        3. في آخر خطوة فقط (Result Display):
           Retrieval Service يستدعي Document Store Service بقائمة
           top-10 doc_ids → يحصل على الوثائق الأصلية (raw) لعرضها
           للمستخدم.

    لماذا هذا أسرع؟ (Lecture 2 — فلسفة الفهرسة)
        - البحث يعمل على بنى بيانات صغيرة (inverted index + vectors)
          بدون الحاجة لتحميل ملايين النصوص الكاملة في الذاكرة.
        - فقط 10 وثائق (النتيجة النهائية) تُقرأ كاملة من DB — استرجاع
          O(1) بالـ doc_id (Primary Key Index في SQLite).
        - هذا يطابق فلسفة "Indexes تُسرّع الاسترجاع وتقلل الـ I/O".

    لماذا SQLite وليس MongoDB؟
        - حجم البيانات هنا (200K وثيقة × ~2KB) ≈ 400MB — مناسب جداً
          لـ SQLite بدون خادم منفصل.
        - SQLite يدعم فهرسة doc_id كـ Primary Key → بحث O(log n) أو
          O(1) مع B-Tree index.
        - يمكن استبدالها بـ MongoDB بسهولة (الواجهة/الـ API لا تتغير)
          إذا أصبح حجم البيانات أكبر — هذا مذكور كخيار مستقبلي.

SOA Role:
    خدمة مستقلة على Port 8009.
    - Dataset Loader يكتب الوثائق الأصلية هنا أثناء التحميل.
    - Retrieval Service يقرأ منها فقط في آخر خطوة (top-K results).
    - Indexing/Embedding لا يحتاجان الوصول لها أبداً (يعملان على
      processed_text المُخزَّن في ملفاتهم الخاصة).
"""

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Dict, Optional
import sqlite3
from pathlib import Path
from contextlib import contextmanager

app = FastAPI(
    title="Document Store Service",
    description="قاعدة بيانات SQLite للوثائق الأصلية (raw) — تُستعلَم فقط عند عرض النتائج النهائية",
    version="1.0.0",
)

DB_DIR = Path("data/document_store")
DB_DIR.mkdir(parents=True, exist_ok=True)


# ──────────────────────────────────────────────
# اتصال قاعدة البيانات
# ──────────────────────────────────────────────

def get_db_path(dataset_id: str) -> Path:
    """كل Dataset له ملف SQLite منفصل: data/document_store/{dataset_id}.db"""
    return DB_DIR / f"{dataset_id}.db"


@contextmanager
def get_connection(dataset_id: str):
    """
    اتصال SQLite مع تفعيل WAL mode للقراءة/الكتابة المتزامنة
    وتفعيل index على doc_id (Primary Key) للاسترجاع السريع O(1)/O(log n).
    """
    db_path = get_db_path(dataset_id)
    conn = sqlite3.connect(str(db_path))
    conn.execute("PRAGMA journal_mode=WAL")  # سماح بقراءة متزامنة أثناء الكتابة
    try:
        yield conn
        conn.commit()
    finally:
        conn.close()


def init_db(dataset_id: str):
    """
    إنشاء جدول الوثائق الأصلية.
    doc_id هو PRIMARY KEY → SQLite يبني B-Tree index تلقائياً عليه
    → استرجاع سريع جداً عند البحث بـ WHERE doc_id IN (...).
    """
    with get_connection(dataset_id) as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS documents (
                doc_id TEXT PRIMARY KEY,
                raw_text TEXT NOT NULL,
                title TEXT DEFAULT '',
                metadata TEXT DEFAULT '{}'
            )
        """)


# ──────────────────────────────────────────────
# Schemas
# ──────────────────────────────────────────────

class DocumentIn(BaseModel):
    doc_id: str
    raw_text: str
    title: Optional[str] = ""
    metadata: Optional[str] = "{}"   # JSON string


class BulkInsertRequest(BaseModel):
    dataset_id: str
    documents: List[DocumentIn]


class DocumentOut(BaseModel):
    doc_id: str
    raw_text: str
    title: str


class BatchGetRequest(BaseModel):
    """
    طلب استرجاع دفعة من الوثائق الأصلية بالـ doc_ids.
    هذا هو الـ endpoint الذي يُستدعى في "آخر خطوة" فقط
    (عند عرض top-K نتائج البحث للمستخدم).
    """
    dataset_id: str
    doc_ids: List[str]


# ──────────────────────────────────────────────
# API Endpoints
# ──────────────────────────────────────────────

@app.get("/health")
def health():
    return {"status": "ok", "service": "document_store"}


@app.post("/insert/bulk")
def insert_bulk(req: BulkInsertRequest):
    """
    إدراج دفعة من الوثائق الأصلية في قاعدة البيانات.
    يُستدعى من Dataset Loader أثناء التحميل — مرة واحدة فقط.

    هذا منفصل تماماً عن خطوات Preprocessing/Indexing:
    - Indexing Service يعمل على النصوص المُعالَجة (tokens) لبناء
      Inverted Index — لا يحتاج النص الكامل بعد بناء الفهرس.
    - Document Store يحتفظ بالنص الأصلي (raw) كما هو، دون أي معالجة،
      لعرضه للمستخدم لاحقاً.
    """
    init_db(req.dataset_id)

    with get_connection(req.dataset_id) as conn:
        conn.executemany(
            "INSERT OR REPLACE INTO documents (doc_id, raw_text, title, metadata) VALUES (?, ?, ?, ?)",
            [(d.doc_id, d.raw_text, d.title or "", d.metadata or "{}") for d in req.documents],
        )

    return {"inserted": len(req.documents), "dataset_id": req.dataset_id}


@app.post("/get/batch", response_model=List[DocumentOut])
def get_batch(req: BatchGetRequest):
    """
    ═══════════════════════════════════════════════════
    الـ Endpoint الأهم — يُستدعى في "آخر خطوة" فقط
    ═══════════════════════════════════════════════════

    يُستخدَم من Retrieval Service بعد حصوله على top-K doc_ids
    من BM25/TF-IDF/SBERT (التي تعمل على indexes خفيفة بدون نص).

    SQL: SELECT ... WHERE doc_id IN (...)
    بفضل PRIMARY KEY على doc_id، هذا الاستعلام يستخدم B-Tree index
    → استرجاع سريع جداً حتى لمجموعات بيانات كبيرة (200K+ وثيقة).

    الترتيب: يُحافَظ على نفس ترتيب doc_ids المُمرَّر (ترتيب نتائج البحث).
    """
    db_path = get_db_path(req.dataset_id)
    if not db_path.exists():
        raise HTTPException(404, f"لا توجد قاعدة بيانات للـ dataset '{req.dataset_id}'. حمّل الـ Dataset أولاً.")

    if not req.doc_ids:
        return []

    with get_connection(req.dataset_id) as conn:
        placeholders = ",".join("?" * len(req.doc_ids))
        cursor = conn.execute(
            f"SELECT doc_id, raw_text, title FROM documents WHERE doc_id IN ({placeholders})",
            req.doc_ids,
        )
        rows = {row[0]: {"doc_id": row[0], "raw_text": row[1], "title": row[2]} for row in cursor.fetchall()}

    # الحفاظ على ترتيب doc_ids الأصلي (ترتيب نتائج البحث)
    results = []
    for doc_id in req.doc_ids:
        if doc_id in rows:
            results.append(DocumentOut(**rows[doc_id]))
        else:
            # وثيقة غير موجودة (نادر) — نُرجع placeholder بدل كسر الـ pipeline
            results.append(DocumentOut(doc_id=doc_id, raw_text="[Document not found in store]", title=""))

    return results


@app.get("/get/{dataset_id}/{doc_id}", response_model=DocumentOut)
def get_single(dataset_id: str, doc_id: str):
    """استرجاع وثيقة واحدة بالـ ID — للاختبار اليدوي أو RAG."""
    db_path = get_db_path(dataset_id)
    if not db_path.exists():
        raise HTTPException(404, f"لا توجد قاعدة بيانات للـ dataset '{dataset_id}'")

    with get_connection(dataset_id) as conn:
        cursor = conn.execute(
            "SELECT doc_id, raw_text, title FROM documents WHERE doc_id = ?",
            (doc_id,),
        )
        row = cursor.fetchone()

    if not row:
        raise HTTPException(404, f"الوثيقة '{doc_id}' غير موجودة في '{dataset_id}'")

    return DocumentOut(doc_id=row[0], raw_text=row[1], title=row[2])


@app.get("/stats/{dataset_id}")
def stats(dataset_id: str):
    """عدد الوثائق المخزَّنة لكل Dataset."""
    db_path = get_db_path(dataset_id)
    if not db_path.exists():
        return {"dataset_id": dataset_id, "exists": False, "document_count": 0}

    with get_connection(dataset_id) as conn:
        count = conn.execute("SELECT COUNT(*) FROM documents").fetchone()[0]

    return {
        "dataset_id": dataset_id,
        "exists": True,
        "document_count": count,
        "db_path": str(db_path),
        "db_size_mb": round(db_path.stat().st_size / (1024 * 1024), 2),
    }


@app.delete("/reset/{dataset_id}")
def reset(dataset_id: str):
    """حذف قاعدة بيانات Dataset كاملة (لإعادة التحميل)."""
    db_path = get_db_path(dataset_id)
    if db_path.exists():
        db_path.unlink()
        # حذف ملفات WAL المرافقة إن وجدت
        for suffix in ["-wal", "-shm"]:
            extra = Path(str(db_path) + suffix)
            if extra.exists():
                extra.unlink()
    return {"reset": True, "dataset_id": dataset_id}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8009)