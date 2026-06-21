"""
build_sbert_index.py (v2 — FIXED: keyset pagination)
=====================================================================
بناء SBERT embeddings + FAISS index لكامل MS MARCO (8.8M وثيقة) محلياً.

★ FIX v2: المشكلة بالنسخة الأولى كانت استخدام:
    SELECT ... ORDER BY doc_id OFFSET %s
  هاد بيتباطأ تصاعدياً لأن PostgreSQL لازم "يعد ويتجاوز" offset صفوف
  من البداية كل استعلام (كان طلع 60+ ساعة بدل 7 ساعات).

  الحل: keyset pagination — نتذكر آخر doc_id قرأناه ونكمل منه مباشرة:
    SELECT ... WHERE doc_id > 'last_seen_id' ORDER BY doc_id LIMIT N
  هاد يستخدم الـ index على doc_id (PRIMARY KEY) مباشرة بدون عدّ أي شي،
  سرعته ثابتة بغض النظر عن مكانك بالجدول.

  ⚠️ ملاحظة مهمة: doc_id بـ MS MARCO هو نص يمثل رقم (e.g. "0", "1", "10",
  "100", "11", ...) — الترتيب الأبجدي (lexicographic) للنصوص مختلف عن
  الترتيب الرقمي. هذا غير مهم لصحة البيانات (كل وثيقة هتتـ encode مرة
  وحدة بالنهاية بغض النظر عن الترتيب)، فقط الترتيب اللي رح تظهر فيه
  مختلف عن "0,1,2,3..." — هذا لا يؤثر على التقييم أو النتائج إطلاقاً.

تصميم آمن للذاكرة (8GB RAM):
    - server-side cursor + keyset pagination (سريع وثابت السرعة)
    - encode بـ batches صغيرة
    - حفظ تدريجي (checkpoint) على القرص كل CHECKPOINT_EVERY batch
    - عند إعادة التشغيل: يكمل من آخر doc_id محفوظ تلقائياً

الاستخدام:
    python build_sbert_index.py
"""

import json
import time
import sys
from pathlib import Path

import numpy as np
import psycopg2

# ──────────────────────────────────────────────
# إعدادات
# ──────────────────────────────────────────────
DB_CONFIG = {
    "host": "localhost",
    "port": 5432,
    "dbname": "ir2_db",        # ← تعديل 1
    "user": "postgres",
    "password": "root",
}

DATASET_ID = "touche"          # ← تعديل 2

SBERT_MODEL_NAME = "all-MiniLM-L6-v2"
EMBEDDING_DIM = 384

FETCH_BATCH = 2000          # كم وثيقة تُقرأ من DB دفعة وحدة
ENCODE_BATCH = 64           # batch size لـ SBERT encode
CHECKPOINT_EVERY = 20       # كل 20 fetch batch = كل 40K وثيقة تقريباً

OUT_DIR = Path("indexes/embeddings") / DATASET_ID / "sbert"
OUT_DIR.mkdir(parents=True, exist_ok=True)

EMB_CHECKPOINT = OUT_DIR / "embeddings_checkpoint.npy"
IDS_CHECKPOINT = OUT_DIR / "doc_ids_checkpoint.json"
PROGRESS_FILE = OUT_DIR / "progress.json"   # last_doc_id بدل offset رقمي
FAISS_OUT = OUT_DIR / "faiss.index"
DOC_IDS_PKL = OUT_DIR / "doc_ids.pkl"


def log(msg):
    ts = time.strftime("%H:%M:%S")
    print(f"[{ts}] {msg}", flush=True)


def load_progress():
    if PROGRESS_FILE.exists():
        with open(PROGRESS_FILE) as f:
            p = json.load(f)
        return p.get("last_doc_id")
    return None


def save_progress(last_doc_id, total_done):
    with open(PROGRESS_FILE, "w") as f:
        json.dump({
            "last_doc_id": last_doc_id,
            "total_done": total_done,
            "updated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        }, f)


def load_existing_embeddings():
    if EMB_CHECKPOINT.exists() and IDS_CHECKPOINT.exists():
        embs = np.load(EMB_CHECKPOINT)
        with open(IDS_CHECKPOINT) as f:
            ids = json.load(f)
        return embs, ids
    return np.zeros((0, EMBEDDING_DIM), dtype=np.float32), []


def get_total_count(conn):
    with conn.cursor() as cur:
        cur.execute("SELECT COUNT(*) FROM documents")
        return cur.fetchone()[0]


def fetch_next_batch(conn, last_doc_id, batch_size):
    """
    Keyset pagination: يجيب batch_size صف بعد last_doc_id مباشرة.
    سريع وثابت بغض النظر عن مكانك بالجدول (يستخدم index على doc_id).
    """
    with conn.cursor() as cur:
        if last_doc_id is None:
            cur.execute(
                "SELECT doc_id, raw_text FROM documents "
                "ORDER BY doc_id LIMIT %s",
                (batch_size,),
            )
        else:
            cur.execute(
                "SELECT doc_id, raw_text FROM documents "
                "WHERE doc_id > %s ORDER BY doc_id LIMIT %s",
                (last_doc_id, batch_size),
            )
        return cur.fetchall()


def main():
    log("=" * 60)
    log("SBERT Full-Corpus Encoding FIXED")
    log("=" * 60)

    log(f"Loading SBERT model '{SBERT_MODEL_NAME}' (from local cache)...")
    t0 = time.time()
    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer(SBERT_MODEL_NAME)
    model.max_seq_length = 256   
    log(f"Model loaded in {time.time()-t0:.1f}s")

    conn = psycopg2.connect(**DB_CONFIG)
    total_docs = get_total_count(conn)
    log(f"Total documents in DB: {total_docs:,}")

    last_doc_id = load_progress()
    all_embeddings, all_doc_ids = load_existing_embeddings()

    if last_doc_id is not None:
        log(f"⏩ Resuming from checkpoint: last_doc_id={last_doc_id}, "
            f"already encoded={len(all_doc_ids):,}")
    else:
        log("Starting fresh (no previous checkpoint found).")

    fetch_count_since_checkpoint = 0
    start_time = time.time()
    docs_processed_this_run = 0

    def flush_checkpoint(current_last_id):
        np.save(EMB_CHECKPOINT, all_embeddings)
        with open(IDS_CHECKPOINT, "w") as f:
            json.dump(all_doc_ids, f)
        save_progress(current_last_id, len(all_doc_ids))

    try:
        while True:
            t0 = time.time()
            rows = fetch_next_batch(conn, last_doc_id, FETCH_BATCH)
            t_fetch = time.time() - t0
            if not rows:
                break  # خلصت كل الوثائق

            batch_ids = [r[0] for r in rows]
            batch_texts = [(r[1] or "")[:2000] for r in rows]

            # ── Encode هالدفعة ──
            t0 = time.time()
            vectors = model.encode(
                batch_texts,
                batch_size=ENCODE_BATCH,
                convert_to_numpy=True,
                normalize_embeddings=True,
                show_progress_bar=False,
            )
            t_encode = time.time() - t0
            log(f"⏱️ fetch={t_fetch:.2f}s | encode={t_encode:.2f}s | avg_text_len={sum(len(t) for t in batch_texts)/len(batch_texts):.0f} chars")
            all_embeddings = np.vstack([all_embeddings, vectors.astype(np.float32)])
            all_doc_ids.extend(batch_ids)
            docs_processed_this_run += len(batch_ids)

            last_doc_id = batch_ids[-1]  # ★ نتذكر آخر doc_id لهالـ batch
            fetch_count_since_checkpoint += 1

            # ── طباعة تقدّم ──
            elapsed = time.time() - start_time
            rate = docs_processed_this_run / elapsed if elapsed > 0 else 0
            done_total = len(all_doc_ids)
            remaining = total_docs - done_total
            eta_hours = (remaining / rate / 3600) if rate > 0 else float("inf")

            log(f"Progress: {done_total:,}/{total_docs:,} "
                f"({100*done_total/total_docs:.1f}%) | "
                f"Rate: {rate:.1f} docs/s | "
                f"ETA: {eta_hours:.2f}h")

            # ── Checkpoint دوري ──
            if fetch_count_since_checkpoint >= CHECKPOINT_EVERY:
                log(f"💾 Saving checkpoint at {done_total:,} docs (last_doc_id={last_doc_id})...")
                flush_checkpoint(last_doc_id)
                fetch_count_since_checkpoint = 0

        # ── حفظ نهائي ──
        log(f"💾 Final checkpoint save: {len(all_doc_ids):,} docs total")
        flush_checkpoint(last_doc_id)

    except KeyboardInterrupt:
        log("⚠️ Interrupted by user. Saving progress before exit...")
        flush_checkpoint(last_doc_id)
        log(f"✅ Safe to resume later. Progress saved at {len(all_doc_ids):,} docs "
            f"(last_doc_id={last_doc_id}). Just re-run the same command.")
        sys.exit(0)
    except Exception as e:
        log(f"❌ Error occurred: {e}")
        log("Saving progress before exit...")
        flush_checkpoint(last_doc_id)
        raise
    finally:
        conn.close()

    # ══════════════════════════════════════════════
    # بناء الـ FAISS index النهائي
    # ══════════════════════════════════════════════
    log("=" * 60)
    log(f"All {len(all_doc_ids):,} documents encoded. Building FAISS index...")
    import faiss
    import pickle

    dim = all_embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(all_embeddings)
    faiss.write_index(index, str(FAISS_OUT))

    with open(DOC_IDS_PKL, "wb") as f:
        pickle.dump(all_doc_ids, f)

    log(f"✅ FAISS index saved: {index.ntotal:,} vectors, dim={dim}")
    log(f"✅ Files written to: {OUT_DIR}")
    log("=" * 60)
    log("DONE. You can now delete embeddings_checkpoint.npy and "
        "doc_ids_checkpoint.json to save disk space (faiss.index + doc_ids.pkl "
        "are the final files used by embedding/main.py).")


if __name__ == "__main__":
    main()