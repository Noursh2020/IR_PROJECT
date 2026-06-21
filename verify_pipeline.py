"""
verify_pipeline.py
يفحص كل مراحل المشروع لـ dataset 'touche' دفعة واحدة ويطبع ✅/❌ لكل بند.
شغّليه: python verify_pipeline.py
"""
import json
import httpx
from pathlib import Path

DATASET = "touche"
GATEWAY = "http://localhost:8000"

def check(label, condition, detail=""):
    mark = "✅" if condition else "❌"
    print(f"{mark} {label}" + (f"  → {detail}" if detail else ""))
    return condition

print("=" * 60)
print("1) حالة الخدمات (Health Check)")
print("=" * 60)
try:
    h = httpx.get(f"{GATEWAY}/health", timeout=10).json()
    for svc, status in h["services"].items():
        check(f"خدمة {svc}", status == "ok", status)
except Exception as e:
    check("الاتصال بالـ Gateway", False, str(e))

print("\n" + "=" * 60)
print("2) قاعدة البيانات (ir2_db)")
print("=" * 60)
try:
    import psycopg2
    conn = psycopg2.connect(host="localhost", port=5432, dbname="ir2_db",
                             user="postgres", password="root")
    cur = conn.cursor()
    cur.execute("SELECT COUNT(*) FROM documents")
    n_docs = cur.fetchone()[0]
    check("عدد الوثائق بجدول documents", n_docs > 350_000, f"{n_docs:,} وثيقة")

    cur.execute("SELECT COUNT(*) FROM terms")
    n_terms = cur.fetchone()[0]
    check("عدد المصطلحات بجدول terms", n_terms > 0, f"{n_terms:,} مصطلح")

    cur.execute("SELECT COUNT(*) FROM postings")
    n_postings = cur.fetchone()[0]
    check("عدد postings", n_postings > 0, f"{n_postings:,} posting")

    cur.execute("SELECT last_index FROM indexing_progress WHERE dataset_id = %s", (DATASET,))
    row = cur.fetchone()
    check("تقدّم الفهرسة المحفوظ", row is not None, f"last_index={row[0] if row else None}")
    cur.close(); conn.close()
except Exception as e:
    check("الاتصال بـ ir2_db", False, str(e))

print("\n" + "=" * 60)
print("3) ملفات queries / qrels / meta")
print("=" * 60)
base = Path(f"data/{DATASET}")
for fname, expected_min in [("queries.json", 40), ("qrels.json", 1), ("meta.json", None)]:
    fpath = base / fname
    if not fpath.exists():
        check(f"وجود {fname}", False, "الملف غير موجود")
        continue
    with open(fpath, encoding="utf-8") as f:
        data = json.load(f)
    if fname == "meta.json":
        check("meta.json محتواه", True, json.dumps(data, ensure_ascii=False))
        check("embedding_model مبني؟", data.get("embedding_model") is not None,
              f"embedding_model = {data.get('embedding_model')}")
    else:
        n = len(data)
        check(f"عدد العناصر بـ {fname}", n >= expected_min, f"{n} عنصر")

print("\n" + "=" * 60)
print("4) ملفات FAISS (Embeddings)")
print("=" * 60)
sbert_path = Path(f"indexes/embeddings/{DATASET}/sbert/faiss.index")
check("FAISS index لـ SBERT موجود", sbert_path.exists(),
      str(sbert_path) if sbert_path.exists() else "غير موجود — لم يُبنى بعد")
w2v_path = Path(f"indexes/embeddings/{DATASET}/word2vec/faiss.index")
check("FAISS index لـ Word2Vec موجود", w2v_path.exists(),
      str(w2v_path) if w2v_path.exists() else "غير موجود — لم يُبنى بعد")

print("\n" + "=" * 60)
print("5) ملف الإحصائيات (avg_doc_length) — مهم جداً لـ BM25 الصحيح")
print("=" * 60)
stats_path = base / "stats.json"
check("stats.json موجود", stats_path.exists(),
      "غير موجود — BM25 سيستخدم قيمة افتراضية خاطئة (33.29 من MS MARCO)!" if not stats_path.exists() else "")
if stats_path.exists():
    with open(stats_path) as f:
        print("   →", json.load(f))

print("\n" + "=" * 60)
print("6) اختبار الاسترجاع الفعلي لكل نموذج")
print("=" * 60)
test_query = "should social media be banned"
for model in ["bm25", "tfidf", "sbert", "word2vec", "hybrid_serial", "hybrid_parallel"]:
    try:
        r = httpx.post(f"{GATEWAY}/search", json={
            "query": test_query, "dataset_id": DATASET, "model": model,
            "top_k": 5, "use_query_refinement": False,
        }, timeout=60)
        n_results = len(r.json().get("results", []))
        check(f"نموذج {model}", r.status_code == 200 and n_results > 0,
              f"status={r.status_code}, نتائج={n_results}")
    except Exception as e:
        check(f"نموذج {model}", False, str(e)[:150])

print("\n" + "=" * 60)
print("انتهى الفحص")
print("=" * 60)