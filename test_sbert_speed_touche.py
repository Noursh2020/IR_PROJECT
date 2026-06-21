# test_sbert_speed_touche.py  ← ملف جديد، بجذر المشروع
import time
import psycopg2
from sentence_transformers import SentenceTransformer

conn = psycopg2.connect(host="localhost", port=5432, dbname="ir2_db",
                         user="postgres", password="root")
cur = conn.cursor()
cur.execute("SELECT raw_text FROM documents ORDER BY doc_id LIMIT 1000")
texts = [(r[0] or "")[:2000] for r in cur.fetchall()]
cur.close(); conn.close()

print(f"عدد النصوص: {len(texts)} | متوسط الطول: {sum(len(t) for t in texts)/len(texts):.0f} حرف")

model = SentenceTransformer("all-MiniLM-L6-v2")
model.max_seq_length = 256   # كافٍ لنصوص حجاجية أطول من passages

t0 = time.time()
vectors = model.encode(texts, batch_size=32, show_progress_bar=False,
                        convert_to_numpy=True, normalize_embeddings=True)
elapsed = time.time() - t0
rate = len(texts) / elapsed
print(f"الزمن: {elapsed:.1f}s | المعدّل: {rate:.1f} نص/ثانية")
print(f"الوقت المتوقع لكامل 373,514 وثيقة: {373514/rate/3600:.2f} ساعة")