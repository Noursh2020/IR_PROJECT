"""
test_preprocess_speed.py
يقيس سرعة Preprocessing Service وحدها فقط — بدون أي تأثير على
الفهرسة الجارية بالخلفية (قراءة فقط من القاعدة، نداء API منفصل).
"""
import time
import httpx
import psycopg2

conn = psycopg2.connect(host="localhost", port=5432, dbname="ir2_db",
                         user="postgres", password="root")
cur = conn.cursor()
cur.execute("SELECT raw_text FROM documents ORDER BY doc_id LIMIT 250")
texts = [r[0] for r in cur.fetchall()]
cur.close(); conn.close()

print(f"عدد النصوص: {len(texts)} | متوسط الطول: {sum(len(t) for t in texts)/len(texts):.0f} حرف")

t0 = time.time()
r = httpx.post("http://localhost:8001/preprocess/batch",
                json={"texts": texts, "use_lemmatization": True, "remove_stopwords": True},
                timeout=120)
elapsed = time.time() - t0
print(f"الزمن: {elapsed:.2f} ثانية | المعدّل: {len(texts)/elapsed:.1f} نص/ثانية")