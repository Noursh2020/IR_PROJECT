"""
train_word2vec_safe.py - المطور والمضمون 100%
يضمن عدم تعليق الشاشة، ويوفر حماية كاملة للذاكرة أثناء بناء FAISS
"""
import gensim
from gensim.models import Word2Vec
import psycopg2
import pickle
import numpy as np
from pathlib import Path
import faiss
import logging

# تفعيل الطباعة التلقائية لـ Gensim لتري كم وثيقة يتم معالجتها حالياً!
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

DATASET_ID = "touche"
OUT_DIR = Path(f"indexes/embeddings/{DATASET_ID}/word2vec")
OUT_DIR.mkdir(parents=True, exist_ok=True)

def stream_texts(batch_size=5000):
    """يقرأ النصوص من DB بدفعات، آمن تماماً للذاكرة."""
    conn = psycopg2.connect(host="localhost", port=5432, dbname="ir2_db",
                           user="postgres", password="root")
    last_id = None
    while True:
        cur = conn.cursor()
        if last_id is None:
            cur.execute("SELECT doc_id, raw_text FROM documents "
                        "ORDER BY doc_id LIMIT %s", (batch_size,))
        else:
            cur.execute("SELECT doc_id, raw_text FROM documents "
                        "WHERE doc_id > %s ORDER BY doc_id LIMIT %s",
                        (last_id, batch_size))
        rows = cur.fetchall()
        cur.close()
        if not rows:
            break
        last_id = rows[-1][0]
        for doc_id, text in rows:
            yield doc_id, (text or "")
    conn.close()


class CorpusIterator:
    """يمر على الوثائق ويطبع تقدم القراءة حتى لا تظني أن السكريبت معلق"""
    def __iter__(self):
        count = 0
        for _, text in stream_texts(batch_size=5000):
            count += 1
            if count % 50000 == 0:
                print(f"🔄 Gensim يقرأ الآن الوثيقة رقم: {count:,}...")
            yield text.lower().split()


print("=" * 60)
print("🚀 بدء تدريب Word2Vec مع نظام التتبع الذكي...")
print("=" * 60)

model = Word2Vec(
    sentences=CorpusIterator(),
    vector_size=300,
    window=5,
    min_count=2,
    workers=4,
    epochs=5,  
)
print(f"✅ انتهى التدريب بنجاح! حجم القاموس: {len(model.wv)}")

model.save(str(OUT_DIR / "word2vec.model"))
print(f"✅ تم حفظ الموديل في: {OUT_DIR / 'word2vec.model'}")

# ──────────────────────────────────────────────
# 🛡️ بناء FAISS بـ Streaming حقيقي لحماية الـ RAM (Batch-by-Batch)
# ──────────────────────────────────────────────
print("\n⚡ بدء بناء كشاف FAISS بنظام الحماية القصوى للذاكرة...")

index = faiss.IndexFlatIP(300) # المتجهات dim=300
doc_ids = []

batch_vectors = []
batch_size = 5000

for doc_id, text in stream_texts(batch_size=batch_size):
    tokens = text.lower().split()
    word_vecs = [model.wv[w] for w in tokens if w in model.wv]
    
    if word_vecs:
        vec = np.mean(word_vecs, axis=0).astype(np.float32)
    else:
        vec = np.zeros(300, dtype=np.float32)
        
    norm = np.linalg.norm(vec)
    if norm > 0:
        vec = vec / norm
        
    doc_ids.append(doc_id)
    batch_vectors.append(vec)
    
    # عندما تمتلئ الدفعة، نقوم بحقنها داخل FAISS ومسحها من الذاكرة فوراً!
    if len(batch_vectors) == batch_size:
        matrix = np.array(batch_vectors, dtype=np.float32)
        index.add(matrix)
        batch_vectors = [] # تفريغ الذاكرة فوراً 🧹
        if len(doc_ids) % 50000 == 0:
            print(f"📥 تم حقن {len(doc_ids):,} متجه داخل كشاف FAISS وحماية الـ RAM...")

# إضافة المتبقي من الوثائق
if batch_vectors:
    matrix = np.array(batch_vectors, dtype=np.float32)
    index.add(matrix)

# حفظ الكشاف والـ IDs
faiss.write_index(index, str(OUT_DIR / "faiss.index"))
with open(OUT_DIR / "doc_ids.pkl", "wb") as f:
    pickle.dump(doc_ids, f)

print(f"🎉 تم بنجاح خارق! كشاف FAISS جاهز تماماً ويحتوي على {index.ntotal:,} متجه!")