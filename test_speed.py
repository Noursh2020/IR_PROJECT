import time
from sentence_transformers import SentenceTransformer

print("Loading model...")
t0 = time.time()
model = SentenceTransformer("all-MiniLM-L6-v2")
print(f"Model loaded in {time.time()-t0:.1f}s")

# جيبي 2000 جملة حقيقية (مثلاً من جدول documents عندك أو placeholder نصوص)
texts = ["This is a sample passage about machine learning and information retrieval systems."] * 2000

t0 = time.time()
vectors = model.encode(texts, batch_size=32, show_progress_bar=True)
elapsed = time.time() - t0
print(f"\nEncoded {len(texts)} docs in {elapsed:.1f}s")
print(f"Rate: {len(texts)/elapsed:.1f} docs/sec")
print(f"Estimated time for 8.8M docs: {8_800_000/(len(texts)/elapsed)/3600:.2f} hours")