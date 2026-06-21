import httpx

# 1. تحميل الـ FAISS index لذاكرة خدمة embedding
r = httpx.post("http://localhost:8006/load/touche?model=sbert")
print("Load:", r.json())

# 2. اختبار sbert مباشرة
r = httpx.post("http://localhost:8003/retrieve", json={
    "query": "should social media be banned",
    "dataset_id": "touche", "model": "sbert", "top_k": 5,
}, timeout=60)
print("sbert:", r.status_code, len(r.json().get("results", [])))

# 3. اختبار hybrid_serial (BM25 → SBERT rerank)
r = httpx.post("http://localhost:8003/retrieve", json={
    "query": "should social media be banned",
    "dataset_id": "touche", "model": "hybrid_serial", "top_k": 5,
}, timeout=60)
print("hybrid_serial:", r.status_code, len(r.json().get("results", [])))

# 4. اختبار hybrid_parallel (BM25 + SBERT → fusion)
r = httpx.post("http://localhost:8003/retrieve", json={
    "query": "should social media be banned",
    "dataset_id": "touche", "model": "hybrid_parallel", "top_k": 5,
}, timeout=60)
print("hybrid_parallel:", r.status_code, len(r.json().get("results", [])))