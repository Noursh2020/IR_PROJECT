import httpx
for model in ["bm25", "tfidf"]:
    r = httpx.post("http://localhost:8003/retrieve", json={
        "query": "should social media be banned",
        "dataset_id": "touche", "model": model, "top_k": 5,
    }, timeout=30)
    print(model, r.status_code, len(r.json().get("results", [])))
    for item in r.json().get("results", [])[:2]:
        print("  ", item["doc_id"], round(item["score"], 4), item["snippet"][:100])