# fix_hybrid_serial_eval.py
import asyncio, json, httpx

GATEWAY = "http://localhost:8000"
DATASET = "touche"

async def run():
    queries = httpx.get(f"{GATEWAY}/datasets/queries/{DATASET}?limit=100").json()["queries"]
    qrels = httpx.get("http://localhost:8007/qrels/touche?limit=10000").json()["qrels"]
    sem = asyncio.Semaphore(5)

    async def search_one(client, qid, qtext, refine):
        async with sem:
            r = await client.post(f"{GATEWAY}/search", json={
                "query": qtext, "dataset_id": DATASET, "model": "hybrid_serial",
                "top_k": 10, "use_query_refinement": refine,
            }, timeout=90)
            data = r.json()
            retrieved = [{"doc_id": x["doc_id"], "score": x["score"], "rank": i+1}
                         for i, x in enumerate(data.get("results", []))]
            return qid, retrieved

    with open("data/touche/evaluation_results.json", "r", encoding="utf-8") as f:
        results_summary = json.load(f)

    async with httpx.AsyncClient() as client:
        for refine in [False, True]:
            tasks = [search_one(client, qid, qtext, refine) for qid, qtext in queries.items()]
            outcomes = await asyncio.gather(*tasks)
            evaluations = []
            for qid, retrieved in outcomes:
                q_qrels = [q for q in qrels if q["query_id"] == qid]
                evaluations.append({"query_id": qid, "retrieved_docs": retrieved, "qrels": q_qrels, "k": 10})
            ev = httpx.post("http://localhost:8004/evaluate/batch", json={"evaluations": evaluations, "k": 10}, timeout=60).json()
            key = f"hybrid_serial_{'after' if refine else 'before'}_refinement"
            results_summary[key] = {"MAP": ev["map_score"], "Recall": ev["mean_recall"],
                                     "P@10": ev["mean_precision_at_k"], "nDCG": ev["mean_ndcg"], "n_queries": len(queries)}
            print(key, results_summary[key])

    with open("data/touche/evaluation_results.json", "w", encoding="utf-8") as f:
        json.dump(results_summary, f, indent=2, ensure_ascii=False)
    print("✅ تم تحديث hybrid_serial فقط.")

asyncio.run(run())