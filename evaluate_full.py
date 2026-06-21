import asyncio, json, httpx

GATEWAY = "http://localhost:8000"
DATASET = "touche"
MODELS = ["tfidf", "bm25", "sbert", "word2vec", "hybrid_serial", "hybrid_parallel"]

async def run_eval():
    queries = httpx.get(f"{GATEWAY}/datasets/queries/{DATASET}?limit=100").json()["queries"]
    qrels = httpx.get("http://localhost:8007/qrels/touche?limit=10000").json()["qrels"]
    print(f"عدد الاستعلامات المستخدمة بالتقييم: {len(queries)}")   # يجب = 49

    sem = asyncio.Semaphore(5)

    async def search_one(client, qid, qtext, model, refine):
        async with sem:
            try:
                r = await client.post(f"{GATEWAY}/search", json={
                    "query": qtext, "dataset_id": DATASET, "model": model,
                    "top_k": 10, "use_query_refinement": refine,
                }, timeout=90)
                data = r.json()
                retrieved = [{"doc_id": x["doc_id"], "score": x["score"], "rank": i+1}
                             for i, x in enumerate(data.get("results", []))]
                return qid, retrieved
            except Exception as e:
                print(f"  ⚠️ فشل {model}/{qid}: {e}")
                return qid, []

    results_summary = {}
    async with httpx.AsyncClient() as client:
        for refine in [False, True]:
            for model in MODELS:
                tasks = [search_one(client, qid, qtext, model, refine)
                         for qid, qtext in queries.items()]
                outcomes = await asyncio.gather(*tasks)

                evaluations = []
                for qid, retrieved in outcomes:
                    q_qrels = [q for q in qrels if q["query_id"] == qid]
                    evaluations.append({"query_id": qid, "retrieved_docs": retrieved,
                                         "qrels": q_qrels, "k": 10})

                ev = httpx.post("http://localhost:8004/evaluate/batch",
                                 json={"evaluations": evaluations, "k": 10}, timeout=60).json()

                key = f"{model}_{'after' if refine else 'before'}_refinement"
                results_summary[key] = {
                    "MAP": ev["map_score"], "Recall": ev["mean_recall"],
                    "P@10": ev["mean_precision_at_k"], "nDCG": ev["mean_ndcg"],
                    "n_queries": len(queries),
                }
                print(key, results_summary[key])

    with open("data/touche/evaluation_results.json", "w", encoding="utf-8") as f:
        json.dump(results_summary, f, indent=2, ensure_ascii=False)
    print("\n✅ النتائج محفوظة بـ data/touche/evaluation_results.json")

asyncio.run(run_eval())