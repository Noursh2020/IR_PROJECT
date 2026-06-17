import csv
import json
from pathlib import Path

BASE = Path(r"C:\Users\LENOVO\Desktop\IR\.ir_datasets\msmarco-passage")

def load_queries():
    queries = {}
    with open(BASE / "dev/small/queries.tsv", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split("\t")
            if len(parts) == 2:
                queries[parts[0]] = parts[1]
    print(f"Queries: {len(queries)}")
    return queries

def load_qrels():
    qrels = {}
    with open(BASE / "dev/small/qrels", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 4:
                qid, _, did, rel = parts[0], parts[1], parts[2], parts[3]
                if qid not in qrels:
                    qrels[qid] = {}
                qrels[qid][did] = int(rel)
    print(f"Qrels: {len(qrels)}")
    return qrels

if __name__ == "__main__":
    Path("data/msmarco").mkdir(parents=True, exist_ok=True)
    
    queries = load_queries()
    qrels = load_qrels()
    
    with open("data/msmarco/queries.json", "w") as f:
        json.dump(queries, f)
    
    with open("data/msmarco/qrels.json", "w") as f:
        json.dump(qrels, f)
    
    print("Done")