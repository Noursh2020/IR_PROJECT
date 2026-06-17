import csv
import json
import pickle
from pathlib import Path
from collections import defaultdict
from tqdm import tqdm
from rank_bm25 import BM25Okapi
from services.preprocessing.main import preprocess_single

BASE = Path(r"C:\Users\LENOVO\Desktop\IR\.ir_datasets\msmarco-passage")
INDEX_DIR = Path("indexes/msmarco")
INDEX_DIR.mkdir(parents=True, exist_ok=True)

def build():
    doc_ids = []
    corpus = []
    
    print("Reading collection.tsv and building indexes...")
    with open(BASE / "collection.tsv", encoding="utf-8") as f:
        reader = csv.reader(f, delimiter="\t")
        for row in tqdm(reader, desc="Processing docs"):
            if len(row) < 2:
                continue
            doc_id, text = row[0], row[1]
            tokens = preprocess_single(text).processed_tokens
            if not tokens:
                continue
            doc_ids.append(doc_id)
            corpus.append(tokens)
    
    print(f"Total docs: {len(doc_ids)}")
    
    print("Building BM25...")
    bm25 = BM25Okapi(corpus)
    
    print("Saving...")
    with open(INDEX_DIR / "bm25.pkl", "wb") as f:
        pickle.dump(bm25, f)
    
    with open(INDEX_DIR / "doc_ids.json", "w") as f:
        json.dump(doc_ids, f)
    
    print("Done")

if __name__ == "__main__":
    build()