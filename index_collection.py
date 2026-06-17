import csv
import psycopg2
from psycopg2.extras import execute_values
from pathlib import Path
from tqdm import tqdm
from db import get_connection
from services.preprocessing.main import preprocess_single

COLLECTION = Path(r"C:\Users\LENOVO\Desktop\IR\.ir_datasets\msmarco-passage\collection.tsv")
BATCH_SIZE = 500

def get_term_ids(cur, terms):
    """احصل على IDs المصطلحات أو أنشئها."""
    if not terms:
        return {}
    execute_values(cur,
        "INSERT INTO terms (term) VALUES %s ON CONFLICT (term) DO NOTHING",
        [(t,) for t in terms]
    )
    cur.execute("SELECT term, id FROM terms WHERE term = ANY(%s)", (list(terms),))
    return {row[0]: row[1] for row in cur.fetchall()}

def index_batch(cur, batch):
    """فهرسة دفعة من الوثائق."""
    # 1. احفظ الوثائق
   
    execute_values(cur,
    "INSERT INTO documents (doc_id, raw_text, content) VALUES %s ON CONFLICT DO NOTHING",
    [(doc_id, text, text) for doc_id, text in batch]
)
    
    # 2. للكل وثيقة احسب TF وافهرس
    all_terms = set()
    doc_tokens = {}
    for doc_id, text in batch:
        tokens = preprocess_single(text).processed_tokens
        doc_tokens[doc_id] = tokens
        all_terms.update(tokens)
    
    if not all_terms:
        return
    
    # 3. احصل على term_ids
    term_ids = get_term_ids(cur, all_terms)
    
    # 4. احفظ postings
    postings = []
    for doc_id, tokens in doc_tokens.items():
        tf_map = {}
        for t in tokens:
            tf_map[t] = tf_map.get(t, 0) + 1
        for term, tf in tf_map.items():
            if term in term_ids:
                postings.append((term_ids[term], doc_id, tf))
    
    if postings:
        execute_values(cur,
            """INSERT INTO postings (term_id, doc_id, tf) VALUES %s
               ON CONFLICT (term_id, doc_id) DO UPDATE SET tf = EXCLUDED.tf""",
            postings
        )

def get_last_index(cur):
    cur.execute("SELECT last_index FROM indexing_progress WHERE dataset_id = 'msmarco'")
    row = cur.fetchone()
    return row[0] if row else 0

def save_progress(cur, idx):
    cur.execute("""
        INSERT INTO indexing_progress (dataset_id, last_index) VALUES ('msmarco', %s)
        ON CONFLICT (dataset_id) DO UPDATE SET last_index = EXCLUDED.last_index
    """, (idx,))

def main():
    conn = get_connection()
    cur = conn.cursor()
    
    start = get_last_index(cur)
    print(f"Resuming from doc {start}")
    
    batch = []
    total = 0
    
    with open(COLLECTION, encoding="utf-8") as f:
        reader = csv.reader(f, delimiter="\t")
        for i, row in enumerate(tqdm(reader, desc="Indexing")):
            if i < start:
                continue
            if len(row) < 2:
                continue
            
            doc_id, text = row[0], row[1]
            batch.append((doc_id, text))
            
            if len(batch) == BATCH_SIZE:
                index_batch(cur, batch)
                total += len(batch)
                save_progress(cur, i)
                conn.commit()
                batch = []
        
        if batch:
            index_batch(cur, batch)
            total += len(batch)
            conn.commit()
    
    cur.close()
    conn.close()
    print(f"Done — indexed {total} docs with real doc_ids")

if __name__ == "__main__":
    main()