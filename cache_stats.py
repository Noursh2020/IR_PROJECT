import json
import psycopg2
from db import get_connection

conn = get_connection()
cur = conn.cursor()

cur.execute("SELECT COUNT(*) FROM documents")
total_docs = cur.fetchone()[0]

cur.execute("""
    SELECT AVG(tf_sum) FROM (
        SELECT doc_id, SUM(tf) as tf_sum 
        FROM postings 
        GROUP BY doc_id
    ) t
""")
avg_dl = float(cur.fetchone()[0])

cur.close()
conn.close()

stats = {"total_documents": total_docs, "avg_doc_length": avg_dl}

with open("data/msmarco/stats.json", "w") as f:
    json.dump(stats, f)

print(stats)