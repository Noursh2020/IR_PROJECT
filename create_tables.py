from db import get_connection

conn = get_connection()
cur = conn.cursor()

cur.execute("""
    CREATE TABLE IF NOT EXISTS documents (
        doc_id TEXT PRIMARY KEY,
        title TEXT DEFAULT '',
        content TEXT NOT NULL,
        metadata TEXT DEFAULT '{}'
    )
""")

cur.execute("""
    CREATE TABLE IF NOT EXISTS terms (
        id SERIAL PRIMARY KEY,
        term TEXT UNIQUE NOT NULL
    )
""")

cur.execute("""
    CREATE TABLE IF NOT EXISTS postings (
        term_id INTEGER REFERENCES terms(id),
        doc_id TEXT REFERENCES documents(doc_id),
        tf INTEGER NOT NULL,
        PRIMARY KEY (term_id, doc_id)
    )
""")

cur.execute("""
    CREATE TABLE IF NOT EXISTS indexing_progress (
        dataset_id TEXT PRIMARY KEY,
        last_index INTEGER DEFAULT 0
    )
""")

conn.commit()
cur.close()
conn.close()
print("All tables created")