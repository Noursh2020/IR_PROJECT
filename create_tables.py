"""
create_tables_ir2.py
يُنشئ نفس الـ schema بس على قاعدة بيانات منفصلة ir2_db
مخصصة لـ dataset: beir/webis-touche2020
"""
import psycopg2

DB_CONFIG = {
    "host": "localhost",
    "port": 5432,
    "dbname": "ir2_db",   # ← القاعدة الجديدة
    "user": "postgres",
    "password": "root",
}

def get_connection():
    return psycopg2.connect(**DB_CONFIG)

def create_tables():
    conn = get_connection()
    cur = conn.cursor()

    cur.execute("""
        CREATE TABLE IF NOT EXISTS documents (
            doc_id TEXT PRIMARY KEY,
            title TEXT DEFAULT '',
            raw_text TEXT NOT NULL,
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
    print("✅ ir2_db tables created successfully")

if __name__ == "__main__":
     create_tables()