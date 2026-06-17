import psycopg2
from psycopg2.extras import execute_values

def get_connection():
    return psycopg2.connect(
        host="localhost",
        port=5432,
        dbname="ir_db",
        user="postgres",
        password="root"
    )

def create_tables():
    conn = get_connection()
    cur = conn.cursor()
    cur.execute("""
        CREATE TABLE IF NOT EXISTS documents (
            doc_id TEXT PRIMARY KEY,
            raw_text TEXT NOT NULL
        )
    """)
    conn.commit()
    cur.close()
    conn.close()
    print("Tables created successfully")

if __name__ == "__main__":
    create_tables()