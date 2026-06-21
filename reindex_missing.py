"""
reindex_missing.py
يقارن كل doc_ids الموجودة بـ ir_datasets مع الموجودة فعلياً بقاعدة البيانات،
ويفهرس فقط الناقصة (بدون أي قص أو تجاهل أي وثيقة، حتى الفاضية تقريباً).
"""
import asyncio
import httpx
import psycopg2
import ir_datasets

INDEXING_URL = "http://localhost:8002/index"
BATCH = 250

def get_existing_ids():
    conn = psycopg2.connect(host="localhost", port=5432, dbname="ir2_db",
                             user="postgres", password="root")
    cur = conn.cursor()
    cur.execute("SELECT doc_id FROM documents")
    ids = set(r[0] for r in cur.fetchall())
    cur.close(); conn.close()
    return ids

async def main():
    existing = get_existing_ids()
    print(f"موجود حالياً بقاعدة البيانات: {len(existing):,}")

    ds = ir_datasets.load("beir/webis-touche2020")
    missing = []
    total = 0
    empty_count = 0

    for doc in ds.docs_iter():
        total += 1
        doc_id = str(doc.doc_id)
        if doc_id in existing:
            continue
        raw_text = getattr(doc, "text", "") or ""
        title = getattr(doc, "title", "") or ""
        if not raw_text.strip():
            empty_count += 1
        missing.append({
            "doc_id": doc_id,
            # ★ لا نتجاهل أي وثيقة حتى الفاضية تماماً — نضع مسافة كحد أدنى
            #   لتفادي مشاكل NOT NULL، لكنها تبقى موجودة بالفهرس فعلياً
            "text": raw_text if raw_text.strip() else " ",
            "title": title[:200],
            "metadata": {},
        })

    print(f"إجمالي الداتاسيت الحقيقي: {total:,}")
    print(f"الناقص فعلياً وسيُفهرس الآن: {len(missing):,}")
    print(f"  منها فاضية تماماً (نص خالٍ بالأصل): {empty_count:,}")

    if not missing:
        print("✅ لا يوجد نقص — كل الوثائق موجودة فعلياً.")
        return

    async with httpx.AsyncClient(timeout=300) as client:
        for i in range(0, len(missing), BATCH):
            batch = missing[i:i + BATCH]
            r = await client.post(INDEXING_URL,
                                   json={"dataset_id": "touche", "documents": batch})
            print(f"  batch {i}-{i+len(batch)}: status={r.status_code}, "
                  f"indexed={r.json().get('indexed')}")

    print("✅ انتهى استكمال الفهرسة الناقصة.")

asyncio.run(main())