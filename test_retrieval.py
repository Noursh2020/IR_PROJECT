import httpx, asyncio
async def test():
    async with httpx.AsyncClient(timeout=300) as c:
        r = await c.post('http://localhost:8003/retrieve', json={'query': 'capital france', 'dataset_id': 'msmarco', 'model': 'bm25', 'top_k': 3})
        print(r.status_code)
        print(r.text[:500])
asyncio.run(test())