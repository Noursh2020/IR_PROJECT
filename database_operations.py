from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import sessionmaker

from db import engine
from database_models import (
    DocumentDB,
    PostingDB,
    ProgressDB
)

SessionLocal = sessionmaker(
    engine,
    class_=AsyncSession,
    expire_on_commit=False
)


async def save_doc(
    doc_id,
    title,
    text
):
    async with SessionLocal() as session:

        doc = DocumentDB(
            doc_id=doc_id,
            title=title,
            text=text
        )

        session.add(doc)

        await session.commit()
        
        
        
async def save_posting(
    term,
    doc_id,
    tf
):
    async with SessionLocal() as session:

        posting = PostingDB(
            term=term,
            doc_id=doc_id,
            tf=tf
        )

        session.add(posting)

        await session.commit()
        
        





async def save_progress(
    dataset_id,
    last_doc
):
    async with SessionLocal() as session:

        progress = ProgressDB(
            dataset_id=dataset_id,
            last_doc=last_doc
        )

        session.merge(progress)

        await session.commit()