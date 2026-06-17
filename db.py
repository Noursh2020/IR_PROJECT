from sqlalchemy.ext.asyncio import create_async_engine

DATABASE_URL = (
    "postgresql+asyncpg://postgres:root@localhost:5432/ir_db"
)

engine = create_async_engine(
    DATABASE_URL,
    echo=True
)