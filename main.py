
@asynccontextmanager
async def lifespan(app):

    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)

    yield

from fastapi import FastAPI
from sqlalchemy import text
from db import engine


from contextlib import asynccontextmanager

from db import engine
from database_models import Base

app = FastAPI(
    lifespan=lifespan
)

@app.get("/")
def home():
    return {"message": "IR Project Running"}

@app.get("/db-test")
async def db_test():
    async with engine.connect() as conn:
        await conn.execute(text("SELECT 1"))
        return {"database": "connected"}