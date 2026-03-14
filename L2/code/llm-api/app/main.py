from __future__ import annotations

from contextlib import asynccontextmanager

from fastapi import FastAPI

from app.api.routes import router as api_router
from app.db.database import verify_database


@asynccontextmanager
async def lifespan(_: FastAPI):
    verify_database()
    yield


app = FastAPI(title="Lab 2 Q&A Assistant API", version="1.0.0", lifespan=lifespan)
app.include_router(api_router)


@app.get("/health")
def healthcheck() -> dict[str, str]:
    return {"status": "ok"}
