from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.api.routes import router
from app.database import init_db

@asynccontextmanager
async def lifespan(app: FastAPI):
    init_db()
    yield

app = FastAPI(
    title="ContextLens — Multimodal Visual Asset Intelligence & Mismatch Prevention",
    description=(
        "Production AI decision system combining structured vision understanding, semantic embeddings, "
        "and an explainable Mismatch Guard to safely pair visual assets with written content."
    ),
    version="1.0.0",
    lifespan=lifespan,
)

# Enable CORS for browser access
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Register API Router
app.include_router(router)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)

