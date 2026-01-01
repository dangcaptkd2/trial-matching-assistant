"""FastAPI application with OpenAI-compatible API."""

from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from src.api.routes import chat, conversations, health, models
from chainlit.utils import mount_chainlit


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifespan context manager for FastAPI app."""
    # Startup
    yield
    # Shutdown


# Create FastAPI app
app = FastAPI(
    title="Clinical Trial Assistant API",
    description="OpenAI-compatible API for clinical trial matching assistant",
    version="1.0.0",
    lifespan=lifespan,
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify allowed origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers
app.include_router(health.router, prefix="/api", tags=["health"])
app.include_router(conversations.router, prefix="/api", tags=["conversations"])
# OpenAI-compatible endpoints
app.include_router(chat.router, prefix="/v1", tags=["openai-chat"])
app.include_router(models.router, prefix="/v1", tags=["openai-models"])

# Mount Chainlit UI
mount_chainlit(app=app, target="src/ui/chainlit_app.py", path="/demo")


@app.get("/")
async def root():
    """Root endpoint."""
    return JSONResponse(
        {
            "message": "Clinical Trial Assistant API",
            "docs": "/docs",
            "openai_base_url": "/v1",
            "endpoints": {
                "chat": "/v1/chat/completions",
                "models": "/v1/models",
                "health": "/api/health",
                "conversations": "/api/conversations",
            },
        }
    )


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "healthy", "service": "clinical-trial-assistant"}
