# app/main.py
# Entry point for the FastAPI application
# WHY FASTAPI over Flask/Django?
# FastAPI: async, automatic docs, Pydantic validation, 3x faster than Flask
# Django: too heavy for ML APIs
# Flask: no async, no automatic validation

from fastapi import FastAPI, Depends, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from app.api.routes import router
from app.api.auth import (
    verify_password, create_access_token,
    DEMO_USER
)
from app.models.schemas import LoginRequest, TokenResponse

# Create FastAPI app
app = FastAPI(
    title="RAG Assistant API",
    description="Document Q&A powered by LangChain + FAISS + Groq",
    version="1.0.0"
)

# CORS middleware — allows frontend (Gradio on HF Spaces) to call this API
# WHY CORS: browsers block cross-origin requests by default
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # in production: specify exact frontend URL
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount all routes from routes.py
app.include_router(router, prefix="/api")

@app.post("/login", response_model=TokenResponse)
def login(request: LoginRequest):
    """
    Login endpoint — returns JWT token.
    WHY /login at root not /api/login:
    OAuth2PasswordBearer expects tokenUrl="/login"
    """
    if request.username != DEMO_USER["username"]:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid credentials"
        )
    if not verify_password(request.password, DEMO_USER["hashed_password"]):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid credentials"
        )
    token = create_access_token({"sub": request.username})
    return TokenResponse(access_token=token)

@app.get("/health")
def health():
    """Health check endpoint — used by K8s to verify app is running."""
    return {"status": "healthy", "service": "rag-assistant"}