# app/models/schemas.py
# WHY PYDANTIC SCHEMAS?
# FastAPI uses these to:
# 1. Validate incoming request data automatically
# 2. Generate API documentation at /docs
# 3. Return consistent response shapes
# If data doesn't match schema → FastAPI returns 422 error automatically

from pydantic import BaseModel
from typing import List, Optional

class LoginRequest(BaseModel):
    username: str
    password: str

class TokenResponse(BaseModel):
    access_token: str
    token_type: str = "bearer"

class QuestionRequest(BaseModel):
    question: str
    use_local_llm: bool = False  # default: use Groq

class SourceDocument(BaseModel):
    page: int
    content: str

class AnswerResponse(BaseModel):
    question: str
    answer: str
    sources: List[SourceDocument]
    cached: bool = False  # tells client if answer came from Redis cache

class UploadResponse(BaseModel):
    message: str
    chunks_created: int
    filename: str