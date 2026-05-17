# app/api/routes.py
# WHY THIS FILE?
# Keeps all API endpoints in one place
# main.py just mounts these routes — clean separation of concerns

import os
from fastapi import APIRouter, UploadFile, File, Depends, HTTPException
from app.api.auth import get_current_user
from app.core.rag_pipeline import build_rag_chain, ask_question
from app.core.cache import get_cached_answer, cache_answer
from app.models.schemas import QuestionRequest, AnswerResponse, UploadResponse

router = APIRouter()

# Global chain — built once, reused for all requests
# WHY GLOBAL: rebuilding chain on every request = slow
rag_chain_tuple = None

@router.post("/upload", response_model=UploadResponse)
async def upload_pdf(
    file: UploadFile = File(...),
    current_user: str = Depends(get_current_user)  # JWT protected
):
    """
    Accepts PDF upload → builds FAISS index → ready for questions.
    WHY async: file I/O is non-blocking, FastAPI handles concurrent uploads
    """
    global rag_chain_tuple

    if not file.filename.endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files accepted")

    # Save uploaded file temporarily
    pdf_path = f"temp_{file.filename}"
    with open(pdf_path, "wb") as f:
        content = await file.read()
        f.write(content)

    # Build RAG chain from uploaded PDF
    rag_chain_tuple = build_rag_chain(pdf_path=pdf_path)

    # Count chunks for response
    from app.core.embeddings import load_faiss_index
    vectorstore = load_faiss_index()
    chunk_count = vectorstore.index.ntotal

    # Clean up temp file
    os.remove(pdf_path)

    return UploadResponse(
        message="PDF processed successfully",
        chunks_created=chunk_count,
        filename=file.filename
    )

@router.post("/ask", response_model=AnswerResponse)
async def ask(
    request: QuestionRequest,
    current_user: str = Depends(get_current_user)  # JWT protected
):
    """
    Accepts question → checks Redis cache → if miss, runs RAG chain.
    WHY CACHE CHECK FIRST: saves Groq API calls for repeated questions
    """
    global rag_chain_tuple

    if rag_chain_tuple is None:
        # Try loading existing index if no PDF uploaded this session
        from app.core.embeddings import index_exists
        if index_exists():
            rag_chain_tuple = build_rag_chain()
        else:
            raise HTTPException(
                status_code=400,
                detail="No document uploaded. POST /upload first."
            )

    # Check Redis cache first
    cached = get_cached_answer(request.question)
    if cached:
        cached["cached"] = True
        return AnswerResponse(**cached)

    # Cache miss — run RAG chain
    result = ask_question(rag_chain_tuple, request.question)

    # Store in cache for next time
    cache_answer(request.question, {
        "question": request.question,
        "answer": result["answer"],
        "sources": result["sources"],
        "cached": False
    })

    return AnswerResponse(
        question=request.question,
        answer=result["answer"],
        sources=result["sources"],
        cached=False
    )