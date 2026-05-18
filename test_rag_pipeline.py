# test_rag_pipeline.py
# Tests the complete RAG pipeline end to end:
# PDF → FAISS index → retriever → LLM → cited answer

from app.core.rag_pipeline import build_rag_chain, ask_question

def test_full_rag():
    print("=" * 55)
    print("Day 3 — Testing Complete RAG Pipeline")
    print("=" * 55)

    # Build RAG chain using existing FAISS index from Day 2
    # use_local_llm=False → use Groq (faster)
    print("\n📌 Step 1: Build RAG chain")
    chain = build_rag_chain(use_local_llm=False)

    # Test questions based on the graph theory PDF
    questions = [
        "What is color energy in graph theory?",
        "What is a chromatic number?",
        "What are the main topics covered in this document?",
    ]

    print("\n📌 Step 2: Ask questions")
    print("-" * 55)

    results = []
    for question in questions:
        result = ask_question(chain, question)
        results.append(result)
        print("-" * 55)

    print("\n🎉 RAG Pipeline working!")
    print("    Question → FAISS retrieval → Groq LLM → Cited answer ✅")

if __name__ == "__main__":
    test_full_rag()