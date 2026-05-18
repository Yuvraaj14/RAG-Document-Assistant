# test_llm.py
from app.core.llm import get_llm

def test_ollama():
    print("=" * 40)
    print("Testing Ollama (local)...")
    print("=" * 40)
    llm = get_llm(use_local=True)
    response = llm.invoke("Say hello in one sentence only.")
    print(f"✅ Ollama response: {response}")

def test_groq():
    print("=" * 40)
    print("Testing Groq API...")
    print("=" * 40)
    llm = get_llm(use_local=False)
    response = llm.invoke("Say hello in one sentence only.")
    print(f"✅ Groq response: {response.content}")

if __name__ == "__main__":
    test_ollama()
    print()
    test_groq()
    print()
    print("🎉 Both LLMs working! Moving to Day 2 — FAISS vector store.")