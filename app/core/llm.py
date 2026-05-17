# app/core/llm.py
import os
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_ollama import OllamaLLM

load_dotenv()

def get_llm(use_local: bool = False):
    """
    use_local=True  → Ollama (free, local, development)
    use_local=False → Groq API (fast, free tier, production)
    """
    if use_local:
        return OllamaLLM(
            model="llama3.2",
            base_url="http://localhost:11434"
        )
    else:
        return ChatGroq(
            api_key=os.getenv("GROQ_API_KEY"),
            model_name="llama-3.3-70b-versatile",  # current active model
            temperature=0,
            max_tokens=1024
        )