# app/core/cache.py
# WHY REDIS CACHING?
# If user asks same question twice → no need to call Groq API again
# Redis stores question → answer pairs with TTL (time to live)
# Result: faster responses + fewer API calls
#
# WHY UPSTASH REDIS?
# Regular Redis needs a server running 24/7
# Upstash is serverless Redis — free tier, no server needed
# We call it via HTTP REST API — works from anywhere

import os
import json
import hashlib
import redis
from dotenv import load_dotenv

load_dotenv()

def get_redis_client():
    """
    Returns Redis client connected to Upstash.
    WHY: centralised client creation, easy to swap Redis provider
    """
    return redis.Redis(
        host=os.getenv("REDIS_URL", "").replace("https://", "").replace("http://", ""),
        port=6379,
        password=os.getenv("REDIS_TOKEN"),
        ssl=True,  # Upstash requires SSL
        decode_responses=True  # returns strings not bytes
    )

def make_cache_key(question: str) -> str:
    """
    Creates a cache key from the question.
    WHY HASH: keeps keys short and consistent
    MD5 is fine here — not for security, just for key generation
    """
    return f"rag:answer:{hashlib.md5(question.lower().strip().encode()).hexdigest()}"

def get_cached_answer(question: str) -> dict | None:
    """
    Checks Redis for cached answer.
    Returns None if not found or Redis unavailable.
    """
    try:
        client = get_redis_client()
        key = make_cache_key(question)
        cached = client.get(key)
        if cached:
            print(f"⚡ Cache HIT for: '{question[:50]}...'")
            return json.loads(cached)
        print(f"❌ Cache MISS for: '{question[:50]}...'")
        return None
    except Exception as e:
        # If Redis fails, don't crash the app — just skip caching
        print(f"⚠️  Redis unavailable: {e}")
        return None

def cache_answer(question: str, answer_data: dict, ttl_seconds: int = 3600):
    """
    Stores answer in Redis with TTL.
    ttl_seconds=3600 means cache expires after 1 hour
    WHY TTL: prevents stale answers if document is updated
    """
    try:
        client = get_redis_client()
        key = make_cache_key(question)
        client.setex(key, ttl_seconds, json.dumps(answer_data))
        print(f"💾 Cached answer for: '{question[:50]}...'")
    except Exception as e:
        print(f"⚠️  Could not cache: {e}")