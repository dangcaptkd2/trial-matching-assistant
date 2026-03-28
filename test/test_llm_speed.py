"""
LLM Speed Diagnostic Test
=========================
Run from the project root:
    .venv/bin/python -m test.test_llm_speed

Tests (in order):
  1. Raw httpx call to OpenAI  → pure network latency
  2. LangChain ChatOpenAI      → LangChain overhead
  3. LangChain + LangSmith     → tracing overhead
  4. with_structured_output    → structured output overhead
  5. Async ainvoke             → async vs sync

Each test prints elapsed time so you can pinpoint the bottleneck.
"""

import asyncio
import os
import time

# ── 0. Load settings (reads .env, configures env vars) ──────────────────────
from src.config.settings import settings  # noqa: E402  (must be first)

MODEL = settings.llm_model           # e.g. "gpt-4.1-nano"
BASE_URL = settings.llm_model_url    # e.g. "https://api.openai.com/v1"
API_KEY = settings.openai_api_key
PROMPT = "Reply with exactly one word: Hello"


# ─────────────────────────────────────────────────────────────────────────────
def _separator(title: str) -> None:
    print(f"\n{'=' * 60}")
    print(f"  {title}")
    print("=" * 60)


# ─────────────────────────────────────────────────────────────────────────────
# TEST 1 – Raw httpx (no LangChain, no tracing)
# ─────────────────────────────────────────────────────────────────────────────
def test_raw_httpx() -> None:
    _separator("TEST 1 — Raw httpx (baseline network latency)")
    import httpx

    url = f"{BASE_URL.rstrip('/')}/chat/completions"
    payload = {
        "model": MODEL,
        "messages": [{"role": "user", "content": PROMPT}],
        "max_tokens": 10,
        "temperature": 0,
    }
    headers = {"Authorization": f"Bearer {API_KEY}", "Content-Type": "application/json"}

    t0 = time.perf_counter()
    resp = httpx.post(url, json=payload, headers=headers, timeout=60)
    elapsed = time.perf_counter() - t0

    resp.raise_for_status()
    content = resp.json()["choices"][0]["message"]["content"]
    print(f"  Response : {content!r}")
    print(f"  ⏱  {elapsed:.2f}s")


# ─────────────────────────────────────────────────────────────────────────────
# TEST 2 – LangChain ChatOpenAI, tracing DISABLED
# ─────────────────────────────────────────────────────────────────────────────
def test_langchain_no_tracing() -> None:
    _separator("TEST 2 — LangChain ChatOpenAI (tracing OFF)")
    # Temporarily disable tracing
    os.environ["LANGCHAIN_TRACING_V2"] = "false"

    from langchain_core.messages import HumanMessage
    from langchain_openai import ChatOpenAI

    llm = ChatOpenAI(model=MODEL, temperature=0, base_url=BASE_URL, max_tokens=10)

    t0 = time.perf_counter()
    resp = llm.invoke([HumanMessage(content=PROMPT)])
    elapsed = time.perf_counter() - t0

    print(f"  Response : {resp.content!r}")
    print(f"  ⏱  {elapsed:.2f}s")

    # Restore tracing setting
    os.environ["LANGCHAIN_TRACING_V2"] = "true" if settings.langchain_tracing_v2 else "false"


# ─────────────────────────────────────────────────────────────────────────────
# TEST 3 – LangChain ChatOpenAI, tracing ENABLED (LangSmith)
# ─────────────────────────────────────────────────────────────────────────────
def test_langchain_with_tracing() -> None:
    _separator("TEST 3 — LangChain ChatOpenAI (tracing ON → LangSmith latency)")
    os.environ["LANGCHAIN_TRACING_V2"] = "true"

    from langchain_core.messages import HumanMessage
    from langchain_openai import ChatOpenAI

    llm = ChatOpenAI(model=MODEL, temperature=0, base_url=BASE_URL, max_tokens=10)

    t0 = time.perf_counter()
    resp = llm.invoke([HumanMessage(content=PROMPT)])
    elapsed = time.perf_counter() - t0

    print(f"  Response : {resp.content!r}")
    print(f"  ⏱  {elapsed:.2f}s")
    print("  → Compare with TEST 2 to measure LangSmith tracing overhead")


# ─────────────────────────────────────────────────────────────────────────────
# TEST 4 – with_structured_output (schema parsing)
# ─────────────────────────────────────────────────────────────────────────────
def test_structured_output() -> None:
    _separator("TEST 4 — with_structured_output (tracing OFF)")
    os.environ["LANGCHAIN_TRACING_V2"] = "false"

    from pydantic import BaseModel, Field
    from langchain_core.messages import HumanMessage
    from langchain_openai import ChatOpenAI

    class SimpleAnswer(BaseModel):
        word: str = Field(description="A single word answer")

    llm = ChatOpenAI(model=MODEL, temperature=0, base_url=BASE_URL, max_tokens=50)
    structured_llm = llm.with_structured_output(SimpleAnswer)

    t0 = time.perf_counter()
    result = structured_llm.invoke([HumanMessage(content="Reply with exactly one word: Hello")])
    elapsed = time.perf_counter() - t0

    print(f"  Response : {result}")
    print(f"  ⏱  {elapsed:.2f}s")
    print("  → Compare with TEST 2 to measure structured-output overhead")

    os.environ["LANGCHAIN_TRACING_V2"] = "true" if settings.langchain_tracing_v2 else "false"


# ─────────────────────────────────────────────────────────────────────────────
# TEST 5 – Async ainvoke
# ─────────────────────────────────────────────────────────────────────────────
async def _async_call() -> tuple[str, float]:
    os.environ["LANGCHAIN_TRACING_V2"] = "false"
    from langchain_core.messages import HumanMessage
    from langchain_openai import ChatOpenAI

    llm = ChatOpenAI(model=MODEL, temperature=0, base_url=BASE_URL, max_tokens=10)
    t0 = time.perf_counter()
    resp = await llm.ainvoke([HumanMessage(content=PROMPT)])
    elapsed = time.perf_counter() - t0
    return resp.content, elapsed


def test_async() -> None:
    _separator("TEST 5 — Async ainvoke (tracing OFF)")
    content, elapsed = asyncio.run(_async_call())
    print(f"  Response : {content!r}")
    print(f"  ⏱  {elapsed:.2f}s")
    print("  → Compare with TEST 2 to see async vs sync difference")


# ─────────────────────────────────────────────────────────────────────────────
# TEST 6 – Multiple parallel async calls (like rerank_with_llm_node does)
# ─────────────────────────────────────────────────────────────────────────────
async def _parallel_calls(n: int = 5) -> float:
    os.environ["LANGCHAIN_TRACING_V2"] = "false"
    from langchain_core.messages import HumanMessage
    from langchain_openai import ChatOpenAI

    llm = ChatOpenAI(model=MODEL, temperature=0, base_url=BASE_URL, max_tokens=10)
    tasks = [llm.ainvoke([HumanMessage(content=PROMPT)]) for _ in range(n)]
    t0 = time.perf_counter()
    await asyncio.gather(*tasks)
    return time.perf_counter() - t0


def test_parallel_async() -> None:
    n = 5
    _separator(f"TEST 6 — {n}× parallel async calls (simulates reranking {n} trials)")
    elapsed = asyncio.run(_parallel_calls(n))
    print(f"  ⏱  {elapsed:.2f}s total for {n} concurrent calls")
    print(f"     (~{elapsed/n:.2f}s per call if sequential vs {elapsed:.2f}s parallel)")


# ─────────────────────────────────────────────────────────────────────────────
# Summary
# ─────────────────────────────────────────────────────────────────────────────
def print_summary_guide() -> None:
    print(f"\n{'=' * 60}")
    print("  DIAGNOSIS GUIDE")
    print("=" * 60)
    print("""
  TEST 1 ≈ pure network round-trip to OpenAI
  TEST 2 – TEST 1 = LangChain framework overhead
  TEST 3 – TEST 2 = LangSmith tracing overhead  ← often 1–3s extra
  TEST 4 – TEST 2 = structured_output overhead
  TEST 5 vs TEST 2 = async benefit (should be similar for single call)
  TEST 6 shows parallelism gains during reranking

  Common culprits for slowness:
    • LangSmith tracing adds a synchronous HTTP call per node
      → Fix: set LANGCHAIN_TRACING_V2=false in .env to disable
    • Base URL points to slow/remote self-hosted model
      → Check: llm_model_url in settings.py
    • Model is large (e.g. gpt-4o instead of gpt-4.1-nano)
      → Check: llm_model in settings.py
    • Rate limiting / quota → check API dashboard
    """)


# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print(f"\nModel  : {MODEL}")
    print(f"Base URL: {BASE_URL}")

    try:
        test_raw_httpx()
    except Exception as e:
        print(f"  ✗ {e}")

    try:
        test_langchain_no_tracing()
    except Exception as e:
        print(f"  ✗ {e}")

    try:
        test_langchain_with_tracing()
    except Exception as e:
        print(f"  ✗ {e}")

    try:
        test_structured_output()
    except Exception as e:
        print(f"  ✗ {e}")

    try:
        test_async()
    except Exception as e:
        print(f"  ✗ {e}")

    try:
        test_parallel_async()
    except Exception as e:
        print(f"  ✗ {e}")

    print_summary_guide()
