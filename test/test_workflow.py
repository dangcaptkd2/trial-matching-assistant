"""
Interactive manual test for the clinical-trial matching workflow.

Run from the project root:

    python -m test.test_workflow
    # or
    python test/test_workflow.py

You will be prompted to type any query (English or Vietnamese).
Type 'quit' or press Ctrl-C / Ctrl-D to exit.

Example query:
    Bạn có thể giải thích các tiêu chí thu nhận của NCT05104866 bằng những từ ngữ đơn giản không?
"""

import asyncio
import sys
import textwrap


async def run_once(query: str, thread_id: str = "manual-test-1") -> None:
    """Invoke the workflow with *query* and pretty-print the result."""
    # Import here so the module can be imported without side-effects
    from src.core.graph import create_workflow

    app = create_workflow()

    initial_state = {
        "user_input": query,
        "messages": [],
        "top_k": 5,
    }
    config = {"configurable": {"thread_id": thread_id}}

    print("\n" + "=" * 70)
    print("Running workflow …")
    print("=" * 70)

    try:
        result = await app.ainvoke(initial_state, config=config)
    except Exception as exc:
        import traceback

        print(f"\n✗ Workflow raised an exception: {exc}")
        traceback.print_exc()
        return

    # ── Intent & extraction ──────────────────────────────────────────────────
    intent_type        = result.get("intent_type", "—")
    patient_info       = result.get("patient_info", "—")
    trial_ids          = result.get("trial_ids") or []
    location_info      = result.get("location_info", "—")
    clarification      = result.get("clarification_reason", "—")
    chitchat_response  = result.get("chitchat_response", "")

    print(f"\n📌 Intent          : {intent_type}")
    print(f"👤 Patient info    : {patient_info}")
    print(f"🔖 Trial IDs       : {trial_ids if trial_ids else '—'}")
    print(f"📍 Location        : {location_info}")

    if clarification and clarification != "—":
        print(f"❓ Clarification   : {clarification}")

    # ── Search & rerank results ─────────────────────────────────────────────
    search_results   = result.get("search_results", []) or []
    reranked_results = result.get("reranked_results", []) or []
    trial_data       = result.get("trial_data", []) or []

    print(f"\n🔍 Search results  : {len(search_results)} trial(s)")
    print(f"🏆 Reranked results: {len(reranked_results)} trial(s)")

    if reranked_results:
        print("\n  Top reranked trials:")
        for i, trial in enumerate(reranked_results[:5], 1):
            nct_id = trial.get("nct_id") or trial.get("id", "N/A")
            title  = trial.get("title", "N/A")
            score  = trial.get("LLM_match_score") or trial.get("score", "")
            line   = f"  {i}. {nct_id}: {title[:65]}"
            if score != "":
                line += f"  [score={score}]"
            print(line)
    elif search_results:
        print("\n  Top search hits:")
        for i, trial in enumerate(search_results[:5], 1):
            nct_id = trial.get("nct_id") or trial.get("id", "N/A")
            title  = trial.get("title", "N/A")
            print(f"  {i}. {nct_id}: {title[:65]}")

    if trial_data:
        print(f"\n📄 Trial data fetched: {len(trial_data)} document(s)")
        for i, doc in enumerate(trial_data[:3], 1):
            nct_id = doc.get("nct_id") or doc.get("id", "N/A")
            print(f"  {i}. {nct_id}")

    # ── Chitchat / off-topic ─────────────────────────────────────────────────
    if chitchat_response:
        print("\n💬 Chitchat response:")
        print(textwrap.indent(textwrap.fill(chitchat_response, width=75), "  "))

    # ── Final answer ─────────────────────────────────────────────────────────
    final_answer = result.get("final_answer", "")
    if final_answer:
        print("\n✅ Final answer:")
        print("-" * 70)
        # Pretty-wrap long lines but keep newlines that are already there
        for paragraph in final_answer.split("\n"):
            if paragraph.strip():
                print(textwrap.fill(paragraph, width=75))
            else:
                print()
        print("-" * 70)
    else:
        print("\n⚠️  No final_answer produced.")

    # ── Token usage ──────────────────────────────────────────────────────────
    prompt_tokens     = result.get("prompt_tokens", 0) or 0
    completion_tokens = result.get("completion_tokens", 0) or 0
    total_tokens      = result.get("total_tokens", 0) or 0
    if total_tokens:
        print(
            f"\n🔢 Tokens used — prompt: {prompt_tokens}, "
            f"completion: {completion_tokens}, total: {total_tokens}"
        )


async def interactive_loop() -> None:
    """Read queries from stdin in a loop."""
    print("=" * 70)
    print("  Clinical Trial Matching — Interactive Workflow Test")
    print("=" * 70)
    print("Type your query (English or Vietnamese) and press Enter.")
    print("Type 'quit' to exit.\n")

    thread_counter = 0

    while True:
        try:
            query = input("Query > ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nBye!")
            break

        if not query:
            continue
        if query.lower() in {"quit", "exit", "q"}:
            print("Bye!")
            break

        thread_counter += 1
        await run_once(query, thread_id=f"manual-test-{thread_counter}")
        print()


def main() -> None:
    query = "Bạn có thể giải thích các tiêu chí thu nhận của NCT05104866 bằng những từ ngữ đơn giản không?"
    asyncio.run(run_once(query, thread_id="manual-test-cli"))
   


if __name__ == "__main__":
    main()
