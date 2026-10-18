"""CLI to test retrieval alone (no LLM).

Run with a question::

    python -m app.retrieve_cli "What topics are in Operating Systems?"
    python -m app.retrieve_cli "When do the first semester exams start?"

Or interactively (no argument)::

    python -m app.retrieve_cli
"""
from __future__ import annotations

import sys

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from app.retrieval import retrieve
from app.vectorstore.chroma_store import ChromaStore

SEP = "=" * 70


def run_one(question: str, store: ChromaStore) -> None:
    print(f"\n{SEP}")
    print(f"Q: {question}")
    print(SEP)

    result = retrieve(question, store)

    print("\nDebug:")
    for key, val in result.debug.items():
        print(f"  {key:<28}: {val}")

    print(f"\nTop {len(result.results)} results:")
    for i, r in enumerate(result.results, 1):
        meta = r.metadata
        print(f"\n  [{i}] score={r.score:.4f}  kind={meta.get('chunk_kind')}")
        print(f"      doc_type   : {meta.get('doc_type')}")
        if meta.get("course_code"):
            print(f"      course     : {meta.get('course_code')} {meta.get('course_title')}")
        if meta.get("event_type"):
            temporal = meta.get("_temporal", "")
            days = meta.get("_days_away", "")
            print(f"      event      : {meta.get('event_type')} ({temporal} {days})")
        if meta.get("start_date"):
            print(f"      dates      : {meta.get('start_date')} .. {meta.get('end_date')}")
        print(f"      source     : {meta.get('source_file')} p{meta.get('page_start')}")
        snippet = r.text[:160].replace("\n", " ")
        print(f"      text       : {snippet}")


def main() -> int:
    store = ChromaStore()
    print(f"ChromaDB loaded: {store.count()} chunks")

    if len(sys.argv) > 1:
        question = " ".join(sys.argv[1:])
        run_one(question, store)
        return 0

    # Interactive loop
    print("Enter a question (empty line to quit):")
    while True:
        try:
            question = input("\n> ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if not question:
            break
        run_one(question, store)
    return 0


if __name__ == "__main__":
    sys.exit(main())
