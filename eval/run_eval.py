"""Evaluation script for the IoT RAG retrieval + answering.

Reads ``eval/questions.csv`` and measures:
  - hit@1, hit@3, hit@5  (did the expected source file appear in the
    top-1/3/5 retrieved chunks?)
  - correct-refusal rate  (for out-of-scope questions, did the system
    reply with the not-found / out-of-scope message?)
  - average latency per question

Saves per-question results to ``eval/results.csv`` and a bar chart to
``eval/results.png``.

Run with::

    python -m eval.run_eval               # retrieval metrics only (fast)
    python -m eval.run_eval --with-llm    # also measure refusal via the LLM
"""
from __future__ import annotations

import argparse
import csv
import sys
import time
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import matplotlib

matplotlib.use("Agg")  # headless backend
import matplotlib.pyplot as plt

from app.answer import answer_question
from app.prompts import NOT_FOUND_REPLY, OUT_OF_SCOPE_REPLY
from app.retrieval import retrieve
from app.vectorstore.chroma_store import ChromaStore

EVAL_DIR = Path(__file__).resolve().parent
QUESTIONS_CSV = EVAL_DIR / "questions.csv"
RESULTS_CSV = EVAL_DIR / "results.csv"
RESULTS_PNG = EVAL_DIR / "results.png"


def load_questions() -> list[dict]:
    with open(QUESTIONS_CSV, encoding="utf-8") as f:
        return list(csv.DictReader(f))


def hit_at_k(expected_file: str, retrieved_files: list[str], k: int) -> bool:
    """Did the expected source file appear in the top-k retrieved files?"""
    return expected_file in retrieved_files[:k]


def is_refusal(answer: str) -> bool:
    """Did the answer refuse (not-found or out-of-scope)?"""
    a = answer.strip().lower()
    return (
        NOT_FOUND_REPLY.lower() in a
        or OUT_OF_SCOPE_REPLY.lower() in a
        or "couldn't find" in a
        or "outside the two documents" in a
    )


def run_eval(with_llm: bool = False) -> dict:
    store = ChromaStore()
    questions = load_questions()

    rows: list[dict] = []
    hits = {1: 0, 3: 0, 5: 0}
    n_file_scope = 0  # in-scope questions WITH an expected file (hit@k denom)
    n_scope = 0  # all in-scope questions (for doc_type accuracy)
    doc_type_correct = 0
    n_oos = 0  # out-of-scope questions
    correct_refusals = 0
    latencies: list[float] = []

    print(f"Evaluating {len(questions)} questions (with_llm={with_llm}) …\n")

    for q in questions:
        question = q["question"]
        expected_file = q.get("expected_source_file", "").strip()
        doc_type = q.get("doc_type", "").strip()
        is_oos = doc_type == "out_of_scope"

        start = time.time()
        result = retrieve(question, store)
        retrieved_files = [r.metadata.get("source_file", "") for r in result.results]
        retrieved_types = [r.metadata.get("doc_type", "") for r in result.results]

        row = {
            "question": question,
            "doc_type": doc_type,
            "expected_source_file": expected_file,
            "retrieved_top1": retrieved_files[0] if retrieved_files else "",
            "doc_type_ok": "",
            "hit@1": "",
            "hit@3": "",
            "hit@5": "",
            "refused": "",
        }

        if is_oos:
            n_oos += 1
            if with_llm:
                resp = answer_question(question, store)
                refused = is_refusal(resp.answer)
            else:
                # Heuristic without LLM: treat as refusal if nothing scored high
                best = max((r.metadata.get("_vector_score", r.score) for r in result.results), default=0)
                refused = best < 0.3
            correct_refusals += int(refused)
            row["refused"] = refused
        else:
            n_scope += 1
            # Primary metric: did the top result come from the right kind of doc?
            dt_ok = bool(retrieved_types) and retrieved_types[0] == doc_type
            doc_type_correct += int(dt_ok)
            row["doc_type_ok"] = dt_ok

            # File-level hit@k only when an expected file is specified
            if expected_file:
                n_file_scope += 1
                for k in (1, 3, 5):
                    hit = hit_at_k(expected_file, retrieved_files, k)
                    hits[k] += int(hit)
                    row[f"hit@{k}"] = hit

        latency = time.time() - start
        latencies.append(latency)
        row["latency_s"] = round(latency, 2)
        rows.append(row)

        status = ""
        if is_oos:
            status = f"refused={row['refused']}"
        else:
            status = f"doc_type_ok={row['doc_type_ok']} hit@3={row['hit@3']}"
        print(f"  [{doc_type:<12}] {question[:48]:<48} {status}  ({latency:.1f}s)")

    # ---- Metrics ----
    metrics = {
        "doc_type_accuracy": doc_type_correct / n_scope if n_scope else 0.0,
        "hit@1": hits[1] / n_file_scope if n_file_scope else 0.0,
        "hit@3": hits[3] / n_file_scope if n_file_scope else 0.0,
        "hit@5": hits[5] / n_file_scope if n_file_scope else 0.0,
        "correct_refusal_rate": correct_refusals / n_oos if n_oos else 0.0,
        "avg_latency_s": sum(latencies) / len(latencies) if latencies else 0.0,
        "n_in_scope": n_scope,
        "n_file_scope": n_file_scope,
        "n_out_of_scope": n_oos,
    }

    # ---- Save per-question CSV ----
    with open(RESULTS_CSV, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "question", "doc_type", "expected_source_file",
                "retrieved_top1", "doc_type_ok", "hit@1", "hit@3", "hit@5",
                "refused", "latency_s",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)

    # ---- Chart ----
    _save_chart(metrics)

    # ---- Print summary ----
    print(f"\n{'=' * 55}")
    print("EVALUATION SUMMARY")
    print(f"{'=' * 55}")
    print(f"  In-scope questions     : {metrics['n_in_scope']}")
    print(f"  (with expected file)   : {metrics['n_file_scope']}")
    print(f"  Out-of-scope questions : {metrics['n_out_of_scope']}")
    print(f"  doc_type accuracy      : {metrics['doc_type_accuracy']:.1%}")
    print(f"  hit@1                  : {metrics['hit@1']:.1%}")
    print(f"  hit@3                  : {metrics['hit@3']:.1%}")
    print(f"  hit@5                  : {metrics['hit@5']:.1%}")
    print(f"  correct refusal rate   : {metrics['correct_refusal_rate']:.1%}")
    print(f"  avg latency            : {metrics['avg_latency_s']:.2f}s")
    print(f"{'=' * 55}")
    print(f"\n  Results CSV : {RESULTS_CSV}")
    print(f"  Chart       : {RESULTS_PNG}")

    return metrics


def _save_chart(metrics: dict) -> None:
    labels = ["doc_type", "hit@1", "hit@3", "hit@5", "refusal"]
    values = [
        metrics["doc_type_accuracy"],
        metrics["hit@1"],
        metrics["hit@3"],
        metrics["hit@5"],
        metrics["correct_refusal_rate"],
    ]
    fig, ax = plt.subplots(figsize=(8, 4))
    bars = ax.bar(labels, values, color=["#937860", "#4C72B0", "#55A868", "#C44E52", "#8172B2"])
    ax.set_ylim(0, 1.0)
    ax.set_ylabel("Rate")
    ax.set_title("IoT RAG retrieval evaluation")
    for bar, v in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2, v + 0.02, f"{v:.0%}", ha="center")
    fig.tight_layout()
    fig.savefig(RESULTS_PNG, dpi=120)
    plt.close(fig)


def main() -> int:
    parser = argparse.ArgumentParser(description="Evaluate IoT RAG retrieval.")
    parser.add_argument(
        "--with-llm",
        action="store_true",
        help="Also call the LLM to measure refusal rate (slower).",
    )
    args = parser.parse_args()
    run_eval(with_llm=args.with_llm)
    return 0


if __name__ == "__main__":
    sys.exit(main())
