"""
RAG evaluation script using Ragas.
Runs 20 hardcoded Q&A pairs and outputs a markdown results table.

Usage:
    ANTHROPIC_API_KEY=... python evals/run_evals.py --backend http://localhost:8000
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "backend"))

import requests
from datasets import Dataset
from ragas import evaluate
from ragas.metrics import answer_faithfulness, context_recall, context_precision

from eval_pairs import EVAL_PAIRS


def call_backend(backend_url: str, question: str) -> dict:
    resp = requests.post(f"{backend_url}/query", json={"question": question}, timeout=120)
    resp.raise_for_status()
    return resp.json()


def keyword_hit(answer: str, keywords: list[str]) -> bool:
    lower = answer.lower()
    return any(kw.lower() in lower for kw in keywords)


def build_ragas_dataset(results: list[dict]) -> Dataset:
    return Dataset.from_dict(
        {
            "question": [r["question"] for r in results],
            "answer": [r["answer"] for r in results],
            "contexts": [r["contexts"] for r in results],
            "ground_truth": [r["ground_truth"] for r in results],
        }
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--backend", default="http://localhost:8000")
    parser.add_argument("--skip-ragas", action="store_true", help="Skip Ragas metrics, keyword-only")
    args = parser.parse_args()

    print(f"Running {len(EVAL_PAIRS)} eval pairs against {args.backend}\n")

    rows = []
    ragas_input = []

    for pair in EVAL_PAIRS:
        qid = pair["id"]
        question = pair["question"]
        expected_kws = pair["expected_keywords"]

        try:
            resp = call_backend(args.backend, question)
            answer = resp["answer"]
            sources = resp["sources"]
            contexts = [s.get("excerpt", "") for s in sources]
            retrieval_ok = bool(contexts)
            kw_pass = keyword_hit(answer, expected_kws)
            hallucination_flag = not kw_pass and retrieval_ok
            status = "PASS" if kw_pass else "FAIL"
        except Exception as exc:
            answer = f"ERROR: {exc}"
            contexts = []
            retrieval_ok = False
            kw_pass = False
            hallucination_flag = False
            status = "ERROR"

        rows.append(
            {
                "id": qid,
                "question": question[:60] + ("…" if len(question) > 60 else ""),
                "retrieval": "yes" if retrieval_ok else "no",
                "kw_match": "yes" if kw_pass else "no",
                "hallucination": "YES" if hallucination_flag else "no",
                "status": status,
            }
        )
        ragas_input.append(
            {
                "question": question,
                "answer": answer,
                "contexts": contexts,
                "ground_truth": ", ".join(expected_kws),
            }
        )
        print(f"  [{qid}] {status}")

    # ── Markdown table ─────────────────────────────────────────────────────────
    header = "| ID   | Question (truncated)                                           | Retrieval | KW Match | Hallucination | Result |"
    sep    = "|------|----------------------------------------------------------------|-----------|----------|---------------|--------|"
    print("\n## Eval Results\n")
    print(header)
    print(sep)
    for r in rows:
        print(f"| {r['id']} | {r['question']:<62} | {r['retrieval']:<9} | {r['kw_match']:<8} | {r['hallucination']:<13} | {r['status']} |")

    passed = sum(1 for r in rows if r["status"] == "PASS")
    print(f"\n**Summary:** {passed}/{len(rows)} passed")

    # ── Ragas metrics ──────────────────────────────────────────────────────────
    if not args.skip_ragas:
        try:
            print("\nRunning Ragas metrics (requires ANTHROPIC_API_KEY)…")
            dataset = build_ragas_dataset(ragas_input)
            result = evaluate(
                dataset,
                metrics=[context_precision, context_recall, answer_faithfulness],
            )
            print("\n## Ragas Scores\n")
            print(f"| Metric               | Score  |")
            print(f"|----------------------|--------|")
            for metric, score in result.items():
                print(f"| {metric:<20} | {score:.4f} |")
        except Exception as exc:
            print(f"\nRagas evaluation skipped: {exc}")


if __name__ == "__main__":
    main()
