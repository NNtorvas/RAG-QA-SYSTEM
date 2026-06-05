"""
RAG evaluation script using Ragas.
Runs 20 hardcoded Q&A pairs and outputs a markdown results table.

Usage:
    # keyword-only (no LLM needed)
    python evals/run_evals.py --backend http://localhost:8000 --skip-ragas

    # Ragas metrics via Anthropic
    ANTHROPIC_API_KEY=... python evals/run_evals.py --backend http://localhost:8000

    # Ragas metrics via HuggingFace Inference API (free)
    HUGGINGFACE_API_KEY=... python evals/run_evals.py --backend http://localhost:8000 --eval-llm huggingface
"""

import argparse
import sys
from pathlib import Path

import requests
from datasets import Dataset

sys.path.insert(0, str(Path(__file__).parent.parent / "backend"))

from eval_pairs import EVAL_PAIRS  # noqa: E402


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
    parser.add_argument(
        "--eval-llm",
        choices=["anthropic", "huggingface"],
        default="anthropic",
        help="LLM provider for Ragas scoring (default: anthropic)",
    )
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
    header = (
        "| ID   | Question (truncated)                                           "
        "| Retrieval | KW Match | Hallucination | Result |"
    )
    sep = (
        "|------|----------------------------------------------------------------"
        "|-----------|----------|---------------|--------|"
    )
    print("\n## Eval Results\n")
    print(header)
    print(sep)
    for r in rows:
        print(
            f"| {r['id']} | {r['question']:<62} | {r['retrieval']:<9} "
            f"| {r['kw_match']:<8} | {r['hallucination']:<13} | {r['status']} |"
        )

    passed = sum(1 for r in rows if r["status"] == "PASS")
    print(f"\n**Summary:** {passed}/{len(rows)} passed")

    # ── Ragas metrics ──────────────────────────────────────────────────────────
    if not args.skip_ragas:
        try:
            import os

            from ragas import evaluate
            from ragas.llms import LangchainLLMWrapper
            from ragas.metrics import context_precision, context_recall, faithfulness

            if args.eval_llm == "huggingface":
                from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint

                hf_token = os.getenv("HUGGINGFACE_API_KEY")
                if not hf_token:
                    raise EnvironmentError("HUGGINGFACE_API_KEY is not set.")
                endpoint = HuggingFaceEndpoint(
                    repo_id="mistralai/Mistral-7B-Instruct-v0.3",
                    huggingfacehub_api_token=hf_token,
                )
                ragas_llm = LangchainLLMWrapper(ChatHuggingFace(llm=endpoint))
                print("\nRunning Ragas metrics via HuggingFace (Mistral-7B)…")
            else:
                from langchain_anthropic import ChatAnthropic

                anthropic_key = os.getenv("ANTHROPIC_API_KEY")
                if not anthropic_key:
                    raise EnvironmentError("ANTHROPIC_API_KEY is not set.")
                ragas_llm = LangchainLLMWrapper(
                    ChatAnthropic(model="claude-haiku-4-5-20251001", anthropic_api_key=anthropic_key)
                )
                print("\nRunning Ragas metrics via Anthropic (claude-haiku)…")

            dataset = build_ragas_dataset(ragas_input)
            result = evaluate(
                dataset,
                metrics=[context_precision, context_recall, faithfulness],
                llm=ragas_llm,
            )
            print("\n## Ragas Scores\n")
            print("| Metric               | Score  |")
            print("|----------------------|--------|")
            for metric, score in result.items():
                print(f"| {metric:<20} | {score:.4f} |")
        except Exception as exc:
            print(f"\nRagas evaluation skipped: {exc}")


if __name__ == "__main__":
    main()
