"""
Evaluation script — runs a test suite and prints a quality report.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
from src.agent.graph import run_agent
from src.agent.nodes import _retriever
from src.evaluation.evaluator import evaluate
from src.utils.logging import get_logger, setup_logging
import time

setup_logging()
logger = get_logger(__name__)

TEST_QUESTIONS = [
    "What is Delta Lake?",
    "How do dbt models work?",
    "What is Spark structured streaming?",
    "How does Delta Lake handle ACID transactions?",
    "What are dbt sources?",
]


def main() -> None:
    print("\n🧪 Running RAG evaluation suite...\n")
    results = []

    for i, question in enumerate(TEST_QUESTIONS):
        if i > 0:
            time.sleep(10)
        print(f"  ▶ {question}")
        start = time.perf_counter()

        # Run agent
        result = run_agent(question)
        latency_ms = int((time.perf_counter() - start) * 1000)

        # Get contexts (chunk texts used)
        chunks = _retriever.retrieve(query=question)
        contexts = [c.content for c in chunks[:4]]

        # Evaluate
        eval_result = evaluate(
            query=question,
            answer=result["answer"],
            contexts=contexts,
            latency_ms=latency_ms,
        )
        results.append(eval_result)

    # Print report
    print("\n" + "="*60)
    print("📊 EVALUATION REPORT")
    print("="*60)
    print(f"{'Question':<40} {'Faith':>6} {'Relev':>6} {'Recall':>7} {'ms':>6}")
    print("-"*60)
    for r in results:
        print(
            f"{r.query[:39]:<40} "
            f"{r.faithfulness:>6.2f} "
            f"{r.answer_relevancy:>6.2f} "
            f"{r.context_recall:>7.2f} "
            f"{r.latency_ms:>6}"
        )
    print("-"*60)
    avg_f = sum(r.faithfulness for r in results) / len(results)
    avg_r = sum(r.answer_relevancy for r in results) / len(results)
    avg_c = sum(r.context_recall for r in results) / len(results)
    print(f"{'AVERAGE':<40} {avg_f:>6.2f} {avg_r:>6.2f} {avg_c:>7.2f}")
    print("="*60)
    print(f"\n✅ {len(results)} traces stored in eval_traces table\n")


if __name__ == "__main__":
    main()