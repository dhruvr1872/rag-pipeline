"""Run RAGAS evaluation against a sample Q&A dataset."""
import json
import logging
from pathlib import Path

from src.pipeline import RAGPipeline
from eval.metrics import run_eval

logging.basicConfig(level=logging.INFO)


def main(dataset_path: str = "eval/sample_dataset.json") -> None:
    data = json.loads(Path(dataset_path).read_text())
    pipeline = RAGPipeline()
    sources = list({row["source"] for row in data})
    pipeline.index(sources)

    questions, answers, contexts, ground_truths = [], [], [], []
    for row in data:
        response = pipeline.query(row["question"])
        questions.append(row["question"])
        answers.append(response.answer)
        contexts.append(response.sources)
        ground_truths.append(row["ground_truth"])

    result = run_eval(questions, answers, contexts, ground_truths)
    print("\n=== Evaluation Results ===")
    print(result)


if __name__ == "__main__":
    main()
