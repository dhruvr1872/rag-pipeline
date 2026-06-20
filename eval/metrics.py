"""RAGAS evaluation metrics: faithfulness, answer relevancy, context recall/precision."""
from __future__ import annotations
from dataclasses import dataclass

from datasets import Dataset
from ragas import evaluate
from ragas.metrics import (
    faithfulness,
    answer_relevancy,
    context_recall,
    context_precision,
)


@dataclass
class EvalResult:
    faithfulness: float
    answer_relevancy: float
    context_recall: float
    context_precision: float

    def __str__(self) -> str:
        return (
            f"Faithfulness:      {self.faithfulness:.3f}\n"
            f"Answer Relevancy:  {self.answer_relevancy:.3f}\n"
            f"Context Recall:    {self.context_recall:.3f}\n"
            f"Context Precision: {self.context_precision:.3f}"
        )


def run_eval(
    questions: list[str],
    answers: list[str],
    contexts: list[list[str]],
    ground_truths: list[str],
) -> EvalResult:
    dataset = Dataset.from_dict({
        "question": questions,
        "answer": answers,
        "contexts": contexts,
        "ground_truth": ground_truths,
    })
    result = evaluate(
        dataset,
        metrics=[faithfulness, answer_relevancy, context_recall, context_precision],
    )
    return EvalResult(
        faithfulness=float(result["faithfulness"]),
        answer_relevancy=float(result["answer_relevancy"]),
        context_recall=float(result["context_recall"]),
        context_precision=float(result["context_precision"]),
    )
