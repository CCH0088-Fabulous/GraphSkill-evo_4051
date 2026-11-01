"""Answer-only EM/F1 with benchmark-specific HotpotQA categorical handling.

SearchQA follows upstream GraphSkillEvo's SQuAD-style metrics. HotpotQA follows
https://github.com/hotpotqa/hotpot/blob/master/hotpot_evaluate_v1.py for answer
normalization and yes/no/noanswer mismatches. Supporting-fact and joint scores
are deliberately not presented as answer-only EM/F1.
"""

from __future__ import annotations

import re
import string
from collections import Counter
from collections.abc import Sequence


def normalize_answer(text: str) -> str:
    lowered = text.lower()
    without_punctuation = "".join(character for character in lowered if character not in string.punctuation)
    without_articles = re.sub(r"\b(a|an|the)\b", " ", without_punctuation)
    return " ".join(without_articles.split())


def extract_answer(text: str) -> str:
    """Match upstream SearchQA: last XML answer, otherwise last non-empty line."""
    matches = re.findall(r"<answer>(.*?)</answer>", text, flags=re.DOTALL | re.IGNORECASE)
    if matches:
        return matches[-1].strip()
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    return lines[-1] if lines else ""


def _validate_answers(answers: Sequence[str]) -> None:
    if isinstance(answers, str) or not answers or not all(isinstance(answer, str) for answer in answers):
        raise ValueError("Reference answers must be a non-empty sequence of strings")


def exact_match(prediction: str, answers: Sequence[str]) -> float:
    _validate_answers(answers)
    normalized = normalize_answer(prediction)
    return float(any(normalized == normalize_answer(answer) for answer in answers))


def f1_score(prediction: str, answers: Sequence[str], *, dataset: str = "searchqa") -> float:
    _validate_answers(answers)
    if dataset not in {"searchqa", "hotpotqa"}:
        raise ValueError(f"Unsupported dataset {dataset!r}")
    prediction_normalized = normalize_answer(prediction)
    prediction_tokens = prediction_normalized.split()
    scores = []
    categorical = {"yes", "no", "noanswer"}
    for answer in answers:
        reference_normalized = normalize_answer(answer)
        if dataset == "hotpotqa" and prediction_normalized != reference_normalized:
            if prediction_normalized in categorical or reference_normalized in categorical:
                scores.append(0.0)
                continue
        reference_tokens = reference_normalized.split()
        if not prediction_tokens or not reference_tokens:
            # Official HotpotQA returns zero for empty overlap, including two empty strings.
            scores.append(float(prediction_tokens == reference_tokens) if dataset == "searchqa" else 0.0)
            continue
        overlap = sum((Counter(prediction_tokens) & Counter(reference_tokens)).values())
        if not overlap:
            scores.append(0.0)
            continue
        precision = overlap / len(prediction_tokens)
        recall = overlap / len(reference_tokens)
        scores.append(2 * precision * recall / (precision + recall))
    return max(scores)


def evaluate(raw_output: str, answers: Sequence[str], *, dataset: str = "searchqa") -> dict[str, float | str]:
    answer = extract_answer(raw_output)
    return {
        "answer": answer,
        "em": exact_match(answer, answers),
        "f1": f1_score(answer, answers, dataset=dataset),
    }
