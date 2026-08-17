"""Small, deterministic regression harness for physics-claim calibration.

The harness scores saved agent answers for required concepts and forbidden overclaims.
It is intentionally model-provider agnostic, so it can compare prompts or models using
the same answer JSON without making an API call itself.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path


DEFAULT_CASES = Path(__file__).resolve().parent / "evals" / "physics_analysis_cases.json"


@dataclass(frozen=True)
class PhysicsEvalResult:
    case_id: str
    passed: bool
    concept_score: float
    missing_concepts: list[list[str]]
    forbidden_hits: list[str]


def evaluate_response(case: dict, response: str) -> PhysicsEvalResult:
    """Score concepts as synonym groups and fail any explicit scientific overclaim."""
    normalized = response.casefold()
    concept_groups = case.get("required_concepts", [])
    missing = [
        group
        for group in concept_groups
        if not any(term.casefold() in normalized for term in group)
    ]
    forbidden_hits = [
        phrase
        for phrase in case.get("forbidden_claims", [])
        if phrase.casefold() in normalized
    ]
    score = 1.0 if not concept_groups else (len(concept_groups) - len(missing)) / len(concept_groups)
    return PhysicsEvalResult(
        case_id=case["id"],
        passed=not missing and not forbidden_hits,
        concept_score=round(score, 6),
        missing_concepts=missing,
        forbidden_hits=forbidden_hits,
    )


def evaluate_answer_file(cases_path: Path, answers_path: Path) -> dict:
    cases = json.loads(cases_path.read_text(encoding="utf-8"))
    answers = json.loads(answers_path.read_text(encoding="utf-8"))
    results = [evaluate_response(case, answers.get(case["id"], "")) for case in cases]
    passed = sum(result.passed for result in results)
    return {
        "passed": passed,
        "total": len(results),
        "pass_rate": round(passed / len(results), 6) if results else None,
        "results": [asdict(result) for result in results],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("answers", type=Path, help="JSON object mapping case id to agent answer")
    parser.add_argument("--cases", type=Path, default=DEFAULT_CASES)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    report = evaluate_answer_file(args.cases, args.answers)
    payload = json.dumps(report, ensure_ascii=False, indent=2) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(payload, encoding="utf-8")
    print(payload, end="")


if __name__ == "__main__":
    main()
