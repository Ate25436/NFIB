"""Calculate per-model accuracy for the local vLLM experiment results."""

from __future__ import annotations

import argparse
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable


DEFAULT_GOLD_PATH = Path("dataset/doccano/output/NF_merged_majority_vote.jsonl")
DEFAULT_RESULTS_DIR = Path("experiment/vllm")
DEFAULT_BATCH_RESULTS_DIR = Path("experiment/batch/results")
TIE_LABEL = "TIE"
NONE_OF_THE_ABOVE_LABEL = "x"


@dataclass(frozen=True)
class EvaluationResult:
    model: str
    backend: str
    correct: int
    total: int
    missing: int
    source: Path

    @property
    def accuracy(self) -> float:
        return self.correct / self.total


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8-sig") as file:
        for line_number, line in enumerate(file, start=1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(
                    f"Invalid JSON in {path} at line {line_number}: {error}"
                ) from error
            if not isinstance(record, dict):
                raise ValueError(f"{path}:{line_number} must contain a JSON object")
            records.append(record)
    return records


def require_id(record: dict[str, Any], path: Path) -> int:
    record_id = record.get("id")
    if not isinstance(record_id, int):
        raise ValueError(f"{path}: record has an invalid id: {record_id!r}")
    return record_id


def gold_label(record: dict[str, Any], path: Path) -> str:
    labels = record.get("label")
    record_id = record.get("id", "<missing>")
    if not isinstance(labels, list) or len(labels) != 1 or not isinstance(labels[0], str):
        raise ValueError(f"{path}: id={record_id} must have exactly one string label")
    return labels[0]


def prediction_label(record: dict[str, Any], path: Path) -> str:
    label = record.get("label")
    record_id = record.get("id", "<missing>")
    if not isinstance(label, str) or not label:
        raise ValueError(f"{path}: id={record_id} has an invalid prediction label: {label!r}")
    return label


def parse_label_text(text: Any, path: Path, record_id: int) -> str:
    """Extract a label from either a JSON response or a plain label string."""
    if not isinstance(text, str) or not text.strip():
        raise ValueError(f"{path}: id={record_id} has no textual response")

    stripped_text = text.strip()
    try:
        parsed = json.loads(stripped_text)
    except json.JSONDecodeError:
        parsed = stripped_text

    label = parsed.get("label") if isinstance(parsed, dict) else parsed
    if not isinstance(label, str) or not label:
        raise ValueError(f"{path}: id={record_id} has an invalid response label: {label!r}")
    return label


def batch_record_id(record: dict[str, Any], path: Path) -> int:
    external_id = record.get("custom_id", record.get("key"))
    if not isinstance(external_id, str):
        raise ValueError(f"{path}: batch record has no custom_id or key")
    match = re.search(r"_(\d+)$", external_id)
    if match is None:
        raise ValueError(f"{path}: cannot extract an integer id from {external_id!r}")
    return int(match.group(1))


def parse_openai_batch_record(
    record: dict[str, Any], path: Path, record_id: int
) -> tuple[str, str | None] | None:
    response = record.get("response")
    body = response.get("body") if isinstance(response, dict) else None
    choices = body.get("choices") if isinstance(body, dict) else None
    if not isinstance(body, dict):
        return None
    model = body.get("model")
    if not isinstance(model, str) or not model:
        raise ValueError(f"{path}: id={record_id} has no OpenAI model name")
    if not isinstance(choices, list) or not choices:
        return model, None
    message = choices[0].get("message") if isinstance(choices[0], dict) else None
    content = message.get("content") if isinstance(message, dict) else None
    if content is None:
        return model, None
    return model, parse_label_text(content, path, record_id)


def parse_anthropic_batch_record(
    record: dict[str, Any], path: Path, record_id: int
) -> tuple[str, str | None] | None:
    result = record.get("result")
    message = result.get("message") if isinstance(result, dict) else None
    if not isinstance(message, dict):
        return None
    content = message.get("content")
    text_blocks = [
        block.get("text")
        for block in content
        if isinstance(block, dict) and block.get("type") == "text"
    ] if isinstance(content, list) else []
    model = message.get("model")
    if not isinstance(model, str) or not model:
        raise ValueError(f"{path}: id={record_id} has no Anthropic model name")
    if not text_blocks:
        return model, None
    if len(text_blocks) != 1:
        raise ValueError(f"{path}: id={record_id} must have exactly one text block")
    return model, parse_label_text(text_blocks[0], path, record_id)


def parse_gemini_batch_record(
    record: dict[str, Any], path: Path, record_id: int
) -> tuple[str, str | None] | None:
    response = record.get("response")
    if not isinstance(response, dict) or "modelVersion" not in response:
        return None
    candidates = response.get("candidates") if isinstance(response, dict) else None
    model = response.get("modelVersion")
    if not isinstance(model, str) or not model:
        raise ValueError(f"{path}: id={record_id} has no Gemini model name")
    if not isinstance(candidates, list) or not candidates:
        return model, None
    content = candidates[0].get("content") if isinstance(candidates[0], dict) else None
    parts = content.get("parts") if isinstance(content, dict) else None
    text_parts = [part.get("text") for part in parts if isinstance(part, dict) and "text" in part] \
        if isinstance(parts, list) else []
    if not text_parts:
        return model, None
    if len(text_parts) != 1:
        raise ValueError(f"{path}: id={record_id} must have exactly one text part")
    return model, parse_label_text(text_parts[0], path, record_id)


def parse_batch_record(record: dict[str, Any], path: Path) -> tuple[int, str, str | None]:
    record_id = batch_record_id(record, path)
    parsers = (
        parse_openai_batch_record,
        parse_anthropic_batch_record,
        parse_gemini_batch_record,
    )
    for parser in parsers:
        parsed = parser(record, path, record_id)
        if parsed is not None:
            model, label = parsed
            return record_id, model, label
    raise ValueError(f"{path}: id={record_id} has an unsupported batch result format")


def index_records(records: Iterable[dict[str, Any]], path: Path) -> dict[int, dict[str, Any]]:
    records_by_id: dict[int, dict[str, Any]] = {}
    for record in records:
        record_id = require_id(record, path)
        if record_id in records_by_id:
            raise ValueError(f"{path}: duplicate id={record_id}")
        records_by_id[record_id] = record
    return records_by_id


def load_gold_labels(path: Path, exclude_x: bool) -> dict[int, str]:
    gold_by_id = index_records(load_jsonl(path), path)
    excluded_labels = {TIE_LABEL}
    if exclude_x:
        excluded_labels.add(NONE_OF_THE_ABOVE_LABEL)

    return {
        record_id: label
        for record_id, record in gold_by_id.items()
        if (label := gold_label(record, path)) not in excluded_labels
    }


def is_result_for_mode(
    path: Path,
    exclude_x: bool,
    result_fragment: str | None = None,
) -> bool:
    """Accept both ``*_with_x`` and ``*-with-x`` filename conventions."""
    normalized_stem = path.stem.lower().replace("-", "_")
    expected_mode = "_without_x" if exclude_x else "_with_x"
    if result_fragment is None:
        return normalized_stem.endswith(expected_mode)

    normalized_fragment = result_fragment.lower().replace("-", "_")
    return expected_mode in normalized_stem and normalized_fragment in normalized_stem


def find_result_files(
    results_dir: Path,
    exclude_x: bool,
    result_fragment: str | None = None,
) -> list[Path]:
    result_files = sorted(
        path
        for path in results_dir.glob("*.jsonl")
        if is_result_for_mode(path, exclude_x, result_fragment)
    )
    if not result_files:
        mode = "without_x" if exclude_x else "with_x"
        fragment_message = (
            f" matching fragment {result_fragment!r}" if result_fragment is not None else ""
        )
        raise ValueError(
            f"No {mode} JSONL files{fragment_message} found in {results_dir}"
        )
    return result_files


def find_batch_result_groups(
    results_dir: Path,
    exclude_x: bool,
    result_fragment: str | None = None,
) -> list[tuple[str, list[Path]]]:
    expected_fragment = result_fragment or (
        "_without_x_batch_output_" if exclude_x else "_with_x_batch_output_"
    )
    normalized_fragment = expected_fragment.lower().replace("-", "_")
    files = sorted(
        path
        for path in results_dir.rglob("*.jsonl")
        if normalized_fragment in path.stem.lower().replace("-", "_")
    )
    groups: dict[str, list[Path]] = {}
    for path in files:
        relative_parent = str(path.parent.relative_to(results_dir))
        groups.setdefault(relative_parent, []).append(path)
    return sorted(groups.items())


def model_name(records: list[dict[str, Any]], path: Path) -> str:
    models = {record.get("model") for record in records}
    if len(models) != 1 or not all(isinstance(model, str) and model for model in models):
        raise ValueError(f"{path}: records must contain one consistent model name")
    return models.pop()


def evaluate_predictions(
    *,
    model: str,
    backend: str,
    predictions_by_id: dict[int, str],
    gold_labels: dict[int, str],
    source: Path,
) -> EvaluationResult:
    evaluated_ids = gold_labels.keys() & predictions_by_id.keys()
    if not evaluated_ids:
        raise ValueError(f"{source}: no prediction ids match the gold records")
    correct = sum(
        predictions_by_id[record_id] == gold_labels[record_id]
        for record_id in evaluated_ids
    )
    return EvaluationResult(
        model=model,
        backend=backend,
        correct=correct,
        total=len(evaluated_ids),
        missing=len(gold_labels) - len(evaluated_ids),
        source=source,
    )


def evaluate_vllm(path: Path, gold_labels: dict[int, str]) -> EvaluationResult:
    prediction_records = load_jsonl(path)
    records_by_id = index_records(prediction_records, path)
    predictions_by_id = {
        record_id: prediction_label(record, path)
        for record_id, record in records_by_id.items()
    }
    return evaluate_predictions(
        model=model_name(prediction_records, path),
        backend="vLLM",
        predictions_by_id=predictions_by_id,
        gold_labels=gold_labels,
        source=path,
    )


def evaluate_batch_group(
    group_name: str, paths: list[Path], gold_labels: dict[int, str]
) -> EvaluationResult:
    predictions_by_id: dict[int, str] = {}
    seen_ids: set[int] = set()
    models: set[str] = set()
    for path in paths:
        for record in load_jsonl(path):
            record_id, model, label = parse_batch_record(record, path)
            if record_id in seen_ids:
                raise ValueError(f"{path}: duplicate batch prediction for id={record_id}")
            seen_ids.add(record_id)
            models.add(model)
            if label is not None:
                predictions_by_id[record_id] = label
    if len(models) != 1:
        raise ValueError(f"Batch group {group_name!r} contains multiple models: {sorted(models)}")
    return evaluate_predictions(
        model=models.pop(),
        backend="Batch API",
        predictions_by_id=predictions_by_id,
        gold_labels=gold_labels,
        source=paths[0].parent,
    )


def print_results(
    results: list[EvaluationResult],
    exclude_x: bool,
    eligible_gold_count: int,
    vllm_result_fragment: str | None = None,
    batch_result_fragment: str | None = None,
) -> None:
    mode = "without_x" if exclude_x else "with_x"
    print(f"mode: {mode}  eligible gold records: {eligible_gold_count}")
    if vllm_result_fragment is not None:
        print(f"vLLM result fragment: {vllm_result_fragment}")
    if batch_result_fragment is not None:
        print(f"Batch result fragment: {batch_result_fragment}")
    print(f"{'backend':<10} {'model':<44} {'accuracy':>10} {'correct/evaluated':>18} {'missing':>8}")
    print(f"{'-' * 10} {'-' * 44} {'-' * 10} {'-' * 18} {'-' * 8}")
    for result in sorted(results, key=lambda item: (item.backend, item.model.lower())):
        print(
            f"{result.backend:<10} "
            f"{result.model:<44} "
            f"{result.accuracy:>9.2%} "
            f"{result.correct:>9}/{result.total:<8} "
            f"{result.missing:>8}"
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare vLLM predictions with the doccano majority-vote labels."
    )
    parser.add_argument(
        "--gold-path",
        type=Path,
        default=DEFAULT_GOLD_PATH,
        help=f"Gold-label JSONL path (default: {DEFAULT_GOLD_PATH}).",
    )
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=DEFAULT_RESULTS_DIR,
        help=f"Directory containing vLLM JSONL files (default: {DEFAULT_RESULTS_DIR}).",
    )
    parser.add_argument(
        "--vllm-result-fragment",
        help=(
            "Only include vLLM result filenames containing this fragment. "
            "The with_x/without_x mode must also match. Matching is "
            "case-insensitive and treats hyphens as underscores."
        ),
    )
    parser.add_argument(
        "--batch-results-dir",
        type=Path,
        default=DEFAULT_BATCH_RESULTS_DIR,
        help=f"Directory containing Batch API results (default: {DEFAULT_BATCH_RESULTS_DIR}).",
    )
    parser.add_argument(
        "--batch-result-fragment",
        help=(
            "Only include Batch API result filenames containing this fragment. "
            "Matching is case-insensitive and treats hyphens as underscores. "
            "By default, the fragment is selected from --exclude-x."
        ),
    )
    parser.add_argument(
        "--exclude-x",
        action="store_true",
        help="Use without_x results and exclude gold label x in addition to TIE.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    gold_labels = load_gold_labels(args.gold_path, args.exclude_x)
    if not gold_labels:
        raise ValueError("No gold records remain after filtering")

    result_files = find_result_files(
        args.results_dir,
        args.exclude_x,
        args.vllm_result_fragment,
    )
    results = [evaluate_vllm(path, gold_labels) for path in result_files]
    batch_groups = find_batch_result_groups(
        args.batch_results_dir,
        args.exclude_x,
        args.batch_result_fragment,
    )
    results.extend(
        evaluate_batch_group(group_name, paths, gold_labels)
        for group_name, paths in batch_groups
    )
    print_results(
        results,
        args.exclude_x,
        len(gold_labels),
        args.vllm_result_fragment,
        args.batch_result_fragment,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
