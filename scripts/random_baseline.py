import argparse
import random
from pathlib import Path
from typing import Any

from llm_experiment_vllm import (
    DEFAULT_EXCLUDE_X_INPUT_PATH,
    DEFAULT_INPUT_PATH,
    append_results,
    extract_option_labels,
    load_records,
    now,
    structured_output_labels,
)


def make_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            'Run a uniformly random baseline for the negation-focus '
            'experiment.'
        ),
    )
    parser.add_argument(
        '--input',
        type=Path,
        default=None,
        help=(
            'Prompt JSON path. Defaults to experiment/prompt/amalgum_with_x.json, '
            'or experiment/prompt/amalgum_without_x.json with --exclude-x.'
        ),
    )
    parser.add_argument(
        '--exclude-x',
        action='store_true',
        help=(
            'Use experiment/prompt/amalgum_without_x.json as the default input '
            'and do not include x among the choices.'
        ),
    )
    parser.add_argument(
        '--output',
        type=Path,
        default=None,
        help=(
            'Prediction JSONL path. Defaults to '
            'experiment/vllm/random-baseline_seed<seed>_<input-stem>.jsonl.'
        ),
    )
    parser.add_argument(
        '--limit',
        type=int,
        default=None,
        help='Maximum number of records to process.',
    )
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument(
        '--overwrite',
        action='store_true',
        help='Replace an existing prediction file.',
    )
    args = parser.parse_args()
    if args.input is None:
        args.input = (
            DEFAULT_EXCLUDE_X_INPUT_PATH if args.exclude_x else DEFAULT_INPUT_PATH
        )
    return args


def baseline_model(seed: int) -> str:
    return f'random-baseline/seed-{seed}'


def default_output_path(seed: int, input_path: Path) -> Path:
    return (
        Path('experiment/vllm')
        / f'random-baseline_seed{seed}_{input_path.stem}.jsonl'
    )


def labels_by_record(
    records: list[dict[str, Any]],
    *,
    exclude_x: bool,
) -> dict[Any, tuple[str, ...]]:
    result: dict[Any, tuple[str, ...]] = {}
    for record in records:
        try:
            labels = structured_output_labels(
                extract_option_labels(record['prompt']),
                exclude_x=exclude_x,
            )
        except ValueError as error:
            raise SystemExit(f'id={record["id"]}: {error}') from error
        result[record['id']] = labels
    return result


def run(args: argparse.Namespace) -> None:
    output_path = args.output or default_output_path(args.seed, args.input)
    if args.overwrite and output_path.exists():
        output_path.unlink()

    records = load_records(args.input)
    if args.limit is not None:
        records = records[:args.limit]
    choices = labels_by_record(records, exclude_x=args.exclude_x)
    model = baseline_model(args.seed)

    rng = random.Random(args.seed)
    results: list[dict[str, Any]] = []
    for record in records:
        label = rng.choice(choices[record['id']])
        results.append(
            {
                'id': record['id'],
                'model': model,
                'label': label,
                'response': {'label': label},
                'raw_response': label,
                'finish_reason': 'random_choice',
                'prompt_tokens': 0,
                'generated_tokens': 0,
                'seed': args.seed,
                'created_at': now(),
            }
        )
    if results:
        append_results(output_path, results)
    print(
        f'Random baseline (seed={args.seed}): '
        f'wrote {len(results)} records to {output_path}'
    )


def validate_args(args: argparse.Namespace) -> None:
    if args.limit is not None and args.limit < 1:
        raise SystemExit('--limit must be at least 1.')


def main() -> None:
    args = make_args()
    validate_args(args)
    run(args)


if __name__ == '__main__':
    main()
