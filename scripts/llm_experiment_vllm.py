from __future__ import annotations

import argparse
import datetime
import importlib.util
import json
import os
import re
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

from experiment_paths import DEFAULT_PROMPT_WITHOUT_X, DEFAULT_PROMPT_WITH_X


DEFAULT_INPUT_PATH = DEFAULT_PROMPT_WITH_X
DEFAULT_EXCLUDE_X_INPUT_PATH = DEFAULT_PROMPT_WITHOUT_X
DEFAULT_MODEL = 'Qwen/Qwen3.5-0.8B'
OPTION_LABEL_PATTERN = re.compile(r'(?m)^([a-z])\.\s*(?:\S.*)?$')


def make_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description='Run the negation-focus experiment with vLLM offline inference.',
    )
    parser.add_argument('--model', default=DEFAULT_MODEL)
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
            'Use experiment/prompt/amalgum_without_x.json as the default input. '
            'The default is experiment/prompt/amalgum_with_x.json.'
        ),
    )
    parser.add_argument(
        '--output',
        type=Path,
        default=None,
        help=(
            'JSONL output path. Defaults to '
            'experiment/vllm/<model>_<input-stem>.jsonl.'
        ),
    )
    parser.add_argument(
        '--limit',
        type=int,
        default=None,
        help='Maximum number of unfinished records to process.',
    )
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--max-tokens', type=int, default=8)
    parser.add_argument('--temperature', type=float, default=0.0)
    parser.add_argument('--top-p', type=float, default=1.0)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--tensor-parallel-size', type=int, default=1)
    parser.add_argument('--max-model-len', type=int, default=8192)
    parser.add_argument('--gpu-memory-utilization', type=float, default=0.9)
    parser.add_argument('--dtype', default='auto')
    parser.add_argument('--trust-remote-code', action='store_true')
    parser.add_argument(
        '--chat-template',
        default=None,
        help='Optional chat template string or path accepted by vLLM.',
    )
    parser.add_argument(
        '--chat-template-kwargs-json',
        default='{}',
        help='JSON object passed to the model chat template.',
    )
    parser.add_argument(
        '--thinking-mode',
        choices=('on', 'off'),
        default=None,
        help=(
            'Enable or disable thinking mode through the chat template. '
            'When omitted, use the model/template default. This option '
            'overrides enable_thinking in --chat-template-kwargs-json.'
        ),
    )
    parser.add_argument(
        '--engine-args-json',
        default='{}',
        help='JSON object merged into the vLLM LLM constructor arguments.',
    )
    parser.add_argument(
        '--no-model-profile',
        action='store_true',
        help='Disable automatic Qwen/Ministral engine settings.',
    )
    parser.add_argument(
        '--overwrite',
        action='store_true',
        help='Replace an existing output instead of resuming it.',
    )
    parser.add_argument(
        '--show-progress',
        action='store_true',
        help='Show the vLLM progress bar for each batch.',
    )
    parser.add_argument(
        '--use-flashinfer-sampler',
        action='store_true',
        help=(
            'Enable FlashInfer top-k/top-p sampling. It is disabled by '
            'default to avoid FlashInfer/CUB JIT compatibility problems.'
        ),
    )
    parser.add_argument(
        '--wandb-project',
        default=None,
        help='Enable Weights & Biases logging under this project.',
    )
    parser.add_argument(
        '--wandb-entity',
        default=None,
        help='Optional Weights & Biases entity/team.',
    )
    parser.add_argument(
        '--wandb-run-name',
        default=None,
        help='Optional Weights & Biases run name.',
    )
    parser.add_argument(
        '--wandb-mode',
        default=None,
        choices=('online', 'offline', 'disabled'),
        help='Optional Weights & Biases mode.',
    )
    parser.add_argument(
        '--wandb-tags',
        default='',
        help='Comma-separated Weights & Biases tags.',
    )
    args = parser.parse_args()
    if args.input is None:
        if args.exclude_x:
            args.input = DEFAULT_EXCLUDE_X_INPUT_PATH
        else:
            args.input = DEFAULT_INPUT_PATH
    return args


def now() -> str:
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


def model_slug(model: str) -> str:
    slug = re.sub(r'[^A-Za-z0-9._-]+', '_', model).strip('._')
    return slug or 'model'


def default_output_path(model: str, input_path: Path) -> Path:
    return Path('experiment/vllm') / f'{model_slug(model)}_{input_path.stem}.jsonl'


def parse_json_object(value: str, option_name: str) -> dict[str, Any]:
    try:
        parsed = json.loads(value)
    except json.JSONDecodeError as error:
        raise SystemExit(f'{option_name} is not valid JSON: {error}') from error
    if not isinstance(parsed, dict):
        raise SystemExit(f'{option_name} must be a JSON object.')
    return parsed


def load_records(path: Path) -> list[dict[str, Any]]:
    with path.open('r', encoding='utf-8-sig') as file:
        records = json.load(file)
    if not isinstance(records, list):
        raise SystemExit(f'{path} must contain a JSON array.')

    seen_ids: set[Any] = set()
    for index, record in enumerate(records):
        if not isinstance(record, dict):
            raise SystemExit(f'{path}: item {index} must be a JSON object.')
        if 'id' not in record or 'prompt' not in record:
            raise SystemExit(f'{path}: item {index} must contain id and prompt.')
        if record['id'] in seen_ids:
            raise SystemExit(f'{path}: duplicate id {record["id"]!r}.')
        if not isinstance(record['prompt'], str):
            raise SystemExit(f'{path}: prompt for id={record["id"]!r} must be text.')
        seen_ids.add(record['id'])
    return records


def extract_option_labels(prompt: str) -> list[str]:
    options_marker = '\nOptions:\n'
    if options_marker not in prompt:
        raise ValueError('Prompt does not contain an Options section.')
    options_text = prompt.rsplit(options_marker, maxsplit=1)[1]
    answer_marker = '\n\nAnswer:'
    if answer_marker in options_text:
        options_text = options_text.split(answer_marker, maxsplit=1)[0]
    labels = OPTION_LABEL_PATTERN.findall(options_text)
    if not labels:
        raise ValueError('Prompt does not contain option labels.')
    if len(labels) != len(set(labels)):
        raise ValueError(f'Prompt contains duplicate labels: {labels}')
    return labels


def structured_output_labels(labels: list[str], *, exclude_x: bool) -> tuple[str, ...]:
    if exclude_x or 'x' in labels:
        return tuple(labels)
    return tuple([*labels, 'x'])


def load_completed_ids(path: Path, model: str) -> set[Any]:
    if not path.exists():
        return set()

    completed_ids: set[Any] = set()
    with path.open('r', encoding='utf-8-sig') as file:
        for line_number, line in enumerate(file, start=1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as error:
                raise SystemExit(
                    f'{path}:{line_number} is not valid JSON: {error}'
                ) from error
            output_model = record.get('model')
            if output_model != model:
                raise SystemExit(
                    f'{path}:{line_number} belongs to model {output_model!r}, '
                    f'not {model!r}. Use another --output or --overwrite.'
                )
            completed_ids.add(record['id'])
    return completed_ids


def append_results(path: Path, results: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('a', encoding='utf-8', newline='\n') as file:
        for result in results:
            file.write(json.dumps(result, ensure_ascii=False) + '\n')
        file.flush()
        os.fsync(file.fileno())


def model_profile(model: str) -> dict[str, Any]:
    normalized = model.lower()
    if normalized.startswith('mistralai/ministral-3-'):
        return {
            'tokenizer_mode': 'mistral',
            'config_format': 'mistral',
            'load_format': 'mistral',
             'limit_mm_per_prompt': {'image': 0},
        }
    if normalized.startswith('qwen/qwen3.5-'):
        return {'language_model_only': True}
    if normalized.startswith('qwen/qwen3.6-'):
        return {'max_num_seqs': 256}
    return {}


def make_engine_args(
    args: argparse.Namespace,
    extra_engine_args: dict[str, Any],
) -> dict[str, Any]:
    engine_args: dict[str, Any] = {
        'model': args.model,
        'tensor_parallel_size': args.tensor_parallel_size,
        'max_model_len': args.max_model_len,
        'gpu_memory_utilization': args.gpu_memory_utilization,
        'dtype': args.dtype,
        'trust_remote_code': args.trust_remote_code,
        'seed': args.seed,
    }
    if not args.no_model_profile:
        engine_args.update(model_profile(args.model))
    engine_args.update(extra_engine_args)
    return engine_args


def chunks(items: list[dict[str, Any]], size: int) -> list[list[dict[str, Any]]]:
    return [items[start:start + size] for start in range(0, len(items), size)]


def parse_wandb_tags(tags: str) -> list[str]:
    return [tag.strip() for tag in tags.split(',') if tag.strip()]


def init_wandb(
    args: argparse.Namespace,
    output_path: Path,
    records_count: int,
    completed_count: int,
    pending_count: int,
    engine_args: dict[str, Any],
) -> Any | None:
    project = args.wandb_project or os.environ.get('WANDB_PROJECT')
    if not project:
        return None
    if importlib.util.find_spec('wandb') is None:
        raise SystemExit(
            'Weights & Biases logging was requested, but wandb is not '
            f'installed in {sys.executable}. Install wandb in the same '
            'Python environment, or omit --wandb-project.'
        )

    import wandb

    config = {
        'model': args.model,
        'thinking_mode': args.thinking_mode,
        'input': str(args.input),
        'output': str(output_path),
        'records_count': records_count,
        'completed_at_start': completed_count,
        'pending_at_start': pending_count,
        'limit': args.limit,
        'batch_size': args.batch_size,
        'max_tokens': args.max_tokens,
        'temperature': args.temperature,
        'top_p': args.top_p,
        'seed': args.seed,
        'engine_args': engine_args,
    }
    return wandb.init(
        project=project,
        entity=args.wandb_entity,
        name=args.wandb_run_name,
        mode=args.wandb_mode,
        tags=parse_wandb_tags(args.wandb_tags),
        config=config,
    )


def run(args: argparse.Namespace) -> None:
    os.environ['VLLM_USE_FLASHINFER_SAMPLER'] = (
        '1' if args.use_flashinfer_sampler else '0'
    )

    if importlib.util.find_spec('vllm') is None:
        raise SystemExit(
            'vLLM is not installed in the Python interpreter running this '
            f'script: {sys.executable}\n'
            'Run the script with `uv run --no-sync python ...`, or compare '
            '`uv run --no-sync python -c "import sys; print(sys.executable)"` '
            'with '
            '`python -c "import sys; print(sys.executable)"`.'
        )

    try:
        from vllm import LLM, SamplingParams
        from vllm.sampling_params import StructuredOutputsParams
    except ImportError as error:
        raise SystemExit(
            'vLLM is installed, but importing it or one of its dependencies '
            f'failed in {sys.executable}.\n'
            f'Original error: {type(error).__name__}: {error}\n'
            'Run `uv run --no-sync python -c "from vllm import LLM, '
            'SamplingParams; '
            'from vllm.sampling_params import StructuredOutputsParams; '
            'print(\\\"vLLM imports OK\\\")"` to obtain the full traceback.'
        ) from error

    output_path = args.output or default_output_path(args.model, args.input)
    if args.overwrite and output_path.exists():
        output_path.unlink()

    records = load_records(args.input)
    completed_ids = load_completed_ids(output_path, args.model)
    pending = [record for record in records if record['id'] not in completed_ids]
    if args.limit is not None:
        pending = pending[:args.limit]
    if not pending:
        print(f'No unfinished records: completed={len(completed_ids)}')
        return

    extra_engine_args = parse_json_object(
        args.engine_args_json,
        '--engine-args-json',
    )
    chat_template_kwargs = parse_json_object(
        args.chat_template_kwargs_json,
        '--chat-template-kwargs-json',
    )
    if args.thinking_mode is not None:
        chat_template_kwargs['enable_thinking'] = args.thinking_mode == 'on'
    engine_args = make_engine_args(args, extra_engine_args)
    wandb_run = init_wandb(
        args,
        output_path,
        len(records),
        len(completed_ids),
        len(pending),
        engine_args,
    )
    processed = 0
    try:
        print(f'Loading model {args.model!r} with engine args: {engine_args}')
        llm = LLM(**engine_args)

        for batch_index, batch in enumerate(chunks(pending, args.batch_size), start=1):
            batch_started = time.perf_counter()
            groups: dict[tuple[str, ...], list[dict[str, Any]]] = defaultdict(list)
            for record in batch:
                try:
                    labels = structured_output_labels(
                        extract_option_labels(record['prompt']),
                        exclude_x=args.exclude_x,
                    )
                except ValueError as error:
                    raise SystemExit(f'id={record["id"]}: {error}') from error
                groups[labels].append(record)

            batch_results: list[dict[str, Any]] = []
            for labels, group in groups.items():
                structured_outputs = StructuredOutputsParams(choice=list(labels))
                sampling_params = SamplingParams(
                    temperature=args.temperature,
                    top_p=args.top_p,
                    max_tokens=args.max_tokens,
                    seed=args.seed,
                    structured_outputs=structured_outputs,
                )
                conversations = [
                    [{'role': 'user', 'content': record['prompt']}]
                    for record in group
                ]
                outputs = llm.chat(
                    conversations,
                    sampling_params=sampling_params,
                    use_tqdm=args.show_progress,
                    chat_template=args.chat_template,
                    chat_template_kwargs=chat_template_kwargs,
                )
                if len(outputs) != len(group):
                    raise RuntimeError(
                        f'vLLM returned {len(outputs)} outputs for {len(group)} inputs.'
                    )

                for record, output in zip(group, outputs):
                    candidate = output.outputs[0]
                    raw_response = candidate.text.strip()
                    if raw_response not in labels:
                        raise RuntimeError(
                            f'id={record["id"]}: invalid label {raw_response!r}; '
                            f'expected one of {list(labels)}.'
                        )
                    batch_results.append(
                        {
                            'id': record['id'],
                            'model': args.model,
                            'label': raw_response,
                            'response': {'label': raw_response},
                            'raw_response': candidate.text,
                            'finish_reason': candidate.finish_reason,
                            'prompt_tokens': len(output.prompt_token_ids),
                            'generated_tokens': len(candidate.token_ids),
                            'created_at': now(),
                        }
                    )

            order = {record['id']: index for index, record in enumerate(batch)}
            batch_results.sort(key=lambda result: order[result['id']])
            append_results(output_path, batch_results)
            processed += len(batch_results)
            completed = len(completed_ids) + processed
            remaining = len(records) - completed
            print(
                f'[{completed}/{len(records)}] wrote {len(batch_results)} records '
                f'to {output_path}'
            )

            if wandb_run is not None:
                label_counts: dict[str, int] = defaultdict(int)
                finish_reason_counts: dict[str, int] = defaultdict(int)
                prompt_tokens = 0
                generated_tokens = 0
                for result in batch_results:
                    label_counts[result['label']] += 1
                    finish_reason_counts[str(result['finish_reason'])] += 1
                    prompt_tokens += result['prompt_tokens']
                    generated_tokens += result['generated_tokens']

                wandb_run.log(
                    {
                        'batch/index': batch_index,
                        'batch/records': len(batch_results),
                        'batch/seconds': time.perf_counter() - batch_started,
                        'progress/processed': processed,
                        'progress/completed': completed,
                        'progress/remaining': remaining,
                        'tokens/prompt': prompt_tokens,
                        'tokens/generated': generated_tokens,
                        'tokens/total': prompt_tokens + generated_tokens,
                        **{
                            f'labels/{label}': count
                            for label, count in label_counts.items()
                        },
                        **{
                            f'finish_reasons/{reason}': count
                            for reason, count in finish_reason_counts.items()
                        },
                    },
                    step=completed,
                )

        remaining = len(records) - len(completed_ids) - processed
        print(
            f'Finished: processed={processed}, '
            f'completed={len(completed_ids) + processed}, remaining={remaining}'
        )
        if wandb_run is not None:
            wandb_run.summary['processed'] = processed
            wandb_run.summary['completed'] = len(completed_ids) + processed
            wandb_run.summary['remaining'] = remaining
            wandb_run.summary['output'] = str(output_path)
    finally:
        if wandb_run is not None:
            wandb_run.finish()


def validate_args(args: argparse.Namespace) -> None:
    if args.limit is not None and args.limit < 1:
        raise SystemExit('--limit must be at least 1.')
    if args.batch_size < 1:
        raise SystemExit('--batch-size must be at least 1.')
    if args.max_tokens < 1:
        raise SystemExit('--max-tokens must be at least 1.')
    if args.tensor_parallel_size < 1:
        raise SystemExit('--tensor-parallel-size must be at least 1.')
    if args.max_model_len < 1:
        raise SystemExit('--max-model-len must be at least 1.')
    if not 0 < args.gpu_memory_utilization <= 1:
        raise SystemExit('--gpu-memory-utilization must be in (0, 1].')
    if args.temperature < 0:
        raise SystemExit('--temperature must not be negative.')
    if not 0 < args.top_p <= 1:
        raise SystemExit('--top-p must be in (0, 1].')


def main() -> None:
    args = make_args()
    validate_args(args)
    run(args)


if __name__ == '__main__':
    main()
