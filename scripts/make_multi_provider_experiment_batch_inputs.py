import argparse
import json
import re
from pathlib import Path
from typing import Callable, Optional

from experiment_paths import DEFAULT_PROMPT_PATH, DEFAULT_PROMPT_WITH_X


OPTION_LABEL_PATTERN = re.compile(r'(?m)^([a-z])\.\s*$')
DEFAULT_MAX_BATCH_TOKENS = 1_000_000
INITIAL_BATCH_STATE = {
    'models': {},
    'batches': [],
}

PROVIDERS = {
    'openai': {
        'model': 'gpt-5.5',
        'extension': 'jsonl',
    },
    'anthropic': {
        'model': 'claude-opus-4-8',
        'extension': 'json',
    },
    'gemini': {
        'model': 'gemini-3.5-flash',
        'extension': 'jsonl',
    },
}


def initialize_batch_state(output_root: Path) -> Path:
    output_root.mkdir(parents=True, exist_ok=True)
    batch_state_path = output_root / 'batch_state.json'
    if batch_state_path.exists():
        print(f'Kept existing batch state: {batch_state_path}')
        return batch_state_path

    with batch_state_path.open('x', encoding='utf-8', newline='\n') as f:
        json.dump(INITIAL_BATCH_STATE, f, ensure_ascii=False, indent=2)
        f.write('\n')
    print(f'Wrote initial batch state: {batch_state_path}')
    return batch_state_path


def extract_option_labels(prompt: str) -> list[str]:
    options_marker = '\nOptions:\n'
    if options_marker not in prompt:
        raise ValueError('Prompt does not contain an Options section.')

    options_text = prompt.rsplit(options_marker, maxsplit=1)[1]
    answer_marker = '\n\nAnswer:\n'
    if answer_marker in options_text:
        options_text = options_text.split(answer_marker, maxsplit=1)[0]
    labels = OPTION_LABEL_PATTERN.findall(options_text)
    if not labels:
        raise ValueError('Prompt does not contain any option labels.')
    if len(labels) != len(set(labels)):
        raise ValueError(f'Prompt contains duplicate option labels: {labels}')
    return labels


def make_token_counter(model: str) -> Callable[[dict], int]:
    try:
        import tiktoken
    except ImportError:
        print(
            'Warning: tiktoken is not installed. '
            'Using a conservative UTF-8 byte estimate.'
        )

        def estimate_tokens(request: dict) -> int:
            payload = json.dumps(request, ensure_ascii=False, sort_keys=True)
            return max(1, (len(payload.encode('utf-8')) + 2) // 3)

        return estimate_tokens

    try:
        encoding = tiktoken.encoding_for_model(model)
    except KeyError:
        encoding = tiktoken.get_encoding('cl100k_base')

    def count_tokens(request: dict) -> int:
        payload = json.dumps(request, ensure_ascii=False, sort_keys=True)
        return len(encoding.encode(payload))

    return count_tokens


def parse_bool(value: str) -> bool:
    normalized = value.lower()
    if normalized in {'1', 'true', 'yes', 'y', 'on'}:
        return True
    if normalized in {'0', 'false', 'no', 'n', 'off'}:
        return False
    raise argparse.ArgumentTypeError(f'Invalid boolean value: {value!r}')


def add_x_label(option_labels: list[str]) -> list[str]:
    if 'x' in option_labels:
        return option_labels
    return [*option_labels, 'x']


def make_label_schema(option_labels: list[str]) -> dict:
    return {
        'type': 'object',
        'properties': {
            'label': {
                'type': 'string',
                'enum': option_labels,
            },
        },
        'required': ['label'],
        'additionalProperties': False,
    }


def build_openai_request(record: dict, model: str, include_x: bool) -> dict:
    option_labels = extract_option_labels(record['prompt'])
    if include_x:
        option_labels = add_x_label(option_labels)
    schema = make_label_schema(option_labels)
    return {
        'custom_id': f"amalgum_{record['id']}",
        'method': 'POST',
        'url': '/v1/chat/completions',
        'body': {
            'model': model,
            'messages': [
                {
                    'role': 'user',
                    'content': record['prompt'],
                }
            ],
            'response_format': {
                'type': 'json_schema',
                'json_schema': {
                    'name': 'option_label',
                    'strict': True,
                    'schema': schema,
                },
            },
        },
    }


def build_anthropic_request(record: dict, model: str, include_x: bool) -> dict:
    option_labels = extract_option_labels(record['prompt'])
    if include_x:
        option_labels = add_x_label(option_labels)
    schema = make_label_schema(option_labels)
    return {
        'custom_id': f"amalgum_{record['id']}",
        'params': {
            'model': model,
            'max_tokens': 64,
            'messages': [
                {
                    'role': 'user',
                    'content': record['prompt'],
                }
            ],
            'output_config': {
                'format': {
                    'type': 'json_schema',
                    'schema': schema,
                },
            },
        },
    }


def build_gemini_request(record: dict, model: str, include_x: bool) -> dict:
    del model
    option_labels = extract_option_labels(record['prompt'])
    if include_x:
        option_labels = add_x_label(option_labels)
    schema = make_label_schema(option_labels)
    return {
        'key': f"amalgum_{record['id']}",
        'request': {
            'contents': [
                {
                    'role': 'user',
                    'parts': [
                        {
                            'text': record['prompt'],
                        }
                    ],
                }
            ],
            'generation_config': {
                'response_mime_type': 'application/json',
                'response_json_schema': schema,
            },
        },
    }


BUILDERS = {
    'openai': build_openai_request,
    'anthropic': build_anthropic_request,
    'gemini': build_gemini_request,
}


def iter_token_batches(
    records: list[dict],
    model: str,
    builder: Callable[[dict, str, bool], dict],
    max_batch_tokens: int,
    max_batch_requests: Optional[int] = None,
    include_x: bool = False,
) -> list[tuple[list[dict], int]]:
    if max_batch_tokens <= 0:
        raise ValueError('--max-batch-tokens must be greater than zero.')
    if max_batch_requests is not None and max_batch_requests <= 0:
        raise ValueError('--batch-size must be greater than zero.')

    count_tokens = make_token_counter(model)
    batches = []
    current_batch = []
    current_tokens = 0

    for record in records:
        request = builder(record, model, include_x)
        request_tokens = count_tokens(request)
        if request_tokens > max_batch_tokens:
            raise ValueError(
                f"Record {record['id']} is estimated at {request_tokens} "
                f'tokens, exceeding --max-batch-tokens={max_batch_tokens}.'
            )
        over_token_limit = current_tokens + request_tokens > max_batch_tokens
        over_request_limit = (
            max_batch_requests is not None
            and len(current_batch) >= max_batch_requests
        )
        if current_batch and (over_token_limit or over_request_limit):
            batches.append((current_batch, current_tokens))
            current_batch = []
            current_tokens = 0
        current_batch.append(request)
        current_tokens += request_tokens

    if current_batch:
        batches.append((current_batch, current_tokens))
    return batches


def write_jsonl(path: Path, requests: list[dict]) -> None:
    with path.open('w', encoding='utf-8', newline='\n') as f:
        for request in requests:
            f.write(json.dumps(request, ensure_ascii=False) + '\n')


def write_anthropic_json(path: Path, requests: list[dict]) -> None:
    with path.open('w', encoding='utf-8', newline='\n') as f:
        json.dump({'requests': requests}, f, ensure_ascii=False, indent=2)
        f.write('\n')


def resolve_input_path(
    input_path: Optional[Path], *, exclude_x: bool
) -> Path:
    if input_path is not None:
        return input_path
    if exclude_x:
        return Path(DEFAULT_PROMPT_PATH)
    return Path(DEFAULT_PROMPT_WITH_X)


def resolve_output_path(
    output_path: Optional[Path],
    default_output_path: Path,
    *,
    provider: str,
    model: str,
    input_stem: str,
    extension: str,
    batch_index: int,
    provider_count: int,
    batch_count: int,
) -> Path:
    if output_path is None:
        return default_output_path

    pattern = str(output_path)
    if provider_count > 1 and not any(
        placeholder in pattern for placeholder in ('{provider}', '{model}')
    ):
        raise ValueError(
            '--output-path must contain {provider} or {model} when generating '
            'multiple providers.'
        )
    if batch_count > 1 and '{batch_index}' not in pattern:
        raise ValueError(
            '--output-path must contain {batch_index} when the input produces '
            f'multiple batches; got {batch_count}.'
        )
    try:
        formatted_path = pattern.format(
            provider=provider,
            model=model,
            input_stem=input_stem,
            extension=extension,
            batch_index=batch_index,
        )
    except KeyError as error:
        raise ValueError(
            f'Unknown --output-path placeholder: {error.args[0]}'
        ) from error
    return Path(formatted_path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--exclude-x',
        nargs='?',
        const=True,
        type=parse_bool,
        default=False,
    )
    parser.add_argument(
        '--input-path',
        '--input_path',
        '--input',
        type=Path,
        default=None,
        help=(
            'Prompt JSON path. When omitted, use amalgum_with_x.json or '
            'amalgum_without_x.json according to --exclude-x.'
        ),
    )
    parser.add_argument('--output-dir', default='experiment/batch')
    parser.add_argument(
        '--output-path',
        '--output_path',
        '--output',
        type=Path,
        default=None,
        help=(
            'Output path or path template. Available placeholders: '
            '{provider}, {model}, {input_stem}, {batch_index}, {extension}. '
            'Multiple providers or batches require distinguishing placeholders.'
        ),
    )
    parser.add_argument(
        '--max-batch-tokens',
        type=int,
        default=DEFAULT_MAX_BATCH_TOKENS,
        help='Estimated maximum tokens per output batch file.',
    )
    parser.add_argument(
        '--batch-size',
        type=int,
        default=None,
        help='Optional maximum requests per output batch file.',
    )
    parser.add_argument(
        '--providers',
        nargs='+',
        choices=tuple(PROVIDERS),
        default=tuple(PROVIDERS),
    )
    args = parser.parse_args()

    input_path = resolve_input_path(
        args.input_path,
        exclude_x=args.exclude_x,
    )
    if not input_path.exists():
        raise FileNotFoundError(f'input file not found: {input_path}')
    output_root = Path(args.output_dir)
    initialize_batch_state(output_root)
    with input_path.open('r', encoding='utf-8') as f:
        records = json.load(f)

    for provider in args.providers:
        config = PROVIDERS[provider]
        model = config['model']
        extension = config['extension']
        output_dir = output_root / model
        if args.output_path is None:
            output_dir.mkdir(parents=True, exist_ok=True)
        builder = BUILDERS[provider]
        batches = iter_token_batches(
            records,
            model,
            builder,
            args.max_batch_tokens,
            args.batch_size,
            include_x=not args.exclude_x,
        )
        for batch_index, (chunk, token_count) in enumerate(batches):
            if args.exclude_x:
                default_output_path = (
                    output_dir
                    / f'amalgum_without_x_batch_input_{batch_index}.{extension}'
                )
            else:
                default_output_path = (
                    output_dir
                    / f'amalgum_with_x_batch_input_{batch_index}.{extension}'
                )
            output_path = resolve_output_path(
                args.output_path,
                default_output_path,
                provider=provider,
                model=model,
                input_stem=input_path.stem,
                extension=extension,
                batch_index=batch_index,
                provider_count=len(args.providers),
                batch_count=len(batches),
            )
            output_path.parent.mkdir(parents=True, exist_ok=True)
            if provider == 'anthropic':
                write_anthropic_json(output_path, chunk)
            else:
                write_jsonl(output_path, chunk)
            print(
                f'Wrote {output_path} with {len(chunk)} requests '
                f'and estimated {token_count} tokens.'
            )


if __name__ == '__main__':
    main()
