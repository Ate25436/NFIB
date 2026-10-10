import argparse
import datetime
import json
import os
from pathlib import Path
from typing import Any, Optional

from dotenv import load_dotenv


DEFAULT_BATCH_ROOT = Path('experiment/batch')
DEFAULT_BATCH_STATE_PATH = DEFAULT_BATCH_ROOT / 'batch_state.json'
DEFAULT_OUTPUT_DIR = DEFAULT_BATCH_ROOT / 'results'
LEGACY_INPUT_PREFIX = 'amalgum_batch_input_'
DEFAULT_INPUT_PREFIX = 'amalgum_with_x_batch_input_'
EXCLUDE_X_INPUT_PREFIX = 'amalgum_without_x_batch_input_'
PROVIDER_MODELS = {
    'openai': {'gpt-5-mini', 'gpt-5.5'},
    'anthropic': {'claude-opus-4-8'},
    'gemini': {'gemini-3.5-flash'},
}
INPUT_EXTENSIONS = {
    'openai': '.jsonl',
    'anthropic': '.json',
    'gemini': '.jsonl',
}
TERMINAL_STATUSES = {'completed', 'failed', 'expired', 'cancelled'}
GEMINI_TERMINAL_STATES = {
    'JOB_STATE_SUCCEEDED',
    'JOB_STATE_FAILED',
    'JOB_STATE_CANCELLED',
    'JOB_STATE_EXPIRED',
}


def make_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', choices=('order', 'check'), default='check')
    parser.add_argument('--provider', choices=tuple(PROVIDER_MODELS), default=None)
    parser.add_argument('--model', default=None)
    parser.add_argument(
        '--batch-input-root',
        '--batch_input_root',
        type=Path,
        default=Path(DEFAULT_BATCH_ROOT),
        help='Default: experiment/batch/{model}',
    )
    parser.add_argument(
        '--batch-state-path',
        '--batch_state_path',
        type=Path,
        default=DEFAULT_BATCH_STATE_PATH,
    )
    parser.add_argument(
        '--output-dir',
        '--output_dir',
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
    )
    parser.add_argument(
        '--exclude-x',
        action='store_true',
        help=(
            'Use amalgum_without_x_batch_input_*. '
            'The default is amalgum_with_x_batch_input_*.'
        ),
    )
    parser.add_argument(
        '--input-prefix',
        default=None,
        help='Override the batch input filename prefix.',
    )
    return parser.parse_args()


def today() -> str:
    return datetime.date.today().isoformat()


def now() -> str:
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


def infer_provider(model: str) -> str:
    for provider, models in PROVIDER_MODELS.items():
        if model in models:
            return provider
    raise SystemExit(
        f'Could not infer provider for model {model!r}. '
        'Specify --provider explicitly.'
    )


def resolve_provider(model: str, provider: Optional[str]) -> str:
    resolved = provider or infer_provider(model)
    known_models = PROVIDER_MODELS[resolved]
    if model not in known_models:
        print(
            f'Warning: model {model!r} is not in the known {resolved} model list.'
        )
    return resolved


def required_api_key(provider: str) -> str:
    variable = {
        'openai': 'OPENAI_API_KEY',
        'anthropic': 'ANTHROPIC_API_KEY',
        'gemini': 'GEMINI_API_KEY',
    }[provider]
    api_key = os.getenv(variable)
    if not api_key:
        raise SystemExit(f'{variable} is not set.')
    return api_key


def create_client(provider: str) -> Any:
    api_key = required_api_key(provider)
    if provider == 'openai':
        from openai import OpenAI

        return OpenAI(api_key=api_key)
    if provider == 'anthropic':
        try:
            import anthropic
        except ImportError as error:
            raise SystemExit(
                'The anthropic package is required. Run: uv sync'
            ) from error
        return anthropic.Anthropic(api_key=api_key)
    try:
        from google import genai
    except ImportError as error:
        raise SystemExit(
            'The google-genai package is required. Run: uv sync'
        ) from error
    return genai.Client(api_key=api_key)


def load_batch_state(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {'models': {}, 'batches': []}
    with path.open('r', encoding='utf-8') as f:
        state = json.load(f)
    state.setdefault('models', {})
    state.setdefault('batches', [])
    return state


def save_batch_state(path: Path, state: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_suffix(path.suffix + '.tmp')
    with temporary_path.open('w', encoding='utf-8', newline='\n') as f:
        json.dump(state, f, ensure_ascii=False, indent=2)
        f.write('\n')
    temporary_path.replace(path)


def get_model_state(
    state: dict[str, Any],
    model: str,
    provider: str,
) -> dict[str, Any]:
    model_state = state['models'].setdefault(
        model,
        {
            'provider': provider,
            'date': None,
            'current_id': 0,
            'current_ids': {},
        },
    )
    model_state.setdefault('provider', provider)
    model_state.setdefault('current_id', 0)
    model_state.setdefault('current_ids', {})
    return model_state


def resolve_input_prefix(
    input_prefix: Optional[str],
    exclude_x: bool,
) -> str:
    if input_prefix is not None:
        return input_prefix
    if exclude_x:
        return EXCLUDE_X_INPUT_PREFIX
    return DEFAULT_INPUT_PREFIX


def get_current_id(model_state: dict[str, Any], input_prefix: str) -> int:
    current_ids = model_state.setdefault('current_ids', {})
    if input_prefix not in current_ids:
        if input_prefix == LEGACY_INPUT_PREFIX:
            current_ids[input_prefix] = model_state.get('current_id', 0)
        else:
            current_ids[input_prefix] = 0
    return current_ids[input_prefix]


def advance_current_id(
    model_state: dict[str, Any],
    input_prefix: str,
    next_id: int,
) -> None:
    model_state.setdefault('current_ids', {})[input_prefix] = next_id
    if input_prefix == LEGACY_INPUT_PREFIX:
        model_state['current_id'] = next_id


def batch_input_prefix(batch_record: dict[str, Any]) -> str:
    input_prefix = batch_record.get('input_prefix')
    if input_prefix:
        return input_prefix

    input_name = Path(batch_record.get('input_path', '')).name
    batch_id = str(batch_record.get('id'))
    suffix = Path(input_name).suffix
    id_suffix = f'{batch_id}{suffix}'
    if suffix and input_name.endswith(id_suffix):
        return input_name[: -len(id_suffix)]
    return LEGACY_INPUT_PREFIX


def batch_output_filename(batch_record: dict[str, Any]) -> str:
    input_prefix = batch_input_prefix(batch_record)
    if input_prefix.endswith('_input_'):
        output_prefix = input_prefix[: -len('_input_')] + '_output_'
    else:
        output_prefix = input_prefix + 'output_'
    return f'{output_prefix}{batch_record["id"]}.jsonl'


def get_batch_input_path(
    batch_input_dir: Optional[Path],
    input_prefix: str,
    current_id: int,
    model: str,
    provider: str,
) -> Path:
    extension = INPUT_EXTENSIONS[provider]
    filename = f'{input_prefix}{current_id}{extension}'
    if batch_input_dir is not None:
        return batch_input_dir / filename

    model_path = DEFAULT_BATCH_ROOT / model / filename
    legacy_path = DEFAULT_BATCH_ROOT / filename
    if model_path.exists() or not legacy_path.exists():
        return model_path
    return legacy_path


def read_jsonl(path: Path) -> list[dict]:
    records = []
    with path.open('r', encoding='utf-8') as f:
        for line_number, line in enumerate(f, start=1):
            if not line.strip():
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError as error:
                raise SystemExit(
                    f'{path}:{line_number} is not valid JSON: {error}'
                ) from error
    return records


def validate_unique_ids(path: Path, ids: list[Any]) -> None:
    if any(not custom_id for custom_id in ids):
        raise SystemExit(f'{path} contains a request without an ID.')
    if len(ids) != len(set(ids)):
        raise SystemExit(f'{path} contains duplicate request IDs.')


def validate_openai_input(path: Path, model: str) -> tuple[str, int]:
    requests = read_jsonl(path)
    endpoints = {request.get('url') for request in requests}
    models = {request.get('body', {}).get('model') for request in requests}
    validate_unique_ids(path, [request.get('custom_id') for request in requests])
    if not requests:
        raise SystemExit(f'{path} contains no batch requests.')
    if models != {model}:
        raise SystemExit(f'{path} uses models {models}, not only {model!r}.')
    if len(endpoints) != 1 or None in endpoints:
        raise SystemExit(f'{path} must contain exactly one valid endpoint.')
    return endpoints.pop(), len(requests)


def validate_anthropic_input(path: Path, model: str) -> tuple[None, int]:
    with path.open('r', encoding='utf-8') as f:
        document = json.load(f)
    requests = document.get('requests', [])
    models = {request.get('params', {}).get('model') for request in requests}
    validate_unique_ids(path, [request.get('custom_id') for request in requests])
    if not requests:
        raise SystemExit(f'{path} contains no batch requests.')
    if models != {model}:
        raise SystemExit(f'{path} uses models {models}, not only {model!r}.')
    return None, len(requests)


def validate_gemini_input(path: Path, model: str) -> tuple[None, int]:
    del model
    requests = read_jsonl(path)
    validate_unique_ids(path, [request.get('key') for request in requests])
    if not requests:
        raise SystemExit(f'{path} contains no batch requests.')
    if any('request' not in request for request in requests):
        raise SystemExit(f'{path} contains a request without request data.')
    return None, len(requests)


def validate_batch_input(
    path: Path,
    model: str,
    provider: str,
) -> tuple[Optional[str], int]:
    validators = {
        'openai': validate_openai_input,
        'anthropic': validate_anthropic_input,
        'gemini': validate_gemini_input,
    }
    return validators[provider](path, model)


def serialize_object(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, float, bool, dict, list)):
        return value
    if hasattr(value, 'model_dump'):
        return value.model_dump(mode='json')
    if hasattr(value, 'to_dict'):
        return value.to_dict()
    return str(value)


def write_bytes(path: Path, content: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_suffix(path.suffix + '.tmp')
    temporary_path.write_bytes(content)
    temporary_path.replace(path)


def write_jsonl_objects(path: Path, records: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_suffix(path.suffix + '.tmp')
    with temporary_path.open('w', encoding='utf-8', newline='\n') as f:
        for record in records:
            serialized = serialize_object(record)
            f.write(json.dumps(serialized, ensure_ascii=False) + '\n')
    temporary_path.replace(path)


def order_openai(
    client: Any,
    input_path: Path,
    endpoint: str,
    model: str,
    current_id: int,
) -> dict[str, Any]:
    upload = None
    try:
        with input_path.open('rb') as f:
            upload = client.files.create(file=f, purpose='batch')
        batch = client.batches.create(
            input_file_id=upload.id,
            endpoint=endpoint,
            completion_window='24h',
            metadata={'model': model, 'local_batch_id': str(current_id)},
        )
    except Exception as error:
        if upload is not None:
            try:
                client.files.delete(upload.id)
            except Exception:
                pass
        body = getattr(error, 'body', None)
        if isinstance(body, dict):
            error_body = body.get('error', {})
            if error_body.get('code') == 'billing_hard_limit_reached':
                raise SystemExit(
                    'OpenAI API の billing hard limit に到達しています。'
                ) from error
        raise
    return {
        'batch_id': batch.id,
        'input_file_id': upload.id,
        'status': getattr(batch, 'status', 'validating'),
        'provider_status': getattr(batch, 'status', 'validating'),
        'request_counts': serialize_object(
            getattr(batch, 'request_counts', None)
        ),
    }


def order_anthropic(
    client: Any,
    input_path: Path,
    endpoint: Optional[str],
    model: str,
    current_id: int,
) -> dict[str, Any]:
    del endpoint, model, current_id
    with input_path.open('r', encoding='utf-8') as f:
        requests = json.load(f)['requests']
    batch = client.messages.batches.create(requests=requests)
    provider_status = batch.processing_status
    return {
        'batch_id': batch.id,
        'input_file_id': None,
        'status': (
            'completed' if provider_status == 'ended' else provider_status
        ),
        'provider_status': provider_status,
        'request_counts': serialize_object(batch.request_counts),
    }


def order_gemini(
    client: Any,
    input_path: Path,
    endpoint: Optional[str],
    model: str,
    current_id: int,
) -> dict[str, Any]:
    del endpoint
    from google.genai import types

    uploaded_file = client.files.upload(
        file=input_path,
        config=types.UploadFileConfig(
            display_name=f'{model}-amalgum-{current_id}',
            mime_type='jsonl',
        ),
    )
    batch = client.batches.create(
        model=model,
        src=uploaded_file.name,
        config={'display_name': f'{model}-amalgum-{current_id}'},
    )
    provider_status = batch.state.name
    return {
        'batch_id': batch.name,
        'input_file_id': uploaded_file.name,
        'status': normalize_gemini_status(provider_status),
        'provider_status': provider_status,
        'request_counts': None,
    }


def normalize_gemini_status(provider_status: str) -> str:
    mapping = {
        'JOB_STATE_SUCCEEDED': 'completed',
        'JOB_STATE_FAILED': 'failed',
        'JOB_STATE_CANCELLED': 'cancelled',
        'JOB_STATE_EXPIRED': 'expired',
    }
    return mapping.get(provider_status, provider_status.lower())


def order_batch(
    client: Any,
    provider: str,
    model: str,
    batch_input_dir: Optional[Path] = None,
    batch_state_path: Path = DEFAULT_BATCH_STATE_PATH,
    input_prefix: str = DEFAULT_INPUT_PREFIX,
) -> None:
    state = load_batch_state(batch_state_path)
    model_state = get_model_state(state, model, provider)
    current_id = get_current_id(model_state, input_prefix)
    input_path = get_batch_input_path(
        batch_input_dir,
        input_prefix,
        current_id,
        model,
        provider,
    )
    if not input_path.exists():
        raise SystemExit(f'Batch input file not found: {input_path}')

    duplicate = next(
        (
            batch
            for batch in state['batches']
            if (
                batch['model'] == model
                and batch['id'] == current_id
                and batch_input_prefix(batch) == input_prefix
            )
        ),
        None,
    )
    if duplicate is not None:
        raise SystemExit(
            f'Batch {model} {input_prefix}id={current_id} is already '
            f'recorded as '
            f'{duplicate["batch_id"]}.'
        )

    endpoint, request_count = validate_batch_input(
        input_path,
        model,
        provider,
    )
    orderers = {
        'openai': order_openai,
        'anthropic': order_anthropic,
        'gemini': order_gemini,
    }
    remote = orderers[provider](
        client,
        input_path,
        endpoint,
        model,
        current_id,
    )
    timestamp = now()
    state['batches'].append(
        {
            'provider': provider,
            'model': model,
            'id': current_id,
            'input_prefix': input_prefix,
            'batch_id': remote['batch_id'],
            'input_file_id': remote['input_file_id'],
            'input_path': str(input_path),
            'endpoint': endpoint,
            'request_count': request_count,
            'status': remote['status'],
            'provider_status': remote['provider_status'],
            'ordered_at': timestamp,
            'updated_at': timestamp,
            'output_file_id': None,
            'error_file_id': None,
            'output_path': None,
            'error_path': None,
            'request_counts': remote['request_counts'],
        }
    )
    model_state['provider'] = provider
    model_state['date'] = today()
    advance_current_id(model_state, input_prefix, current_id + 1)
    save_batch_state(batch_state_path, state)
    print(
        f'Ordered {provider}/{model} {input_prefix}id={current_id}: '
        f'{remote["batch_id"]}'
    )


def check_openai(
    client: Any,
    batch_record: dict[str, Any],
    output_path: Path,
) -> None:
    batch = client.batches.retrieve(batch_record['batch_id'])
    batch_record['provider_status'] = batch.status
    batch_record['status'] = batch.status
    batch_record['request_counts'] = serialize_object(
        getattr(batch, 'request_counts', None)
    )
    batch_record['output_file_id'] = getattr(batch, 'output_file_id', None)
    batch_record['error_file_id'] = getattr(batch, 'error_file_id', None)

    if batch_record['output_file_id'] and not batch_record.get('output_path'):
        content = client.files.content(batch_record['output_file_id'])
        raw = getattr(content, 'content', None)
        if not isinstance(raw, bytes):
            raw = content.read()
        write_bytes(output_path, raw)
        batch_record['output_path'] = str(output_path)

    if batch_record['error_file_id'] and not batch_record.get('error_path'):
        error_path = output_path.with_name(
            output_path.name.replace('_output_', '_error_')
        )
        content = client.files.content(batch_record['error_file_id'])
        raw = getattr(content, 'content', None)
        if not isinstance(raw, bytes):
            raw = content.read()
        write_bytes(error_path, raw)
        batch_record['error_path'] = str(error_path)


def check_anthropic(
    client: Any,
    batch_record: dict[str, Any],
    output_path: Path,
) -> None:
    batch = client.messages.batches.retrieve(batch_record['batch_id'])
    provider_status = batch.processing_status
    batch_record['provider_status'] = provider_status
    batch_record['status'] = (
        'completed' if provider_status == 'ended' else provider_status
    )
    batch_record['request_counts'] = serialize_object(batch.request_counts)
    if provider_status == 'ended' and not batch_record.get('output_path'):
        results = client.messages.batches.results(batch_record['batch_id'])
        write_jsonl_objects(output_path, results)
        batch_record['output_path'] = str(output_path)


def check_gemini(
    client: Any,
    batch_record: dict[str, Any],
    output_path: Path,
) -> None:
    batch = client.batches.get(name=batch_record['batch_id'])
    provider_status = batch.state.name
    batch_record['provider_status'] = provider_status
    batch_record['status'] = normalize_gemini_status(provider_status)
    batch_record['provider_error'] = serialize_object(
        getattr(batch, 'error', None)
    )
    destination = getattr(batch, 'dest', None)
    result_file_id = (
        getattr(destination, 'file_name', None) if destination else None
    )
    batch_record['output_file_id'] = result_file_id
    if (
        provider_status == 'JOB_STATE_SUCCEEDED'
        and result_file_id
        and not batch_record.get('output_path')
    ):
        content = client.files.download(file=result_file_id)
        write_bytes(output_path, content)
        batch_record['output_path'] = str(output_path)


def batch_provider(batch_record: dict[str, Any]) -> str:
    provider = batch_record.get('provider')
    if provider:
        return provider
    provider = infer_provider(batch_record['model'])
    batch_record['provider'] = provider
    return provider


def check_batches(
    clients: dict[str, Any],
    batch_state_path: Path = DEFAULT_BATCH_STATE_PATH,
    output_dir: Path = DEFAULT_OUTPUT_DIR,
    model: Optional[str] = None,
    provider: Optional[str] = None,
) -> None:
    state = load_batch_state(batch_state_path)
    selected_batches = []
    for batch in state['batches']:
        current_provider = batch_provider(batch)
        if model is not None and batch['model'] != model:
            continue
        if provider is not None and current_provider != provider:
            continue
        selected_batches.append(batch)
    if not selected_batches:
        raise SystemExit('No matching batches are recorded.')

    checkers = {
        'openai': check_openai,
        'anthropic': check_anthropic,
        'gemini': check_gemini,
    }
    checked_models = set()
    for batch_record in selected_batches:
        current_provider = batch_provider(batch_record)
        batch_model = batch_record['model']
        checked_models.add((batch_model, current_provider))
        if (
            batch_record.get('status') in TERMINAL_STATUSES
            and (
                batch_record.get('status') != 'completed'
                or batch_record.get('output_path')
            )
        ):
            print(
                f'{current_provider}/{batch_model} id={batch_record["id"]}: '
                f'{batch_record["status"]}'
            )
            continue

        output_path = (
            output_dir
            / batch_model
            / batch_output_filename(batch_record)
        )
        checkers[current_provider](
            clients[current_provider],
            batch_record,
            output_path,
        )
        batch_record['updated_at'] = now()
        print(
            f'{current_provider}/{batch_model} id={batch_record["id"]}: '
            f'{batch_record["status"]}'
        )
        save_batch_state(batch_state_path, state)

    for checked_model, checked_provider in checked_models:
        get_model_state(
            state,
            checked_model,
            checked_provider,
        )['date'] = today()
    save_batch_state(batch_state_path, state)


def providers_for_check(
    state_path: Path,
    model: Optional[str],
    provider: Optional[str],
) -> set[str]:
    if provider:
        return {provider}
    if model:
        return {infer_provider(model)}
    state = load_batch_state(state_path)
    return {batch_provider(batch) for batch in state['batches']}


def main() -> None:
    args = make_args()
    load_dotenv()
    if args.mode == 'order':
        if not args.model:
            raise SystemExit('--model is required in order mode.')
        provider = resolve_provider(args.model, args.provider)
        input_prefix = resolve_input_prefix(args.input_prefix, args.exclude_x)
        order_batch(
            client=create_client(provider),
            provider=provider,
            model=args.model,
            batch_input_dir=Path(args.batch_input_root / args.model),
            batch_state_path=args.batch_state_path,
            input_prefix=input_prefix,
        )
        return

    providers = providers_for_check(
        args.batch_state_path,
        args.model,
        args.provider,
    )
    clients = {provider: create_client(provider) for provider in providers}
    check_batches(
        clients=clients,
        batch_state_path=args.batch_state_path,
        output_dir=args.output_dir,
        model=args.model,
        provider=args.provider,
    )


if __name__ == '__main__':
    main()
