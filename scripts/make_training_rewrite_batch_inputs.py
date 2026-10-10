import argparse
import json
from pathlib import Path

PROMPT_TEMPLATE = (
    'Please rewrite the following sentence into a natural sentence '
    'without changing its meaning and remaining "but":\n'
    '{sentence}'
)

INITIAL_BATCH_STATE = {
    'current_id': 0,
    'batches': [],
}


def build_request(record: dict, model: str) -> dict:
    return {
        'custom_id': f"pos_{record['instance_id']}_{record['sentence_id']}",
        'method': 'POST',
        'url': '/v1/chat/completions',
        'body': {
            'model': model,
            'messages': [
                {
                    'role': 'user',
                    'content': PROMPT_TEMPLATE.format(sentence=record['sentence']),
                }
            ],
        },
    }


def initialize_batch_state(output_dir: Path) -> Path:
    batch_state_path = output_dir / 'batch_state.json'
    if batch_state_path.exists():
        print(f'Kept existing batch state: {batch_state_path}')
        return batch_state_path

    with batch_state_path.open('x', encoding='utf-8', newline='\n') as f:
        json.dump(INITIAL_BATCH_STATE, f, ensure_ascii=False, indent=2)
        f.write('\n')
    print(f'Wrote initial batch state: {batch_state_path}')
    return batch_state_path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', default='dataset/converted/training_converted.json')
    parser.add_argument('--output-dir', default='training_batch/positive_batch_inputs')
    parser.add_argument('--batch-size', type=int, default=100)
    parser.add_argument('--prefix', default='positive_batch_input_')
    parser.add_argument('--model', default='gpt-5')
    args = parser.parse_args()

    input_path = Path(args.input)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    initialize_batch_state(output_dir)

    with input_path.open('r', encoding='utf-8') as f:
        records = json.load(f)

    for batch_index, start in enumerate(range(0, len(records), args.batch_size)):
        chunk = records[start:start + args.batch_size]
        output_path = output_dir / f"{args.prefix}{batch_index}.jsonl"
        with output_path.open('w', encoding='utf-8', newline='\n') as f:
            for record in chunk:
                request = build_request(record, args.model)
                f.write(json.dumps(request, ensure_ascii=False) + '\n')


if __name__ == '__main__':
    main()
