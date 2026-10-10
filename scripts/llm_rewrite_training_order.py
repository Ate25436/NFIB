import argparse
import datetime
import json
import os
from pathlib import Path

from dotenv import load_dotenv
from openai import BadRequestError, OpenAI

load_dotenv()

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

PROMPT_PREFIX = (
    'Please rewrite the following sentence into a natural sentence '
    'without changing its meaning and remaining "but":\n'
)


def is_billing_hard_limit_error(error: BadRequestError) -> bool:
    body = getattr(error, "body", None)
    if isinstance(body, dict):
        error_body = body.get("error", {})
        return error_body.get("code") == "billing_hard_limit_reached"
    return "billing_hard_limit_reached" in str(error)


def raise_batch_order_error(error: BadRequestError) -> None:
    if is_billing_hard_limit_error(error):
        raise SystemExit(
            "OpenAI API の billing hard limit に到達しているため、バッチを作成できませんでした。\n"
            "OpenAI Platform の Billing/Usage limits で上限または残高を確認し、"
            "上限を引き上げるか請求設定を更新してから同じコマンドを再実行してください。\n"
            "batch id ファイルは更新していません。"
        ) from error
    raise error

def make_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", type=str, default="check", help="Mode: order or check")
    parser.add_argument(
        "--batch-input-dir",
        "--batch_input_dir",
        type=str,
        default=None,
        help="Directory containing batch input JSONL files. If omitted, use training_batch/{data_type}_batch_inputs.",
    )
    parser.add_argument(
        "--batch-id-path",
        "--batch_id_path",
        type=str,
        default="training_batch/batch_id.json",
        help="Path to the JSON file that tracks ordered batch IDs.",
    )
    parser.add_argument(
        "--output",
        "--output_path",
        type=str,
        default=None,
        help="Output path for check results. If omitted, use the default dataset/rewrote paths.",
    )
    args = parser.parse_args()
    return args


def get_batch_input_path(data_type: str, current_id: int, batch_input_dir=None):
    if batch_input_dir is None:
        return Path(f"training_batch/{data_type}_batch_inputs/{data_type}_batch_input_{current_id}.jsonl")
    return Path(batch_input_dir) / f"{data_type}_batch_input_{current_id}.jsonl"


def load_batch_info(batch_id_path: str) -> dict:
    path = Path(batch_id_path)
    if not path.exists():
        return {"current_id": 0, "batches": []}
    with path.open("r") as f:
        return json.load(f)


def load_json_list(path: Path) -> list:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def save_json_list(path: Path, records: list) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(records, f, ensure_ascii=False, indent=4)


def parse_custom_id(custom_id: str) -> tuple[int, int]:
    parts = custom_id.rsplit("_", 2)
    if len(parts) != 3:
        raise ValueError(f"Invalid custom_id: {custom_id}")
    _, instance_id, sentence_id = parts
    return int(instance_id), int(sentence_id)


def extract_sentence_from_request(request: dict) -> str:
    messages = request.get("body", {}).get("messages", [])
    user_messages = [message for message in messages if message.get("role") == "user"]
    if not user_messages:
        raise ValueError(f"Missing user message in request: {request.get('custom_id')}")
    content = user_messages[-1].get("content", "")
    if content.startswith(PROMPT_PREFIX):
        return content[len(PROMPT_PREFIX):]
    return content


def load_batch_input_sentences(input_path: Path) -> dict[str, dict]:
    if not input_path.exists():
        raise FileNotFoundError(
            f"Batch input file not found: {input_path}. "
            "Cannot safely attach original sentences to results."
        )

    sentences = {}
    with input_path.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            request = json.loads(line)
            custom_id = request["custom_id"]
            instance_id, sentence_id = parse_custom_id(custom_id)
            sentences[custom_id] = {
                "sentence_id": sentence_id,
                "instance_id": instance_id,
                "sentence": extract_sentence_from_request(request),
            }
    return sentences


def order_batch(batch_input_dir=None, batch_id_path="training_batch/batch_id.json"):
    batch_info = load_batch_info(batch_id_path)
    data_types = ["positive", "negative"]
    current_id = batch_info["current_id"]
    ordered_batches = batch_info.get("batches", [])
    new_batches = []
    for data_type in data_types:
        input_path = get_batch_input_path(data_type, current_id, batch_input_dir)
        if not input_path.exists():
            continue
        upload = None
        with input_path.open("rb") as f:
            try:
                upload = client.files.create(file=f, purpose="batch")
                batch = client.batches.create(
                    input_file_id=upload.id,
                    endpoint="/v1/chat/completions",
                    completion_window="24h"  # 実行時間の上限
                )
            except BadRequestError as error:
                if upload is not None:
                    try:
                        client.files.delete(upload.id)
                    except Exception:
                        pass
                raise_batch_order_error(error)
        new_batches.append({
            "data_type": data_type,
            "id": current_id,
            "batch_id": batch.id,
            "input_path": str(input_path)
        })
    current_id += 1
    d_today = datetime.date.today()
    ordered_batches.extend(new_batches)
    Path(batch_id_path).parent.mkdir(parents=True, exist_ok=True)
    with open(batch_id_path, "w") as f:
        json.dump({
            "date": d_today.isoformat(),
            "current_id": current_id,
            "batches": ordered_batches
        }, f, indent=4)

def check_batches(batch_id_path="training_batch/batch_id.json", output_path=None):
    with open(batch_id_path, "r") as f:
        batch_info = json.load(f)
    positive_batch_results = []
    negative_batch_results = []
    output_path = Path(output_path) if output_path is not None else None
    if output_path is None:
        current_positive_rewrote = load_json_list(Path("dataset/rewrote/training_rewrote.jsonl"))
        current_negative_rewrote = load_json_list(Path("dataset/rewrote/training_negative_rewrote.jsonl"))
    else:
        current_rewrote = load_json_list(output_path)

    completed = []
    for batch in batch_info.get("batches", []):
        batch_id = batch["batch_id"]
        data_type = batch["data_type"]
        input_id = batch["id"]
        status = client.batches.retrieve(batch_id)
        result = client.batches.retrieve(batch_id)
        
        if status.status == "completed":
            input_sentences = load_batch_input_sentences(Path(batch["input_path"]))
            
            output = client.files.content(result.output_file_id)
            for line in output.text.strip().split("\n"):
                record = json.loads(line)
                response = record["response"]["body"]["choices"][0]["message"]["content"]
                custom_id = record["custom_id"]
                source = input_sentences.get(custom_id)
                if source is None:
                    raise KeyError(
                        f"{custom_id} was returned by batch {batch_id}, "
                        f"but it is missing from {batch['input_path']}"
                    )
                result_record = {
                    "sentence_id": source["sentence_id"],
                    "instance_id": source["instance_id"],
                    "sentence": source["sentence"],
                    "rewrote": response
                }
                if data_type == "positive":
                    positive_batch_results.append(result_record)
                else:
                    negative_batch_results.append(result_record)
            completed.append((data_type, input_id))
            print(f"Batch {data_type}-{input_id} completed.")
        elif status.status in ["in_progress", "validating", "finalizing"]:
            print(result.request_counts)
        else:
            print(f"Batch {batch_id} failed with status: {status.status}")
    if output_path is None:
        current_positive_rewrote.extend(positive_batch_results)
        current_negative_rewrote.extend(negative_batch_results)
    else:
        current_rewrote.extend(positive_batch_results + negative_batch_results)
    batches = [b for b in batch_info.get("batches", []) if (b["data_type"], b["id"]) not in completed]
    d_today = datetime.date.today()
    with open(batch_id_path, "w") as f:
        json.dump({
            "date": d_today.isoformat(),
            "current_id": batch_info["current_id"],
            "batches": batches
        }, f, indent=4)

    if output_path is None:
        save_json_list(Path("dataset/rewrote/training_rewrote.jsonl"), current_positive_rewrote)
        save_json_list(Path("dataset/rewrote/training_negative_rewrote.jsonl"), current_negative_rewrote)
    else:
        save_json_list(output_path, current_rewrote)
    
        


if __name__ == "__main__":
    args = make_args()
    if args.mode == "check":
        check_batches(batch_id_path=args.batch_id_path, output_path=args.output)
    elif args.mode == "order":
        order_batch(batch_input_dir=args.batch_input_dir, batch_id_path=args.batch_id_path)
    else:
        print(f"Invalid mode {args.mode}. Use 'order' or 'check'.")
