import argparse
import json
from collections import defaultdict
from pathlib import Path


CONTEXT_KEYS = [
    "previous_sentence",
    "before_sentence",
    "subsequent_sentence",
]

REPLACEMENTS = {
    "-LRB-": "(",
    "-RRB-": ")",
    "-LSB-": "[",
    "-RSB-": "]",
    "-LCB-": "{",
    "-RCB-": "}",
    "``": '"',
    "''": '"',
}

OPTION_LABELS = "abcdefghijklmnopqrstuvw"


def load_records(path: Path) -> list[dict]:
    text = path.read_text(encoding="utf-8")
    return json.loads(text)


def normalize_text(text: str) -> str:
    for src, dest in REPLACEMENTS.items():
        text = text.replace(src, dest)
    return text.strip()


def build_text(context: dict, rewrites: list[str]) -> str:
    if len(rewrites) > len(OPTION_LABELS):
        raise ValueError(f"Too many rewrite options: {len(rewrites)}")

    lines = ["Context:"]
    lines.extend(normalize_text(context[key]) for key in CONTEXT_KEYS)
    lines.append("")
    lines.append("Options:")
    lines.extend(
        f"{label}.\n{context.get('previous_sentence', '')}\n{normalize_text(rewrite)}\n{context.get('subsequent_sentence', '')}"
        for label, rewrite in zip(OPTION_LABELS, rewrites)
    )
    lines.append("x. None of the above")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--converted",
        default="dataset/converted/training_converted.json",
    )
    parser.add_argument(
        "--rewrote",
        default="dataset/rewrote/training_rewrote.jsonl",
    )
    parser.add_argument(
        "--output",
        default="dataset/doccano/input/training_rewrite_sentences.jsonl",
    )
    args = parser.parse_args()

    converted_records = load_records(Path(args.converted))
    rewrote_records = load_records(Path(args.rewrote))

    context_by_sentence_id = {}
    for record in converted_records:
        sentence_id = record["sentence_id"]
        if sentence_id not in context_by_sentence_id:
            context_by_sentence_id[sentence_id] = {
                key: record[key]
                for key in CONTEXT_KEYS
            }

    rewrites_by_sentence_id = defaultdict(list)
    seen_sentences_by_sentence_id = defaultdict(set)
    for record in rewrote_records:
        sentence_id = record["sentence_id"]
        sentence = record["sentence"]
        rewrote = record["rewrote"]
        if "\n" in rewrote or "\r" in rewrote:
            continue
        if "but" not in rewrote:
            continue
        if sentence in seen_sentences_by_sentence_id[sentence_id]:
            continue
        seen_sentences_by_sentence_id[sentence_id].add(sentence)
        rewrites_by_sentence_id[sentence_id].append(rewrote)

    output_records = []
    for sentence_id in sorted(rewrites_by_sentence_id):
        rewrites = rewrites_by_sentence_id[sentence_id]
        if len(rewrites) <= 1:
            continue
        if sentence_id not in context_by_sentence_id:
            raise KeyError(f"Missing context for sentence_id={sentence_id}")
        output_records.append({
            "id": sentence_id,
            "text": build_text(context_by_sentence_id[sentence_id], rewrites),
        })

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="\n") as f:
        for record in output_records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()
