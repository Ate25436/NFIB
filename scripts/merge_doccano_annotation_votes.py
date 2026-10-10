import argparse
import json
from collections import Counter
from pathlib import Path


def load_jsonl(path: Path) -> list[dict]:
    records: list[dict] = []
    with path.open("r", encoding="utf-8-sig") as f:
        for line_number, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON in {path} at line {line_number}: {exc}") from exc
    return records


def normalize_label(record: dict, source_name: str) -> str:
    labels = record.get("label")
    record_id = record.get("id", "<missing>")
    if not isinstance(labels, list) or len(labels) != 1:
        raise ValueError(f"{source_name}: id={record_id} must have exactly one label")

    label = labels[0]
    if not isinstance(label, str) or not label:
        raise ValueError(f"{source_name}: id={record_id} has invalid label value: {label!r}")

    return label


def build_input_map(input_paths: list[Path]) -> list[dict[int, dict]]:
    records_by_id: list[dict[int, dict]] = []
    for input_path in input_paths:
        for record in load_jsonl(input_path):
            record_id = record["id"]
            if record_id in [r["id"] for r in records_by_id]:
                raise ValueError(f"Duplicate input record id={record_id} in {input_path}")

            text = record.get("text") or record.get("data")
            if text is None:
                raise KeyError(f"Missing 'text' or 'data' in input record id={record_id}")

            input_record = {"id": record_id, "data": text, "genre": record.get("genre")}
            records_by_id.append(input_record)
    return records_by_id


def build_annotation_maps(output_files: list[Path], expected_records: list[dict[int, dict]]) -> dict[str, dict[int, str]]:
    annotation_maps: dict[str, dict[int, str]] = {}
    for output_path in output_files:
        annotations_by_id: dict[int, str] = {}
        for record in load_jsonl(output_path):
            record_id = record["id"]
            if record_id not in [r["id"] for r in expected_records]:
                continue
            if record_id in annotations_by_id:
                raise ValueError(f"{output_path.name}: duplicate annotation for id={record_id}")

            annotations_by_id[record_id] = normalize_label(record, output_path.name)
        annotation_maps[output_path.name] = annotations_by_id
    return annotation_maps


def majority_vote(labels: list[str], min_agreement: int | None = None) -> tuple[str, dict[str, int]]:
    counts = Counter(labels)
    most_common = counts.most_common()
    if not most_common:
        raise ValueError("Cannot vote on an empty label list")

    top_count = most_common[0][1]
    winners = sorted(label for label, count in most_common if count == top_count)
    if len(winners) != 1 or (min_agreement is not None and top_count < min_agreement):
        return "TIE", dict(sorted(counts.items()))
    return winners[0], dict(sorted(counts.items()))


def merge_record(
    record_id: int,
    base_record: dict,
    annotator_records: list[tuple[str, str]],
    min_agreement: int | None = None,
) -> dict:
    labels = [label for _, label in annotator_records]
    merged_label, vote_counts = majority_vote(labels, min_agreement)

    merged_record = {
        "id": record_id,
        "data": base_record["data"],
        "label": [merged_label],
        "votes": vote_counts,
    }
    if base_record.get("genre") is not None:
        merged_record["genre"] = base_record["genre"]
    return merged_record

def records_search_by_id(records: list[dict], record_id: int) -> dict:
    for record in records:
        if record.get("id") == record_id:
            return record
    raise ValueError(f"Record with id={record_id} not found")


def add_min_agreement_to_output_path(output_path: Path, min_agreement: int | None) -> Path:
    if min_agreement is None:
        return output_path

    suffix = f"_min_agreement_{min_agreement}"
    if output_path.stem.endswith(suffix):
        return output_path
    return output_path.with_name(f"{output_path.stem}{suffix}{output_path.suffix}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Merge doccano classification labels across annotators using majority vote."
    )
    parser.add_argument(
        "--input-path",
        action="append",
        dest="input_paths",
        default=None,
        help="Source doccano input JSONL file. Can be specified multiple times.",
    )
    parser.add_argument(
        "--output-dir",
        default="dataset/doccano/output",
        help="Directory containing annotator JSONL files.",
    )
    parser.add_argument(
        "--glob",
        default="NF_annotator_*.jsonl",
        help="Glob pattern for annotator JSONL files.",
    )
    parser.add_argument(
        "--output-path",
        default="dataset/doccano/output/NF_merged_majority_vote.jsonl",
        help="Path to the merged JSONL file.",
    )
    parser.add_argument(
        "--min-agreement",
        type=int,
        default=None,
        metavar="N",
        help="Require at least N annotators to agree; otherwise label the record TIE.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.min_agreement is not None and args.min_agreement < 1:
        raise ValueError("--min-agreement must be at least 1")

    input_paths = [
        Path(path)
        for path in (
            args.input_paths
            or [
                "dataset/doccano/input/amalgum_max200_pre.jsonl",
                "dataset/doccano/input/amalgum_max200_sub.jsonl",
            ]
        )
    ]
    output_dir = Path(args.output_dir)
    output_path = add_min_agreement_to_output_path(Path(args.output_path), args.min_agreement)

    expected_records = build_input_map(input_paths)
    output_files = sorted(path for path in output_dir.glob(args.glob) if path.resolve() != output_path.resolve())
    if not output_files:
        raise ValueError(f"No annotator JSONL files found in {output_dir} matching {args.glob!r}")

    annotation_maps = build_annotation_maps(output_files, expected_records)

    common_record_ids = set(r["id"] for r in expected_records)
    for _, annotations_by_id in annotation_maps.items():
        common_record_ids &= set(annotations_by_id)

    merged_records: list[dict] = []
    for record_id in sorted(common_record_ids):
        annotator_records = [
            (annotator_name, annotations_by_id[record_id])
            for annotator_name, annotations_by_id in annotation_maps.items()
        ]
        merged_records.append(
            merge_record(
                record_id,
                records_search_by_id(expected_records, record_id),
                annotator_records,
                args.min_agreement,
            )
        )

    skipped_missing_ids = sorted(set(r["id"] for r in expected_records) - common_record_ids)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="\n") as f:
        for record in merged_records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    annotator_names = ", ".join(name for name, _ in annotation_maps.items())
    print(f"Annotator files: {annotator_names}")
    print(f"Wrote {len(merged_records)} merged record(s) to {output_path}.")
    if skipped_missing_ids:
        preview = ", ".join(str(record_id) for record_id in skipped_missing_ids[:10])
        suffix = "..." if len(skipped_missing_ids) > 10 else ""
        print(f"Skipped {len(skipped_missing_ids)} record(s) missing from at least one annotator: {preview}{suffix}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
