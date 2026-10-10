import argparse
import csv
import json
import random
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


def load_id_genre_mapping(path: Path) -> dict[str, str]:
    mapping: dict[str, str] = {}
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is None or not {"id", "genre"}.issubset(reader.fieldnames):
            raise ValueError(f"{path} must have 'id' and 'genre' columns")

        for line_number, row in enumerate(reader, start=2):
            record_id = (row.get("id") or "").strip()
            genre = (row.get("genre") or "").strip()
            if not record_id or not genre:
                raise ValueError(f"Missing id or genre in {path} at line {line_number}")
            if record_id in mapping and mapping[record_id] != genre:
                raise ValueError(
                    f"Conflicting genres for id={record_id}: "
                    f"{mapping[record_id]!r} and {genre!r}"
                )
            mapping[record_id] = genre

    return mapping


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line_number, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"Invalid JSON in {path} at line {line_number}: {exc}"
                ) from exc
            if "id" not in record:
                raise ValueError(f"Missing id in {path} at line {line_number}")
            records.append(record)
    return records


def sample_by_genre(
    records: list[dict[str, Any]],
    mapping: dict[str, str],
    max_per_genre: int,
    seed: int,
) -> tuple[list[dict[str, Any]], Counter[str]]:
    indices_by_genre: dict[str, list[int]] = defaultdict(list)
    missing_ids: list[str] = []

    for index, record in enumerate(records):
        record_id = str(record["id"])
        genre = mapping.get(record_id)
        if genre is None:
            missing_ids.append(record_id)
            continue
        indices_by_genre[genre].append(index)

    if missing_ids:
        preview = ", ".join(missing_ids[:10])
        suffix = " ..." if len(missing_ids) > 10 else ""
        raise ValueError(
            f"{len(missing_ids)} input ids are missing from the mapping: "
            f"{preview}{suffix}"
        )

    rng = random.Random(seed)
    selected_indices: set[int] = set()
    sampled_counts: Counter[str] = Counter()
    for genre in sorted(indices_by_genre):
        indices = indices_by_genre[genre]
        selected = rng.sample(indices, k=min(len(indices), max_per_genre))
        selected_indices.update(selected)
        sampled_counts[genre] = len(selected)

    sampled_records = [
        record for index, record in enumerate(records) if index in selected_indices
    ]
    return sampled_records, sampled_counts


def write_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_name(f".{path.name}.tmp")
    try:
        with temporary_path.open("w", encoding="utf-8", newline="\n") as f:
            for record in records:
                f.write(json.dumps(record, ensure_ascii=False))
                f.write("\n")
        temporary_path.replace(path)
    finally:
        temporary_path.unlink(missing_ok=True)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Sample at most N records per genre from an amalgum JSONL file."
    )
    parser.add_argument(
        "--input",
        default="dataset/doccano/input/amalgum.jsonl",
        help="Input JSONL path.",
    )
    parser.add_argument(
        "--mapping",
        default="json/amalgum_1300_id_genre.csv",
        help="CSV path containing id and genre columns.",
    )
    parser.add_argument(
        "--output",
        default="dataset/doccano/input/amalgum_max200_per_genre.jsonl",
        help="Output JSONL path.",
    )
    parser.add_argument(
        "--max-per-genre",
        type=int,
        default=200,
        help="Maximum records to sample from each genre.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed used for reproducible sampling.",
    )
    args = parser.parse_args()

    if args.max_per_genre <= 0:
        parser.error("--max-per-genre must be greater than zero")

    input_path = Path(args.input)
    mapping_path = Path(args.mapping)
    output_path = Path(args.output)

    mapping = load_id_genre_mapping(mapping_path)
    records = load_jsonl(input_path)
    sampled_records, sampled_counts = sample_by_genre(
        records,
        mapping,
        args.max_per_genre,
        args.seed,
    )
    write_jsonl(output_path, sampled_records)

    print(f"Wrote {len(sampled_records)} records to {output_path}")
    for genre in sorted(sampled_counts):
        print(f"{genre}: {sampled_counts[genre]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
