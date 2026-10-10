from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

from experiment_paths import PROMPT_DIR


NONE_OF_THE_ABOVE_LINE = re.compile(
    r"(?m)^[ \t]*x\.[ \t]*None of the above[ \t]*\r?\n?"
)


def read_jsonl(path: Path) -> list[dict]:
    records: list[dict] = []
    with path.open("r", encoding="utf-8-sig") as input_file:
        for line_number, line in enumerate(input_file, start=1):
            if not line.strip():
                continue

            record = json.loads(line)
            if "id" not in record or "text" not in record:
                raise ValueError(
                    f"{path}:{line_number}: each record must contain id and text"
                )
            records.append(record)

    return records


def remove_none_of_the_above(text: str) -> str:
    cleaned_text, count = NONE_OF_THE_ABOVE_LINE.subn("", text)
    if count != 1:
        raise ValueError(
            "expected exactly one 'x. None of the above' option, "
            f"but found {count}"
        )
    return cleaned_text.rstrip()


def build_experiment_records(
    input_path: Path, template: str, *, exclude_x: bool
) -> list[dict]:
    experiment_records: list[dict] = []

    for record in read_jsonl(input_path):
        text = record["text"]
        if exclude_x:
            text = remove_none_of_the_above(text)
        experiment_records.append(
            {
                "id": record["id"],
                "prompt": template.replace("{text}", text),
            }
        )

    return experiment_records


def write_json(path: Path, records: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as output_file:
        json.dump(records, output_file, ensure_ascii=False, indent=2)
        output_file.write("\n")


def resolve_template_path(
    template_path: Path | None, *, exclude_x: bool
) -> Path:
    if template_path is not None:
        return template_path
    if exclude_x:
        return Path("dataset/templates/template_wo_x.txt")
    return Path("dataset/templates/template_w_x.txt")


def resolve_output_path(
    output_dir: Path,
    template_path: Path,
    output_path: Path | None,
) -> Path:
    if output_path is not None:
        return output_path

    template_name = template_path.stem
    if template_name.startswith("template_"):
        template_name = template_name.removeprefix("template_")
    output_suffix = {
        "w_x": "with_x",
        "wo_x": "without_x",
        "w_x_reason": "with_x_reason",
        "wo_x_reason": "without_x_reason",
    }.get(template_name, template_name)
    return output_dir / f"amalgum_{output_suffix}.json"


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Create model-independent experiment JSON files from doccano "
            "inputs using a prompt template."
        )
    )
    parser.add_argument(
        "--input-path",
        default="dataset/doccano/input/amalgum_max200_per_genre.jsonl",
    )
    parser.add_argument(
        "--output-dir",
        default=str(PROMPT_DIR),
    )
    parser.add_argument(
        "--output-path",
        type=Path,
        default=None,
        help=(
            "Exact output JSON path. When omitted, derive the filename from "
            "the template filename and place it under --output-dir."
        ),
    )
    parser.add_argument(
        "--template-path",
        "--template",
        type=Path,
        default=None,
        help=(
            "Prompt template path. When omitted, use template_w_x.txt or "
            "template_wo_x.txt according to --exclude-x."
        ),
    )
    parser.add_argument(
        "--exclude-x",
        action="store_true",
        help="Remove the 'x. None of the above' option from each prompt text.",
    )
    args = parser.parse_args()

    input_path = Path(args.input_path)
    if not input_path.exists():
        raise FileNotFoundError(f"input file not found: {args.input_path}")

    exclude_x = args.exclude_x
    template_path = resolve_template_path(
        args.template_path,
        exclude_x=exclude_x,
    )
    if not template_path.exists():
        raise FileNotFoundError(f"template file not found: {template_path}")
    template = template_path.read_text(encoding="utf-8-sig")
    if template.count("{text}") != 1:
        raise ValueError(
            f"{template_path}: template must contain exactly one {{text}} placeholder"
        )

    output_dir = Path(args.output_dir)
    records = build_experiment_records(
        input_path, template, exclude_x=exclude_x
    )
    output_path = resolve_output_path(
        output_dir,
        template_path,
        args.output_path,
    )
    write_json(output_path, records)
    print(f"{input_path} -> {output_path} ({len(records)} records)")


if __name__ == "__main__":
    main()
