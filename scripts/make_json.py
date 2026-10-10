from __future__ import annotations

import argparse
from static import *
import json
import re
from pathlib import Path


CONTEXT_MARKERS = {"previous", "target", "subsequent"}


class CorpusLayout:
    def __init__(
        self,
        *,
        semantic_role_start: int,
        named_entity_column: int,
        head_column: int,
        dependency_column: int,
        context_marker_column: int | None = None,
        focus_column: int | None = None,
    ) -> None:
        self.semantic_role_start = semantic_role_start
        self.named_entity_column = named_entity_column
        self.head_column = head_column
        self.dependency_column = dependency_column
        self.context_marker_column = context_marker_column
        self.focus_column = focus_column


def read_tsv_corpus(path: Path) -> list[list[list[str]]]:
    sentences: list[list[list[str]]] = []
    sentence: list[list[str]] = []
    with path.open("r", encoding="utf-8") as f:
        for line_number, line in enumerate(f, start=1):
            line = line.rstrip("\n")
            if not line.strip():
                if sentence:
                    sentences.append(sentence)
                    sentence = []
                continue

            columns = line.split("\t")
            if len(columns) < START_OF_SEMANTIC_ROLE_COLUMN + 1:
                raise ValueError(
                    f"{path}: line {line_number} has too few columns for semantic role data"
                )
            sentence.append(columns)

    if sentence:
        sentences.append(sentence)
    return sentences


def detect_layout(sentences: list[list[list[str]]]) -> CorpusLayout:
    first_token = next((sentence[0] for sentence in sentences if sentence), None)
    if first_token is None:
        raise ValueError("input corpus is empty")

    if len(first_token) > 9 and first_token[8] in CONTEXT_MARKERS:
        return CorpusLayout(
            semantic_role_start=9,
            named_entity_column=4,
            head_column=5,
            dependency_column=6,
            context_marker_column=8,
            focus_column=len(first_token) - 1,
        )

    return CorpusLayout(
        semantic_role_start=START_OF_SEMANTIC_ROLE_COLUMN,
        named_entity_column=3,
        head_column=6,
        dependency_column=DEPENDENCY_COLUMN,
        focus_column=len(first_token) - 1,
    )


def normalize_named_entity(named_entity: str) -> str:
    return "O" if named_entity == "-" else named_entity


def sentence_text(sentence: list[list[str]]) -> str:
    return " ".join([word[0] for word in sentence])


def iter_target_sentences(
    sentences: list[list[list[str]]],
    layout: CorpusLayout,
    skip_no_focus: bool,
) -> list[tuple[int, list[list[str]], str, str]]:
    items: list[tuple[int, list[list[str]], str, str]] = []
    if layout.context_marker_column is not None:
        for i, sentence in enumerate(sentences):
            marker = sentence[0][layout.context_marker_column]
            if marker != "target":
                continue
            if skip_no_focus and layout.focus_column is not None:
                if not any(word[layout.focus_column] == "Y" for word in sentence):
                    continue
            previous_sentence = sentence_text(sentences[i - 1]) if i > 0 else ""
            subsequent_sentence = sentence_text(sentences[i + 1]) if i + 1 < len(sentences) else ""
            items.append((i, sentence, previous_sentence, subsequent_sentence))
        return items

    for i in range(1, len(sentences) - 1, 3):
        sentence = sentences[i]
        if skip_no_focus and layout.focus_column is not None:
            if not any(word[layout.focus_column] == "FOCUS" for word in sentence):
                continue
        previous_sentence = sentence_text(sentences[i - 1]) if i > 0 else ""
        subsequent_sentence = sentence_text(sentences[i + 1]) if i + 1 < len(sentences) else ""
        items.append((i, sentence, previous_sentence, subsequent_sentence))
    return items


def make_json(input_path: Path, output_path: Path, skip_no_focus: bool = False) -> None:
    sentences = read_tsv_corpus(input_path)
    layout = detect_layout(sentences)
    json_data: list[dict] = []
    sentences_info = []
    for i, sentence, previous_sentence, subsequent_sentence in iter_target_sentences(sentences, layout, skip_no_focus):
        focus_column_count = 1 if layout.focus_column is not None else 0
        verb_num = len(sentence[0]) - layout.semantic_role_start - focus_column_count
        if verb_num <= 0:
            continue

        semantic_role_dict = {f"column_{j}": [] for j in range(verb_num)}
        open_roles = {f"column_{j}": [] for j in range(verb_num)}

        for j, word in enumerate(sentence):
            # semantic role 列を走査し、(ROLE* ... *) 形式を [start, end] に変換する
            for col in range(verb_num):
                cell = word[layout.semantic_role_start + col]
                column_key = f"column_{col}"

                started_roles = re.findall(r"\(([^()*]+)\*", cell)
                for semantic_role in started_roles:
                    open_roles[column_key].append({"semantic_role": semantic_role, "start": j})

                for _ in range(cell.count(")")):
                    if not open_roles[column_key]:
                        continue
                    role_info = open_roles[column_key].pop()
                    semantic_role_dict[column_key].append({
                        "semantic_role": role_info["semantic_role"],
                        "section": [role_info["start"], j],
                    })

        # 閉じ忘れがあれば文末までを span とみなす
        for col in range(verb_num):
            column_key = f"column_{col}"
            while open_roles[column_key]:
                role_info = open_roles[column_key].pop()
                semantic_role_dict[column_key].append({
                    "semantic_role": role_info["semantic_role"],
                    "section": [role_info["start"], len(sentence) - 1],
                })

        for j, column in enumerate(semantic_role_dict.keys()):
            semantic_role_info = semantic_role_dict[column]
            if not any(record["semantic_role"] == "V" for record in semantic_role_info):
                continue

            verb_idx = next(record for record in semantic_role_info if record["semantic_role"] == "V")["section"][0] + 1
            semantic_roles = [record["semantic_role"] for record in semantic_role_info if record["semantic_role"] != "V"]
            for role in semantic_roles:
                if role not in CONVERATABLE_SEMANTIC_ROLES:
                    continue

                focus_role_sections = [record["section"] for record in semantic_role_info if record["semantic_role"] == role]
                if not focus_role_sections:
                    continue
                role_sections = [record["section"] for record in semantic_role_info ]
                words_info = []
                focus_named_entity = "O"
                exist_negation = False
                for k, word in enumerate(sentence):
                    if any(start <= k <= end for start, end in focus_role_sections):
                        mark = "focus"
                        named_entity = normalize_named_entity(word[layout.named_entity_column])
                        if focus_named_entity == "O" and named_entity != "O":
                            focus_named_entity = named_entity
                    elif word[layout.dependency_column].startswith("aux") and word[layout.head_column] == str(verb_idx):
                        mark = "auxiliary"
                    elif any(
                        negation_role in word[layout.semantic_role_start + j]
                        for negation_role in ("ARGM-NEG", "AM-NEG")
                    ):
                        mark = "negation"
                        exist_negation = True
                    elif k == verb_idx - 1:
                        mark = "verb"
                    else:
                        mark = "other"
                    words_info.append({
                        "word": word[0],
                        "mark": mark,
                        "named_entity": normalize_named_entity(word[layout.named_entity_column]),
                    })
                if exist_negation:
                    end_of_focus = max(end for _, end in role_sections)
                    words_info.insert(end_of_focus + 1, {
                        "word": "",
                        "mark": "end_of_focus",
                        "named_entity": "O",
                    })

                    sentences_info.append({
                        "id": i,
                        "sentence": words_info,
                        "semantic_role_of_focus": role,
                        "focus_named_entity": focus_named_entity,
                        "previous_sentence": previous_sentence,
                        "subsequent_sentence": subsequent_sentence,
                    })
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        json.dump({"infos": sentences_info}, f)


def make_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create semantic role focus JSON from a tab-delimited CoNLL-style TSV file."
    )
    parser.add_argument(
        "--input",
        default="dataset/conll2012_train_first3000.tsv",
        help="Tab-delimited input corpus.",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Output JSON path. Defaults to json/{input_stem}_positive.json.",
    )
    parser.add_argument(
        "--skip-no-focus",
        action="store_true",
        help="Skip sentences whose final column does not contain FOCUS, for older PB-FOC merged files.",
    )
    return parser.parse_args()


if __name__ == '__main__':
    args = make_args()
    input_path = Path(args.input)
    output_path = Path(args.output) if args.output else Path("json") / f"{input_path.stem}_positive.json"
    make_json(input_path, output_path, skip_no_focus=args.skip_no_focus)
