# Negation Focus Annotation Dataset

This repository contains an English dataset for selecting the logically correct
focus interpretation of a negated sentence. Each item presents a three-sentence
context and several rewritten alternatives. Three annotators selected the
alternative that is logically supported by the context, or `x` when none of the
alternatives is supported. The detail is described in [PAPER_URL]

The release contains 966 items sampled from five genres of the
[AMALGUM corpus](https://github.com/gucorpling/amalgum), together with scripts
for data preparation, annotation-vote merging, prompt generation, inference,
and evaluation.

## Data

Data are organized by genre under `data/`:

| Directory         | AMALGUM genre | Source website | Records |
| ----------------- | ------------- | -------------- | ------: |
| `biography/`    | `bio`       | Wikipedia      |     166 |
| `how_to_guide/` | `whow`      | wikiHow        |     200 |
| `interview/`    | `interview` | Wikinews       |     200 |
| `news/`         | `news`      | Wikinews       |     200 |
| `travel_guide/` | `voyage`    | Wikivoyage     |     200 |

Each genre contains two label editions with the same 966 record IDs:

- `majority_vote.jsonl`: the unique plurality label from three annotators;
  `TIE` is used when no label has a unique plurality.
- `unanimous.jsonl`: the shared label only when all three annotators agree;
  otherwise `TIE`.

Each JSONL record contains `id`, `genre`, the task text in `data`, the selected
`label`, per-label `votes`, and source/licensing metadata. See
[`DATA_CARD.md`](DATA_CARD.md) for full documentation and
[`schemas/dataset-record.schema.json`](schemas/dataset-record.schema.json) for
the machine-readable schema.

## Annotation task

Annotators judge only from the supplied context. They choose the logically
correct option closest to the context; if no option is supported, they choose
`x` (None of the above). The complete instructions and examples are in
[`annotation_guideline/ANNOTATION_GUIDELINES.md`](annotation_guideline/ANNOTATION_GUIDELINES.md)
and its
[`PDF edition`](annotation_guideline/ANNOTATION_GUIDELINES.pdf).

## Reading the data

Each line is an independent JSON object:

```python
import json
from pathlib import Path

path = Path("data/biography/majority_vote.jsonl")
with path.open(encoding="utf-8") as stream:
    records = [json.loads(line) for line in stream if line.strip()]

print(records[0]["id"], records[0]["label"])
```

## Reproduction and experiments

The complete pipeline—from the source TSV through prompt creation, model
inference, and evaluation—is documented in
[`REPRODUCIBILITY.md`](REPRODUCIBILITY.md). It requires Python 3.12 or later.
After installing [uv](https://docs.astral.sh/uv/getting-started/installation/),
create the locked common environment with:

```bash
uv sync --frozen
```

Local GPU inference has additional locked dependencies:

```bash
uv sync --frozen --extra vllm
```

Some reproduction stages use paid external APIs. Review generated requests and
the notes in `REPRODUCIBILITY.md` before submitting them.

## Licenses

This repository contains material under multiple licenses. The source-text
license varies by genre, and the wikiHow subset is restricted to
non-commercial use. There is no single blanket license for the entire
repository. The newly created negation-focus labels and original code are
licensed under [CC BY 4.0](LICENSE.txt). Read [`LICENSES.md`](LICENSES.md) and
the `LICENSE.txt` file inside each genre directory before reuse.

## Associated paper

A paper describing this dataset is currently under review. Bibliographic
details and links will be added after the review and publication process.

<!--
Publication metadata placeholder. Keep this block commented out during
anonymous review. After publication, replace the placeholders and remove the
comment markers.

### [Paper title]

[Author 1], [Author 2], and [Author 3]. "[Paper title]." In *[Conference or
workshop name]*, [year].

- Paper: [publisher or proceedings URL]
- Preprint: [preprint URL]
- DOI: [DOI URL]
- Project page: [project URL]

```bibtex
@inproceedings{citation-key,
  title     = {[Paper title]},
  author    = {[Author 1] and [Author 2] and [Author 3]},
  booktitle = {[Conference or workshop name]},
  year      = {[year]},
  url       = {[paper URL]},
  doi       = {[DOI]}
}
```
-->

## Citation

Citation metadata are provided in [`CITATION.cff`](CITATION.cff). The repository
does not yet have a DOI; use the GitHub repository URL and the release tag when
citing a specific version.
