# Dataset card

## Dataset summary

The Negation Focus Annotation Dataset is an English multiple-choice dataset for
evaluating which focus interpretation of a negated sentence is logically
supported by a three-sentence context. Candidate alternatives were constructed
from negation-focus candidates extracted from AMALGUM and include a
none-of-the-above option.

- Language: English
- Records: 966
- Annotators per record: 3
- Genres: biography, how-to guide, interview, news, and travel guide
- Format: UTF-8 JSON Lines
- Upstream corpus revision: `a2dbd33e757f75a61d68d22d1a29e840bdffa711`
- Splits: none

## Dataset structure

Each genre directory contains `majority_vote.jsonl`, `unanimous.jsonl`, and the
license text governing that genre's source text. The two editions contain the
same records in the same order and differ only in `label`.

| Genre | Records | Majority `TIE` | Unanimous `TIE` |
| --- | ---: | ---: | ---: |
| Biography | 166 | 18 | 138 |
| How-to guide | 200 | 37 | 166 |
| Interview | 200 | 35 | 165 |
| News | 200 | 37 | 163 |
| Travel guide | 200 | 31 | 149 |
| **Total** | **966** | **158** | **781** |

The high number of unanimous-edition `TIE` labels reflects the strict
three-of-three agreement criterion; it does not indicate missing annotation.

### Record fields

- `id`: integer identifier shared across label editions.
- `genre`: normalized public genre name.
- `data`: the complete annotation item, including `Context`, `Options`, and the
  `x. None of the above` option.
- `label`: a one-element array containing `a`–`f`, `x`, or `TIE`.
- `votes`: counts for labels selected by the three annotators.
- `source`: AMALGUM provenance, source URL, upstream revision, and license
  metadata.

The exact constraints are defined in
[`schemas/dataset-record.schema.json`](schemas/dataset-record.schema.json).

## Label construction

For `majority_vote.jsonl`, the label with the largest vote count is used only
when it is the unique winner. If multiple labels share the largest count, the
record receives `TIE`.

For `unanimous.jsonl`, a label is used only when all three annotators selected
it. Every other record receives `TIE`.

The annotation policy asks annotators to rely solely on the supplied context,
reject options that conflict with it or cannot be established from it, and use
`x` when no option is logically correct. See
[`annotation_guideline/ANNOTATION_GUIDELINES.md`](annotation_guideline/ANNOTATION_GUIDELINES.md).

## Source and construction

1. Negation-focus candidates were extracted from a CoNLL-style AMALGUM TSV.
2. Synthetic focus alternatives were constructed by removing or relocating the
   negation focus.
3. Candidate sentences were rewritten through an OpenAI Batch API workflow to
   improve naturalness while retaining their intended meaning.
4. At most 200 records were sampled per included genre with random seed 42.
5. Three annotators independently selected a label in doccano.
6. Votes were merged into the majority-vote and unanimous editions.
7. Source metadata were matched to AMALGUM records using the ordered three
   context sentences; provenance is stored in each released record.

The procedural details and commands are in
[`REPRODUCIBILITY.md`](REPRODUCIBILITY.md). Exact reconstruction of the
sentence-rewriting stage requires the archived model responses because API
model outputs may change over time.

## Intended uses

Suitable uses include:

- evaluating negation-focus and contextual logical-inference systems;
- analyzing human agreement on focus alternatives;
- comparing prompting or inference methods on a fixed item set; and
- linguistic research on negation across web genres.

The dataset is not designed for high-stakes decision making, factuality
assessment, or measuring general reasoning ability without considering its
construction process and genre coverage.

## Limitations

- The dataset covers English and five AMALGUM genres only.
- Four genres were capped at 200 records; the sample is not representative of
  the full web or of English generally.
- Candidate alternatives are synthetic and include model-assisted rewrites, so
  artifacts may make some options easier or harder to identify.
- Inter-annotator agreement is low.
- `TIE` has different semantics from `x`: `TIE` is an aggregation outcome,
  while `x` is an annotator-selectable none-of-the-above answer.
- Source pages may contain names, events, opinions, or other information about
  real people. No claim is made that the data are free of personal or sensitive
  information.
- Source URLs and page contents can change after AMALGUM collection.

Users should report results separately by label edition and disclose whether
`TIE` and `x` records were excluded.

## Licensing and attribution

Licensing is layered and varies by genre. In particular, the how-to guide data
derive from wikiHow content under CC BY-NC-SA 3.0. See
[`LICENSES.md`](LICENSES.md) and each genre's `LICENSE.txt`; these notices are
part of the dataset and should be retained with redistributed copies.

The newly created negation-focus `label` and `votes` fields and the original
code are licensed under
[Creative Commons Attribution 4.0 International](LICENSE.txt). This grant does
not replace the licenses governing source text or AMALGUM annotation layers.

## Maintenance

Issues should be reported through the GitHub repository. Future releases should
identify the release tag and document any changes to records, labels, source
metadata, or licensing information.
