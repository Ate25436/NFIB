# Synthetic merged-corpus fixture

`amalgum_1300.synthetic.merged` is a small, entirely synthetic input fixture
that follows the 11-column layout of `dataset/merged/amalgum_1300.merged`. It
does not copy sentences from AMALGUM or from the released dataset.

The fixture contains three `previous`/`target`/`next` context triples. Its
target sentences cover:

- `not` with a `did` auxiliary and `A0`/`A1` arguments;
- `not` with `A0`/`A1` and `AM-TMP` arguments; and
- `never` with `A0` and `AM-LOC` arguments.

The columns are, in order: token, token ID, lemma, UPOS, named-entity tag,
dependency head, dependency relation, reserved field, context marker, one SRL
column, and the historical predicate/focus flag.

From the repository root, the first two construction stages can be checked
without external services:

```bash
python scripts/make_json.py \
  --input tests/fixtures/amalgum_1300.synthetic.merged \
  --output tmp/e2e/synthetic.json

python scripts/convert.py \
  --input_json tmp/e2e/synthetic.json \
  --output_path tmp/e2e/synthetic_converted.json
```

Expected result: `make_json.py` writes seven SRL-focus candidates and
`convert.py` writes seven converted records.
