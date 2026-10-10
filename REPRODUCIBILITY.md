# Reproducibility guide

This document describes the data-construction and experiment pipeline used for
the negation-focus dataset. Run commands from the repository root.

## Reproducibility levels

There are two distinct levels of reproduction.

1. **Annotation input to evaluation:** deterministic except for model inference.
   Prompt creation, vote merging, the random baseline, and evaluation can be
   reproduced from the sampled doccano input and exported annotations.
2. **Source TSV to annotation input:** requires the source CoNLL-style TSV and an
   external sentence-rewriting step through the OpenAI Batch API. Model output
   can change over time, so this stage is procedural rather than bit-for-bit
   reproducible unless the archived rewrite responses are used.

The final public data additionally requires the exported annotations from all
annotators. These exports are inputs to the vote-merging script and are not model
outputs.

## Environment

- Python 3.12 or later
- `OPENAI_API_KEY` for the historical sentence-rewriting stage
- Linux, a CUDA-capable GPU, and vLLM for local-model experiments
- Provider API keys when using the optional Batch API experiment path

Install [uv](https://docs.astral.sh/uv/getting-started/installation/), then
create the project environment and install the exact locked dependencies:

```bash
uv sync --frozen
```

For local vLLM inference, include the optional CUDA-compatible dependencies:

```bash
uv sync --frozen --extra vllm
```

The commands below use `python` for readability. After syncing, either activate
the environment with `source .venv/bin/activate` (Linux/macOS) or prefix each
command with `uv run` (for example, `uv run python scripts/make_json.py ...`).

## A. Source TSV to doccano input

### A0. Prepare the SRL-enriched source file

The historical `amalgum_1300.merged` input used in this work was prepared with
[LLM-SRL](https://github.com/fangfang123gh/LLM-SRL). The LoRA training
configuration is included at [`LLM-SRL/lora_config.yaml`](LLM-SRL/lora_config.yaml),
and the trained PEFT/LoRA adapter weights, tokenizer files, and retained
checkpoints are stored under [`LLM-SRL/model/`](LLM-SRL/model/). The adapter is
based on Meta-Llama-3-8B-Instruct; the base model itself is not included in this
repository and remains subject to its own license and access conditions.

Use the upstream LLM-SRL implementation together with these artifacts for the
SRL stage. The commands below start after that stage and assume that the
resulting `amalgum_1300.merged` file is available locally.

### A1. Extract negation-focus candidates from the TSV

The input is a blank-line-separated, tab-delimited corpus in the layout handled
by `make_json.py`. The historical AMALGUM input was named
`amalgum_1300.merged`, but it is structurally a TSV file.

```bash
python scripts/make_json.py \
  --input path/to/amalgum_1300.merged \
  --output imtermidiate/json/amalgum_1300.json
```

Output: `imtermidiate/json/amalgum_1300.json`.

### A2. Generate synthetic focus alternatives

```bash
python scripts/convert.py \
  --input_json imtermidiate/json/amalgum_1300.json \
  --output_path imtermidiate/converted/amalgum_1300_converted.json
```

### A3. Prepare and run the sentence-rewriting batches

Create OpenAI Batch request files:

```bash
python scripts/make_training_rewrite_batch_inputs.py \
  --input imtermidiate/converted/amalgum_1300_converted.json \
  --output-dir imtermidiate/rewrite_batch \
  --prefix positive_batch_input_ \
  --batch-size 100 \
  --model gpt-5
```

This also creates `imtermidiate/rewrite_batch/batch_state.json`. If the state
file already exists, it is preserved to avoid losing submitted batch IDs.

Submit one batch index at a time. Repeat `order` while another numbered input
file exists, then run `check` until the submitted batches finish:

```bash
python scripts/llm_rewrite_training_order.py \
  --mode order \
  --batch-input-dir imtermidiate/rewrite_batch \
  --batch-id-path imtermidiate/rewrite_batch/batch_state.json

python scripts/llm_rewrite_training_order.py \
  --mode check \
  --batch-id-path imtermidiate/rewrite_batch/batch_state.json \
  --output imtermidiate/rewrote/amalgum.jsonl
```

For exact historical reconstruction, use archived rewrite responses instead of
resubmitting the requests.

### A4. Build and sample the doccano input

```bash
python scripts/make_doccano_training_rewrite_sentence_input.py \
  --converted imtermidiate/converted/amalgum_1300_converted.json \
  --rewrote imtermidiate/rewrote/amalgum.jsonl \
  --output doccano/input/amalgum.jsonl

python scripts/sample_amalgum_by_genre.py \
  --input doccano/input/amalgum.jsonl \
  --mapping resources/amalgum_1300_id_genre.csv \
  --output doccano/input/amalgum_max200_per_genre.jsonl \
  --max-per-genre 200 \
  --seed 42
```

The mapping file and random seed are versioned in this repository.

### A5. Annotate and merge the votes

Import the sampled JSONL from A4 into doccano as a single-label classification
task. Export one JSONL file per annotator using the filename pattern
`NF_annotator_*.jsonl`.

Generate the majority-vote labels:

```bash
python scripts/merge_doccano_annotation_votes.py \
  --input-path doccano/input/amalgum_max200_per_genre.jsonl \
  --output-dir path/to/annotator_exports \
  --output-path doccano/merged/NF_merged_majority_vote.jsonl
```

Generate labels requiring agreement from all three annotators. The script adds
`_min_agreement_3` to the output filename and assigns `TIE` when fewer than
three annotators select the same label:

```bash
python scripts/merge_doccano_annotation_votes.py \
  --input-path doccano/input/amalgum_max200_per_genre.jsonl \
  --output-dir path/to/annotator_exports \
  --output-path doccano/merged/NF_merged_majority_vote.jsonl \
  --min-agreement 3
```

## B. Annotation input to experiment prompts

Create model-independent prompts from the sampled doccano input produced in A4:

```bash
python scripts/make_amalgum_experiment_json.py \
  --input-path doccano/input/amalgum_max200_per_genre.jsonl \
  --template templates/template_w_x.txt \
  --output-path prompt/amalgum_with_x.json
```

For the condition without the `x` option:

```bash
python scripts/make_amalgum_experiment_json.py \
  --input-path doccano/input/amalgum_max200_per_genre.jsonl \
  --exclude-x \
  --template templates/template_wo_x.txt \
  --output-path prompt/amalgum_without_x.json
```

## C. Run an experiment

### C1. Deterministic random baseline

```bash
python scripts/random_baseline.py \
  --input prompt/amalgum_with_x.json \
  --output experiment/vllm/random-baseline_seed42_amalgum_with_x.jsonl \
  --seed 42 \
  --overwrite
```

For the condition without the `x` option:

```bash
python scripts/random_baseline.py \
  --exclude-x \
  --input prompt/amalgum_without_x.json \
  --output experiment/vllm/random-baseline_seed42_amalgum_without_x.jsonl \
  --seed 42 \
  --overwrite
```

### C2. Local model with vLLM

```bash
python scripts/llm_experiment_vllm.py \
  --model Qwen/Qwen3.5-0.8B \
  --input prompt/amalgum_with_x.json \
  --output experiment/vllm/Qwen_Qwen3.5-0.8B_amalgum_with_x.jsonl \
  --batch-size 32 \
  --max-model-len 8192 \
  --overwrite
```

For the condition without the `x` option:

```bash
python scripts/llm_experiment_vllm.py \
  --model Qwen/Qwen3.5-0.8B \
  --exclude-x \
  --input prompt/amalgum_without_x.json \
  --output experiment/vllm/Qwen_Qwen3.5-0.8B_amalgum_without_x.jsonl \
  --batch-size 32 \
  --max-model-len 8192 \
  --overwrite
```

Use `--limit 5` for a smoke run. Record the resolved model revision rather than
relying only on a mutable model name.

### C3. Optional provider Batch APIs

Create provider-specific request files:

```bash
python scripts/make_multi_provider_experiment_batch_inputs.py \
  --input prompt/amalgum_with_x.json \
  --output-dir experiment/batch \
  --providers openai anthropic gemini
```

Create request files for the condition without the `x` option:

```bash
python scripts/make_multi_provider_experiment_batch_inputs.py \
  --exclude-x \
  --input prompt/amalgum_without_x.json \
  --output-dir experiment/batch \
  --providers openai anthropic gemini
```

These commands also create `experiment/batch/batch_state.json`. If the state
file already exists, it is preserved to avoid losing submitted batch IDs.

Inspect the generated requests before submission. Submit the next numbered
input file for each provider/model with:

```bash
python scripts/llm_experiment_batch_order.py \
  --mode order \
  --provider openai \
  --model gpt-5.5 \
  --batch-input-root experiment/batch \
  --batch-state-path experiment/batch/batch_state.json

python scripts/llm_experiment_batch_order.py \
  --mode order \
  --provider anthropic \
  --model claude-opus-4-8 \
  --batch-input-root experiment/batch \
  --batch-state-path experiment/batch/batch_state.json

python scripts/llm_experiment_batch_order.py \
  --mode order \
  --provider gemini \
  --model gemini-3.5-flash \
  --batch-input-root experiment/batch \
  --batch-state-path experiment/batch/batch_state.json
```

Submit the corresponding request files for the condition without the `x`
option with:

```bash
python scripts/llm_experiment_batch_order.py \
  --mode order \
  --exclude-x \
  --provider openai \
  --model gpt-5.5 \
  --batch-input-root experiment/batch \
  --batch-state-path experiment/batch/batch_state.json

python scripts/llm_experiment_batch_order.py \
  --mode order \
  --exclude-x \
  --provider anthropic \
  --model claude-opus-4-8 \
  --batch-input-root experiment/batch \
  --batch-state-path experiment/batch/batch_state.json

python scripts/llm_experiment_batch_order.py \
  --mode order \
  --exclude-x \
  --provider gemini \
  --model gemini-3.5-flash \
  --batch-input-root experiment/batch \
  --batch-state-path experiment/batch/batch_state.json
```

Repeat the corresponding `--mode order` command while another numbered input
file exists. Check status and retrieve completed results with:

```bash
python scripts/llm_experiment_batch_order.py \
  --mode check \
  --provider openai \
  --model gpt-5.5 \
  --batch-state-path experiment/batch/batch_state.json \
  --output-dir experiment/batch/results

python scripts/llm_experiment_batch_order.py \
  --mode check \
  --provider anthropic \
  --model claude-opus-4-8 \
  --batch-state-path experiment/batch/batch_state.json \
  --output-dir experiment/batch/results

python scripts/llm_experiment_batch_order.py \
  --mode check \
  --provider gemini \
  --model gemini-3.5-flash \
  --batch-state-path experiment/batch/batch_state.json \
  --output-dir experiment/batch/results
```

The same `--mode check` commands retrieve both `with_x` and `without_x`
batches recorded in the state file.

These commands perform paid external API operations.

## D. Evaluate predictions

Prediction filenames must contain `with_x` or `without_x`, matching the mode.
Gold `TIE` records are excluded from accuracy. With `--exclude-x`, gold label
`x` is also excluded.

```bash
python scripts/calculate_vllm_accuracy.py \
  --gold-path doccano/merged/NF_merged_majority_vote.jsonl \
  --results-dir experiment/vllm \
  --batch-results-dir experiment/batch/results
```
