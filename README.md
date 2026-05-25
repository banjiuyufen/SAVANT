# SAVANT

This is the anonymized reviewer-release package for SAVANT, an evidence-grounded generator and verifier for structured vaccine-adjuvant designs. The release contains the main generation code, verification code, the SAVANT v1 formal theorem library, and one command-line runner. Internal experiment scripts and non-anonymous local paths are intentionally excluded.

## Contents

- `run_savant.py`: end-to-end command-line entry point for generation plus verification or verification-only runs.
- `savant/generation/`: LLM generator and strict structured-design schemas.
- `savant/grounding/`: predicate representation, unit parsing, functor vocabulary, and ontology dataclasses.
- `savant/theorem/`: theorem database and vector retrieval utilities.
- `savant/verification/`: three-stage evidence matching and verification helpers.
- `theorem_library/`: SAVANT v1 formal theorem JSON files, prebuilt FAISS indexes, and ontology/prior files used by the verifier.
- `requirements.txt`: Python package dependencies.

## Setup

Install the dependencies in a Python environment with FAISS support:

```bash
pip install -r requirements.txt
```

The runner uses a SentenceTransformer-compatible embedding model. By default it refers to `FremyCompany/BioLORD-2023-M`. For an offline review environment, provide a local model path:

```bash
export SAVANT_EMBEDDING_MODEL=/path/to/local/embedding-model
export SAVANT_TOKENIZER_MODEL=/path/to/local/embedding-model
```

The generator can run in three modes:

- `api`: calls an OpenAI-compatible chat completion endpoint. Set `OPENAI_API_KEY` and optionally `OPENAI_BASE_URL`, or pass `--api-key` and `--base-url`.
- `local`: uses a local HuggingFace-compatible causal language model with Outlines FSM constraints.
- `mock`: returns a deterministic example design for smoke tests and package checks.

## Verify Structured Input

```bash
python run_savant.py \
  --input /path/to/structured_designs.jsonl \
  --qid sample_id \
  --output outputs/savant_results.json
```

For a JSONL/list input, use `--qid` or `--row-index` to select a design. A single JSON file can be passed directly.

## Generate And Verify

```bash
python run_savant.py \
  --query "Design a particulate adjuvant strategy for a protein antigen vaccine." \
  --generator-mode api \
  --generator-model gpt-4-turbo \
  --output outputs/savant_generated_result.json
```

For local generation, use `--generator-mode local --generator-model /path/to/local/generator-model`. For smoke tests without an external model, use `--generator-mode mock`.

The generation path is:

1. `savant/generation/generator.py` produces a scientific rationale and a strict structured design.
2. `savant/generation/schemas.py` validates the design schema and serializes `claims_dsl` and `mechanisms_dsl`.
3. The verified structured design is passed to the same theorem-grounded verification path used by `--input`.

API example:

```bash
export OPENAI_API_KEY=your_api_key

python run_savant.py \
  --query "Design a nanoparticle adjuvant for a protein antigen vaccine." \
  --generator-mode api \
  --generator-model gpt-4-turbo \
  --output outputs/generated_and_verified.json
```

Local model example:

```bash
python run_savant.py \
  --query "Design a nanoparticle adjuvant for a protein antigen vaccine." \
  --generator-mode local \
  --generator-model /path/to/local/generator-model \
  --device cuda:0 \
  --model /path/to/local/embedding-model \
  --output outputs/generated_and_verified.json
```

Smoke-test example:

```bash
python run_savant.py \
  --query "Design a particulate adjuvant strategy for a protein antigen vaccine." \
  --generator-mode mock \
  --model /path/to/local/embedding-model \
  --output outputs/mock_generated_and_verified.json
```

The output JSON contains:

- `generation`: generated rationale and structured design, when `--query` is used.
- `stage1`: construction support.
- `stage2`: claimed effect support.
- `stage3`: mechanism-chain support.
- `final`: weighted 1-5 verdict.
- `evidence`: matched theorem evidence for each stage.

To trace which theorem supports a proof step, inspect the evidence entries under each stage result. The key fields are `rule_id`, `conditions_repr`, `effect_repr`, `source_paper`, `evidence_text`, `chain_id`, and `step_in_chain`.

## Input Format

The runner expects a structured design object with a `vaccine` field and optional `claims_dsl` and `mechanisms_dsl` lists. It also accepts rows wrapped as `structured_design` or `formalized_design`.

When `--query` is used, no input file is required. The generated structured design is included under the `generation` key in the output JSON and is also used directly for verification.

## Notes For Review

The package does not include experiment-specific scripts, paper-specific case selections, private absolute paths, or local cache paths. The included indexes and ontology files are metadata-cleaned to point to paths inside `theorem_library/`.
