# Amortizing Sequential Experimental Design in Reasoning LLMs

Official code for the paper **[Amortizing Sequential Experimental Design in Reasoning LLMs](https://arxiv.org/abs/2610.20832)**.

## Requirements

Create a Python environment and install the package requirements:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Training uses local Hugging Face models. Building new data or filling missing oracle likelihoods requires the API credentials used by `utils/api_models.py`; place them in your shell environment or a local `.env` file.

## Train

Train the main animal AR-BED model:

```bash
python scripts/run.py --config-name k1_expected
```

Train the Medi-Q three-domain AR-BED model:

```bash
python scripts/run.py --config-name tq_mediq_3domain_k1_expected
```

## Eval

Evaluate an animal checkpoint:

```bash
python -m scripts.evaluate_tq_checkpoints k1_expected \
  --checkpoint-root outputs/tq_grpo_blau_k1_expected_<timestamp> \
  --steps 250 \
  --output-dir outputs/eval/k1_expected \
  --batched-full-game
```

Evaluate the base model instead of a checkpoint:

```bash
python -m scripts.evaluate_tq_checkpoints k1_expected \
  --evaluate-base-model \
  --output-dir outputs/eval/base_model \
  --batched-full-game
```

Run the BED-LLM baseline:

```bash
python -m scripts.evaluate_bed_llm_baseline \
  --config config/tq_base.yaml \
  --theta-words data/tq/animals_200.json \
  --eval-words data/tq/animals_test.json \
  --output-dir outputs/bed_llm_base_animals \
  --likelihood-cache outputs/bed_llm_base_animals/cache.json
```

## Build Data

The animal and Medi-Q training files used by the public configs are already included under `data/tq/`.

To rebuild animal sampled-state data:

```bash
python scripts/run.py --config-name tq_base mode=build_data \
  data.train_states=100 \
  data.eval_states=20 \
  data.sampled_train_path=data/tq/train_sampled_states_200animals.jsonl \
  data.sampled_eval_path=data/tq/eval_sampled_states_200animals.jsonl
```

To rebuild the Medi-Q disease lists from raw Medi-Q JSONL files:

```bash
python scripts/build_mediq_tq_disease_sets.py \
  --imedqa /path/to/imedqa.jsonl \
  --craft /path/to/craft.jsonl \
  --output-dir data/tq
```

To rebuild Medi-Q sampled-state data for one domain, use `tq_build_mediq_base` with domain-specific overrides. Repeat with the dermatology, neurology, and gastroenterology disease files to recreate the three-domain training set.

```bash
python scripts/run.py --config-name tq_build_mediq_base \
  game.category="dermatologic disease" \
  game.likelihood_cache_path=data/tq/tq_feedback_cache_mediq_dermatology_50.json \
  data.theta_set_path=data/tq/diseases_dermatology_50.json \
  data.sampled_train_path=data/tq/train_sampled_states_mediq_dermatology_50.jsonl \
  data.sampled_eval_path=data/tq/eval_sampled_states_mediq_dermatology_50.jsonl
```