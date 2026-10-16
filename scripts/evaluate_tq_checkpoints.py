"""Run full-game Twenty Questions evaluation for base models or checkpoints.

Typical checkpoint evaluation:

    python -m scripts.evaluate_tq_checkpoints k1_expected \
      --checkpoint-root outputs/tq_grpo_blau_k1_expected_<timestamp> \
      --steps 250 \
      --output-dir outputs/eval/k1_expected \
      --batched-full-game \
      --repeat-seed 42 --repeat-seed 43 --repeat-seed 44

Cross-domain evaluation is done with config overrides for category, theta set,
eval words, and likelihood cache.
"""

from __future__ import annotations

import argparse
import copy
import csv
import gc
import json
import re
import sys
from pathlib import Path
from types import SimpleNamespace

import torch
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf
from tqdm.auto import tqdm

CHECKPOINT_RE = re.compile(r"^checkpoint-(\d+)$")


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("config_name")
    p.add_argument("--checkpoint-root", type=Path)
    p.add_argument(
        "--evaluate-base-model",
        dest="base_model",
        action="store_true",
        help="Evaluate config.model.name directly instead of an adapter checkpoint.",
    )
    p.add_argument("--steps", type=int, nargs="*", default=None)
    p.add_argument("--output-dir", type=Path, default=None)
    p.add_argument("--batched-full-game", action="store_true")
    p.add_argument("--skip-existing", action="store_true")
    p.add_argument(
        "--repeat-seed",
        dest="repeat_seeds",
        type=int,
        action="append",
        default=None,
        help=(
            "Evaluation seed; repeat the flag for multiple seeds while keeping "
            "the model loaded. Results are saved as eval_001, eval_002, ... ."
        ),
    )
    p.add_argument(
        "--repeat-index",
        dest="repeat_indices",
        type=int,
        action="append",
        default=None,
        help=(
            "Run this 1-based entry from the repeated --repeat-seed flags. Repeat "
            "the flag to select multiple entries. This preserves "
            "stable eval_001/eval_002/... paths when repeats are added later."
        ),
    )
    args, overrides = p.parse_known_args()

    if args.base_model:
        if args.checkpoint_root is not None:
            p.error("--evaluate-base-model cannot be combined with --checkpoint-root")
        if args.steps is not None:
            p.error("--evaluate-base-model cannot be combined with --steps")
        if args.output_dir is None:
            p.error("--evaluate-base-model requires --output-dir")
    elif args.checkpoint_root is None:
        p.error("--checkpoint-root is required unless --evaluate-base-model is set")

    args.overrides = overrides
    return args


def load_config(config_name: str, overrides: list[str] | None = None) -> dict:
    repo_root = Path(__file__).resolve().parents[1]
    config_dir = repo_root / "config"

    with initialize_config_dir(
        version_base=None,
        config_dir=str(config_dir.resolve()),
        job_name="evaluate_tq_checkpoints",
    ):
        cfg = compose(config_name=config_name, overrides=overrides or [])

    config = OmegaConf.to_container(cfg, resolve=True)
    if not isinstance(config, dict):
        raise ValueError(f"Expected Hydra config mapping for {config_name}")
    return config


def discover_checkpoints(root: Path, requested_steps=None):
    if not root.is_dir():
        raise FileNotFoundError(f"Checkpoint root does not exist: {root}")

    found = {}
    for path in root.iterdir():
        if not path.is_dir():
            continue
        m = CHECKPOINT_RE.match(path.name)
        if m:
            found[int(m.group(1))] = path

    if requested_steps:
        missing = [step for step in requested_steps if step not in found]
        if missing:
            raise FileNotFoundError(
                f"Requested checkpoints not found under {root}: {missing}"
            )
        return sorted(
            [(step, found[step]) for step in requested_steps],
            key=lambda item: (item[0] % 50 != 0, -item[0]),
        )  # sort in order [250, 200, .., 50, 25, 75, ...]

    if not found:
        raise FileNotFoundError(f"No checkpoint-* directories found under {root}")

    return sorted(found.items(), key=lambda item: (item[0] % 50 != 0, -item[0]))


class StandaloneTrainerShim:
    """Minimal trainer interface required by TQFullGameEvalCallback."""

    def __init__(self, model, processing_class, output_dir: Path):
        self.model = model
        self.processing_class = processing_class
        self.args = SimpleNamespace(output_dir=str(output_dir))

    def log(self, metrics):
        print(json.dumps(metrics, indent=2, sort_keys=True), flush=True)


def write_summary(output_root: Path, rows: list[dict]):
    output_root.mkdir(parents=True, exist_ok=True)

    json_path = output_root / "checkpoint_eval_summary.json"
    with json_path.open("w", encoding="utf-8") as f:
        json.dump(rows, f, indent=2, sort_keys=True)

    csv_path = output_root / "checkpoint_eval_summary.csv"
    metric_names = sorted(
        {
            key
            for row in rows
            for key in row
            if key not in {"checkpoint_step", "checkpoint_path"}
        }
    )
    fieldnames = ["checkpoint_step", "checkpoint_path", *metric_names]

    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    return json_path, csv_path


def main():
    args = parse_args()

    # Allows: python scripts/evaluate_tq_checkpoints.py ...
    repo_root = Path(__file__).resolve().parents[1]
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))

    # Import Unsloth before modules that import Transformers.
    import unsloth  # noqa: F401

    from utils.llm_backend import (
        load_model_and_tokenizer,
        load_model_and_tokenizer_from_checkpoint,
    )
    from utils.tq_full_game_eval import TQFullGameEvalCallback
    from utils.tq_full_game_eval_batched import TQBatchedFullGameEvalCallback
    from utils.tq_prompting import tq_reasoning_enabled
    from utils.tq_trainer import resolve_tq_eval_theta_set, resolve_tq_eval_words

    config = load_config(args.config_name, args.overrides)
    config.setdefault("eval", {})
    config.setdefault("model", {})
    config.setdefault("game", {})

    # Evaluation must use one common information-reward definition across all
    # checkpoints, irrespective of the reward mode used during training.
    # Default to realized Blau/sPCE; override with
    # ++eval.full_game_information_reward_mode=expected if explicitly desired.
    eval_information_reward_mode = str(
        config["eval"].get("full_game_information_reward_mode", "realized")
    ).strip().lower()
    if eval_information_reward_mode not in {"realized", "expected"}:
        raise ValueError(
            "eval.full_game_information_reward_mode must be 'realized' or 'expected'. "
            f"Got: {eval_information_reward_mode!r}."
        )
    config["game"]["information_reward_mode"] = eval_information_reward_mode

    # Use the same train-aligned active threshold for all evaluation runs.
    # The threshold can still be overridden explicitly with
    # ++eval.full_game_active_threshold=... if needed for an ablation.
    eval_active_threshold = float(
        config["eval"].get("full_game_active_threshold", 0.01)
    )
    config["game"]["active_threshold"] = eval_active_threshold

    # In evaluation, once C_{t,l} crosses the active threshold, set it to zero
    # permanently rather than freezing stale nonzero mass in later sPCE sums.
    # This is eval-only: tq_environment.py defaults this flag to False, so the
    # historical training behavior is unchanged.
    zero_inactive_hypotheses = bool(
        config["eval"].get("full_game_zero_inactive_hypotheses", True)
    )
    config["game"]["zero_inactive_hypotheses"] = zero_inactive_hypotheses

    # Constructor expects the key, though standalone mode calls evaluate() directly.
    config["eval"].setdefault("full_game_every_n_steps", 1)

    checkpoint_root = (
        args.checkpoint_root.resolve() if args.checkpoint_root is not None else None
    )
    if args.output_dir is not None:
        output_root = args.output_dir.resolve()
    else:
        if checkpoint_root is None:
            raise ValueError("checkpoint_root is required when output_dir is omitted")
        output_root = checkpoint_root / "standalone_full_game_eval"
    output_root.mkdir(parents=True, exist_ok=True)

    # Put all checkpoint evals in one directory; callback creates step_* subdirs.
    config["eval"]["full_game_output_dir"] = str(output_root)

    checkpoints = (
        [(0, None)]
        if args.base_model
        else discover_checkpoints(checkpoint_root, args.steps)
    )

    theta_set = resolve_tq_eval_theta_set(config)
    eval_words = resolve_tq_eval_words(config, theta_set)

    requested_n = config["eval"].get("full_game_num_eval_words")
    if requested_n is not None:
        # Match the BED-LLM evaluator: use the tail of the held-out list.
        eval_words = eval_words[-int(requested_n) :]
    n_eval = len(eval_words)

    print(
        "\n=== STANDALONE FULL-GAME CHECKPOINT EVAL ==="
        f"\nconfig:          {args.config_name}"
        f"\noverrides:       {args.overrides or '[]'}"
        f"\nmodel source:    {config['model'].get('name') if args.base_model else checkpoint_root}"
        f"\noutput root:     {output_root}"
        f"\ncheckpoints:     {[step for step, _ in checkpoints]}"
        f"\nmode:            {'batched' if args.batched_full_game else 'sequential'}"
        f"\ncategory:        {config.get('game', {}).get('category', 'animals')}"
        f"\nnum eval words:  {n_eval}"
        f"\nbatch size:      {config['eval'].get('full_game_batch_size', 4)}"
        f"\napi batch size:  {config['eval'].get('full_game_api_batch_size', config['eval'].get('full_game_oracle_batch_size', config.get('game', {}).get('oracle_parallel_batch_size', 32)))}"
        f"\nmax turns:       {config['eval'].get('full_game_max_turns', config.get('game', {}).get('max_turns', 20))}"
        f"\nrepeat seeds:    {args.repeat_seeds or [config['eval'].get('full_game_seed', config.get('game', {}).get('seed', 1))]}"
        f"\ninformation reward mode: {config['game']['information_reward_mode']}"
        f"\nenable reasoning: {tq_reasoning_enabled(config)}"
        f"\nactive threshold: {config['game']['active_threshold']}"
        f"\nzero inactive hypotheses: {config['game']['zero_inactive_hypotheses']}"
        f"\nquestion attempts: {config['eval'].get('full_game_question_retries', config.get('evaluation', {}).get('bed_llm', {}).get('question_retries', 1))}"
        f"\ndo sample:       {config['eval'].get('full_game_do_sample', False)}"
        f"\ntemperature:     {config['eval'].get('full_game_temperature', config.get('training', {}).get('temperature', 1.0))}"
        f"\ntheta words:     {config['eval'].get('full_game_eval_theta_set_path', config['eval'].get('full_game_eval_theta_set', '<config/default>'))}"
        f"\neval words:      {config['eval'].get('full_game_eval_words_path', config['eval'].get('full_game_eval_words', '<config/default>'))}",
        flush=True,
    )

    # full config:
    print(
        "\n=== CONFIG ===\n" + json.dumps(config, indent=2, sort_keys=True),
        flush=True,
    )

    base_eval_seed = int(
        config["eval"].get("full_game_seed", config.get("game", {}).get("seed", 1))
    )
    eval_seeds = args.repeat_seeds or [base_eval_seed]
    if len(set(eval_seeds)) != len(eval_seeds):
        raise ValueError(f"--repeat-seed values must be unique, got {eval_seeds}")
    repeat_indices = args.repeat_indices or list(range(1, len(eval_seeds) + 1))
    invalid_repeat_indices = [
        index for index in repeat_indices if index < 1 or index > len(eval_seeds)
    ]
    if invalid_repeat_indices:
        raise ValueError(
            "--repeat-index values must refer to entries in --repeat-seed; "
            f"got {invalid_repeat_indices} for {len(eval_seeds)} seeds"
        )
    if len(set(repeat_indices)) != len(repeat_indices):
        raise ValueError(f"--repeat-index values must be unique, got {repeat_indices}")

    summary_rows = []

    for checkpoint_step, checkpoint_path in tqdm(
        checkpoints,
        desc="checkpoints",
    ):
        model_source = (
            str(config["model"]["name"]) if args.base_model else str(checkpoint_path)
        )
        pending_repeats = []
        for eval_index, eval_seed in enumerate(eval_seeds, start=1):
            if eval_index not in repeat_indices:
                continue
            expected_dir = output_root / (
                f"step_{checkpoint_step:06d}_eval_{eval_index:03d}"
            )
            trajectories_path = expected_dir / "trajectories.json"
            if args.skip_existing and trajectories_path.exists():
                print(
                    f"\nSkipping checkpoint-{checkpoint_step} repeat {eval_index} "
                    f"(seed={eval_seed}): already evaluated.",
                    flush=True,
                )
                with trajectories_path.open("r", encoding="utf-8") as f:
                    saved = json.load(f)
                metrics = TQFullGameEvalCallback._summarize(saved["games"])
                summary_rows.append(
                    {
                        "checkpoint_step": checkpoint_step,
                        "checkpoint_path": model_source,
                        "eval_index": eval_index,
                        "eval_seed": int(saved.get("eval_seed", eval_seed)),
                        **metrics,
                    }
                )
            else:
                pending_repeats.append((eval_index, eval_seed))

        if not pending_repeats:
            write_summary(output_root, summary_rows)
            continue

        print(
            "\n"
            + "=" * 80
            + f"\nEVALUATING {'base model' if args.base_model else f'checkpoint-{checkpoint_step}'}"
            + f"\n{model_source}"
            + "\n"
            + "=" * 80,
            flush=True,
        )

        model = tokenizer = callback = trainer_shim = None

        try:
            if args.base_model:
                model, tokenizer = load_model_and_tokenizer(
                    config,
                    for_inference=True,
                )
            else:
                model, tokenizer = load_model_and_tokenizer_from_checkpoint(
                    config,
                    checkpoint_path,
                    for_inference=True,
                )

            text_tokenizer = getattr(tokenizer, "tokenizer", tokenizer)

            trainer_shim = StandaloneTrainerShim(
                model=model,
                processing_class=text_tokenizer,
                output_dir=checkpoint_root or output_root,
            )

            callback_cls = (
                TQBatchedFullGameEvalCallback
                if args.batched_full_game
                else TQFullGameEvalCallback
            )

            callback = callback_cls(
                config=copy.deepcopy(config),
                trainer=trainer_shim,
                eval_words=eval_words,
                theta_set=theta_set,
                usage_tracker=None,
            )

            for eval_index, eval_seed in pending_repeats:
                callback._eval_call_count = eval_index
                callback.eval_seed = eval_seed
                metrics = callback.evaluate(global_step=checkpoint_step)

                row = {
                    "checkpoint_step": checkpoint_step,
                    "checkpoint_path": model_source,
                    "eval_index": eval_index,
                    "eval_seed": eval_seed,
                    **metrics,
                }
                summary_rows.append(row)

                print(
                    "\nCheckpoint repeat metrics:\n"
                    + json.dumps(row, indent=2, sort_keys=True),
                    flush=True,
                )

                json_path, csv_path = write_summary(output_root, summary_rows)
                print(
                    f"\nUpdated summaries:\n  {json_path}\n  {csv_path}",
                    flush=True,
                )

        finally:
            del callback, trainer_shim, model, tokenizer
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    json_path, csv_path = write_summary(output_root, summary_rows)
    print(
        "\n=== ALL CHECKPOINTS COMPLETE ==="
        f"\nJSON summary: {json_path}"
        f"\nCSV summary:  {csv_path}",
        flush=True,
    )


if __name__ == "__main__":
    main()
