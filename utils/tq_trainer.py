import unsloth
import datetime
import json
import os
from pathlib import Path
from transformers import TrainerCallback

from .api_models import ApiUsageTracker
from .llm_backend import load_model_and_tokenizer
from .misc import resolve_repo_path, set_seed
from .build_tq_dataset import (
    load_training_datasets,
    load_tq_words_file,
    resolve_tq_theta_set,
)
from .tq_prompting import (
    tq_chat_template_kwargs,
    tq_reasoning_enabled,
    tq_system_prompt,
)
from .tq_llm_bed_reward import (
    is_llm_bed_reward_type,
    llm_bed_config_from_game,
    normalize_reward_type,
    reject_cache_config_for_llm_bed,
)
from .tq_environment import (
    make_tq_environment_factory,
    prepare_multiturn_dataset,
    tq_trajectory_format_reward_func,
    tq_trajectory_blau_reward_func,
)
from .tq_prompting import tq_chat_template_kwargs, tq_reasoning_enabled

def set_wandb_env(config):
    wandb_config = config.get("wandb", {})

    if wandb_config.get("project"):
        os.environ["WANDB_PROJECT"] = str(wandb_config["project"])

    if wandb_config.get("entity"):
        os.environ["WANDB_ENTITY"] = str(wandb_config["entity"])

    if wandb_config.get("run_id"):
        os.environ["WANDB_RUN_ID"] = str(wandb_config["run_id"])

    if wandb_config.get("resume"):
        os.environ["WANDB_RESUME"] = str(wandb_config["resume"])


def resolve_tq_eval_words(config, theta_set):
    eval_config = config.get("eval")
    value = (eval_config.get("full_game_eval_words_path"))

    if value is None:
        return list(theta_set)

    if isinstance(value, (str, Path)):
        return load_tq_words_file(value)

    if isinstance(value, list):
        cleaned = []
        seen = set()
        for word in value:
            word = str(word).strip()
            if not word or word in seen:
                continue
            cleaned.append(word)
            seen.add(word)
        if not cleaned:
            raise ValueError("Configured TQ eval words are empty.")
        return cleaned

    raise ValueError(
        "Configured TQ eval words must be a path or JSON/YAML list. "
        f"Got: {type(value).__name__}"
    )


def resolve_tq_eval_theta_set(config):
    eval_config = config.get("eval", {})
    value = (
        eval_config.get("full_game_eval_theta_set_path")
        or eval_config.get("full_game_eval_theta_set")
    )

    if value is None:
        return resolve_tq_theta_set(config)

    if isinstance(value, (str, Path)):
        return load_tq_words_file(value)

    if isinstance(value, list):
        cleaned = []
        seen = set()
        for word in value:
            word = str(word).strip()
            if not word or word in seen:
                continue
            cleaned.append(word)
            seen.add(word)
        if not cleaned:
            raise ValueError("Configured TQ eval theta set is empty.")
        return cleaned

    raise ValueError(
        "Configured TQ eval theta set must be a path or JSON/YAML list. "
        f"Got: {type(value).__name__}"
    )


def make_grpo_config(config, reward_type):
    from trl import GRPOConfig

    training_config = dict(config.get("training"))

    # This is passed to trainer.train(), not GRPOConfig.
    training_config.pop("resume_from_checkpoint", None)

    horizon = int(config.get("game").get("rollout_horizon"))
    training_config["max_tool_calling_iterations"] = horizon
    training_config["remove_unused_columns"] = False
    training_config["chat_template_kwargs"] = tq_chat_template_kwargs(config)
    training_config["chat_template_kwargs"] = tq_chat_template_kwargs(config)
    # TRL 0.29 no longer accepts this older GRPO argument.
    training_config.pop("max_prompt_length", None)

    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    output_template = training_config.pop(
        "output_dir",
        "outputs/tq_grpo_{reward_type}_{timestamp}",
    )

    output_dir = resolve_repo_path(
        output_template.format(
            timestamp=timestamp,
            reward_type=reward_type,
        )
    )

    return GRPOConfig(
        output_dir=output_dir,
        **training_config,
    )


def write_api_usage_snapshot(path: Path, usage: dict):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_suffix(path.suffix + ".tmp")
    with temporary_path.open("w", encoding="utf-8") as handle:
        json.dump(
            usage,
            handle,
            indent=2,
            sort_keys=True,
        )
    os.replace(temporary_path, path)


class TQApiUsageCallback(TrainerCallback):
    def __init__(
        self,
        usage_tracker: ApiUsageTracker,
        output_dir: str | Path,
    ):
        self.usage_tracker = usage_tracker
        self.output_dir = Path(output_dir)
        self.path = self.output_dir / "api_usage.json"

    @staticmethod
    def _is_world_process_zero(state) -> bool:
        return bool(
            getattr(
                state,
                "is_world_process_zero",
                True,
            )
        )

    @staticmethod
    def _log_metrics(usage):
        return {
            "api/estimated_total_cost_usd": float(
                usage.get("estimated_total_cost_usd")
            ),
            "api/estimated_oracle_cost_usd": float(
                usage.get("estimated_oracle_cost_usd", 0.0)
            ),
            "api/oracle_api_calls": int(
                usage.get("oracle_api_calls", 0)
            ),
            "api/oracle_cache_hits": int(
                usage.get("oracle_cache_hits", 0)
            ),
            "api/oracle_cache_misses": int(
                usage.get("oracle_cache_misses", 0)
            ),
            "api/oracle_tq_feedback_invocations": int(
                usage.get("oracle_tq_feedback_invocations", 0)
            ),
            "api/oracle_batches_with_cache_misses": int(
                usage.get("oracle_batches_with_cache_misses", 0)
            ),
            "api/oracle_input_tokens": int(
                usage.get("oracle_input_tokens", 0)
            ),
            "api/oracle_cached_input_tokens": int(
                usage.get("oracle_cached_input_tokens", 0)
            ),
            "api/oracle_output_tokens": int(
                usage.get("oracle_output_tokens", 0)
            ),
        }

    def _write(self):
        usage = self.usage_tracker.snapshot()
        write_api_usage_snapshot(self.path, usage)
        return usage

    def on_log(self, args, state, control, logs=None, **kwargs):
        del args, control, kwargs
        if not self._is_world_process_zero(state):
            return
        usage = self._write()
        if logs is not None:
            logs.update(self._log_metrics(usage))
        print(
            "TQ API/cache usage"
            f" @ step {getattr(state, 'global_step', 'unknown')}: "
            f"hits={usage.get('oracle_cache_hits', 0)}, "
            f"misses={usage.get('oracle_cache_misses', 0)}, "
            f"api_calls={usage.get('oracle_api_calls', 0)}, "
            f"est_cost=${float(usage.get('estimated_total_cost_usd', 0.0)):.6f}",
            flush=True,
        )

    def on_save(self, args, state, control, **kwargs):
        del args, control, kwargs
        if self._is_world_process_zero(state):
            self._write()

    def on_train_end(self, args, state, control, **kwargs):
        del args, control, kwargs
        if not self._is_world_process_zero(state):
            return
        usage = self._write()
        print(
            "\nTQ GPT API usage:"
            f"\n  {self.path}"
            f"\n  oracle_api_calls={usage.get('oracle_api_calls', 0)}"
            f"\n  oracle_cache_hits={usage.get('oracle_cache_hits', 0)}"
            f"\n  oracle_cache_misses={usage.get('oracle_cache_misses', 0)}"
            f"\n  oracle_tq_feedback_invocations="
            f"{usage.get('oracle_tq_feedback_invocations', 0)}"
            f"\n  oracle_batches_with_cache_misses="
            f"{usage.get('oracle_batches_with_cache_misses', 0)}"
            f"\n  oracle_input_tokens={usage.get('oracle_input_tokens', 0)}"
            f"\n  oracle_cached_input_tokens={usage.get('oracle_cached_input_tokens', 0)}"
            f"\n  oracle_output_tokens={usage.get('oracle_output_tokens', 0)}"
            f"\n  estimated_total_cost_usd="
            f"{float(usage.get('estimated_total_cost_usd', 0.0)):.6f}",
            flush=True,
        )


class WandbFullConfigCallback(TrainerCallback):
    def __init__(self, config):
        self.config = json.loads(json.dumps(config, default=str))
        self._logged = False

    def on_train_begin(self, args, state, control, **kwargs):
        del args, state, control, kwargs
        if self._logged:
            return
        try:
            import wandb
        except ImportError:
            return
        if wandb.run is None:
            return
        wandb.config.update(self.config, allow_val_change=True)
        self._logged = True


def run_training(config):
    import unsloth
    from trl import GRPOTrainer

    # Multi-turn TQ uses a narrow custom trainer that preserves raw model
    # tokens through the tool call for the loss while rebuilding later-turn
    # contexts from structured Q/A history only.
    from .tq_grpo_trainer import TQGRPOTrainer
    from .tq_grpo_trainer_batched import TQBatchedGRPOTrainer
    from .tq_grpo_trainer_llm_bed import TQLLMBEDGRPOTrainer

    set_wandb_env(config)
    set_seed(config["game"].get("seed"))

    train_dataset, eval_dataset = load_training_datasets(config)

    game_config = config.get("game")
    reward_type = normalize_reward_type(game_config.get("reward_type"))
    game_config["reward_type"] = reward_type
    rollout_horizon = game_config.get("rollout_horizon")
    continuation_samples = game_config.get("continuation_samples")

    if rollout_horizon < 1:
        raise ValueError("game.rollout_horizon must be at least 1.")
    if reward_type != "blau" and rollout_horizon != 1:
        raise ValueError(
            "Multi-turn training is currently implemented only for reward_type='blau'."
        )
    if is_llm_bed_reward_type(reward_type):
        if int(rollout_horizon) != 1:
            raise ValueError("reward_type='llm_bed' is K=1 only.")
        if int(continuation_samples) != 1:
            raise ValueError("reward_type='llm_bed' requires continuation_samples=1.")
        if float(game_config.get("missing_turn_penalty", 0.0)) != 0.0:
            raise ValueError(
                "reward_type='llm_bed' is EIG-only; set missing_turn_penalty=0.0."
            )
        if float(game_config.get("invalid_rollout_penalty", 0.0)) != 0.0:
            raise ValueError(
                "reward_type='llm_bed' is EIG-only; set invalid_rollout_penalty=0.0."
            )
        reject_cache_config_for_llm_bed(game_config)
        if bool(config.get("eval", {}).get("full_game_enabled", False)):
            raise ValueError(
                "Full-game eval currently uses the cache-backed TQ environment; "
                "set eval.full_game_enabled=false for reward_type='llm_bed'."
            )

    usage_tracker = ApiUsageTracker()

    category = game_config.get("category")
    max_turns = game_config.get("max_turns")

    train_dataset = prepare_multiturn_dataset(
        train_dataset,
        category=category,
        max_turns=max_turns,
        horizon=rollout_horizon,
        system_prompt=tq_system_prompt(config),
        config=config,
    )
    eval_dataset = prepare_multiturn_dataset(
        eval_dataset,
        category=category,
        max_turns=max_turns,
        horizon=rollout_horizon,
        system_prompt=tq_system_prompt(config),
        config=config,
    )

    environment_factory = make_tq_environment_factory(
        config,
        usage_tracker=usage_tracker,
    )

    reward_funcs = [
        tq_trajectory_format_reward_func,
        tq_trajectory_blau_reward_func,
    ]

    model, tokenizer = load_model_and_tokenizer(
        config,
        for_training=True,
    )

    training_args = make_grpo_config(
        config,
        reward_type=reward_type,
    )

    api_usage_path = Path(training_args.output_dir) / "api_usage.json"
    if api_usage_path.exists():
        with api_usage_path.open("r", encoding="utf-8") as handle:
            usage_tracker.update(json.load(handle))

    eval_config = config.get("eval")
    callbacks = [
        TQApiUsageCallback(
            usage_tracker=usage_tracker,
            output_dir=training_args.output_dir,
        ),
        WandbFullConfigCallback(config),
    ]

    processing_class = (getattr(tokenizer, "tokenizer", tokenizer))

    trainer_kwargs = {
        "model": model,
        "processing_class": processing_class,
        "reward_funcs": reward_funcs,
        "args": training_args,
        "train_dataset": train_dataset,
        "eval_dataset": eval_dataset.select(
            range(min(1, len(eval_dataset)))
        ),
        "callbacks": callbacks,
    }

    trainer_kwargs["environment_factory"] = environment_factory
    trainer_kwargs["tq_continuation_samples"] = continuation_samples
    trainer_kwargs["tq_rewarded_token_mode"] = game_config.get(
        "rewarded_token_mode",
        "all",
    )


    if is_llm_bed_reward_type(reward_type):
        trainer_cls = TQLLMBEDGRPOTrainer
        trainer_kwargs.update(
            {
                "llm_bed_config": llm_bed_config_from_game(game_config),
                "category": category,
                "max_turns": max_turns,
                "chat_template_kwargs": tq_chat_template_kwargs(config),
                "usage_tracker": usage_tracker,
            }
        )
    else:
        trainer_cls = TQBatchedGRPOTrainer
        trainer_kwargs.update(
            {
                "likelihood_cache_path": game_config.get("likelihood_cache_path"),
                "usage_tracker": usage_tracker,
                "oracle_batch_size": game_config.get("oracle_batch_size"),
                "oracle_max_output_tokens": game_config.get("oracle_max_output_tokens"),
                "oracle_max_retries": game_config.get("oracle_max_retries"),
                "oracle_reasoning_effort": game_config.get("oracle_reasoning_effort"),
            }
        )
    trainer = trainer_cls(**trainer_kwargs)

    # Attach the full-game callback after trainer construction so the callback
    # can use the current training model and send metrics through trainer.log().
    if eval_config.get("full_game_enabled"):
        from .tq_full_game_eval import TQFullGameEvalCallback

        callback_theta_set = resolve_tq_eval_theta_set(config)
        callback_eval_words = resolve_tq_eval_words(
            config,
            callback_theta_set,
        )

        trainer.add_callback(
            TQFullGameEvalCallback(
                config=config,
                trainer=trainer,
                eval_words=callback_eval_words,
                theta_set=callback_theta_set,
                usage_tracker=usage_tracker,
            )
        )

    resume_from_checkpoint = config.get("training").get("resume_from_checkpoint")

    if isinstance(resume_from_checkpoint, str):
        resume_from_checkpoint = str(
            resolve_repo_path(resume_from_checkpoint)
        )

        checkpoint_path = Path(resume_from_checkpoint)

        if not checkpoint_path.is_dir():
            raise FileNotFoundError(
                f"Checkpoint directory does not exist: {checkpoint_path}"
            )

        if not (checkpoint_path / "trainer_state.json").exists():
            raise FileNotFoundError(
                f"trainer_state.json not found in checkpoint: {checkpoint_path}"
            )

    print(f"Output directory: {training_args.output_dir}")
    print(f"Resume checkpoint: {resume_from_checkpoint!r}")

    llm_bed_config = (
        llm_bed_config_from_game(game_config)
        if is_llm_bed_reward_type(reward_type)
        else None
    )
    likelihood_description = (
        f", llm_bed_likelihood_source={llm_bed_config.likelihood_source}"
        if llm_bed_config is not None
        else ""
    )
    print(
        "Twenty Questions GRPO training: "
        f"reward_type={reward_type}, "
        f"rollout_horizon={rollout_horizon}, "
        f"continuation_samples_M={continuation_samples}, "
        f"rollout_gamma={float(game_config.get('rollout_gamma'))}, "
        f"information_reward_mode="
        f"{game_config['information_reward_mode']}, "
        f"enable_reasoning={tq_reasoning_enabled(config)}, "
        f"rewarded_token_mode={game_config.get('rewarded_token_mode', 'all')}"
        f"{likelihood_description}, "
        f"chat_template_kwargs={getattr(training_args, 'chat_template_kwargs')}"
    )

    trainer.train(
        resume_from_checkpoint=resume_from_checkpoint
    )

    return {
        "output_dir": training_args.output_dir,
        "reward_type": reward_type,
        "information_reward_mode": game_config.get("information_reward_mode"),
        "rollout_horizon": rollout_horizon,
        "continuation_samples": continuation_samples,
        "enable_reasoning": tq_reasoning_enabled(config),
        "api_usage": usage_tracker.snapshot(),
    }
