from __future__ import annotations

import json
import math
from typing import Any, Dict, List, Sequence

import numpy as np

from .build_tq_dataset import ANSWER_LABELS, tq_feedback
from .tq_prompting import REASONING_SYSTEM_PROMPT, tq_row_category, tq_system_prompt
from .tq_llm_bed_reward import is_llm_bed_reward_type, normalize_reward_type


MULTITURN_SYSTEM_PROMPT = REASONING_SYSTEM_PROMPT


def ask_question(question: str) -> str:
    """Ask one question about the hidden target.

    Args:
        question: An informative question that can be answered with yes, no, or maybe.

    Returns:
        The oracle answer and game status. `Status: continue` means the exact
        target has not yet been identified; `Status: solved` means it has.
    """
    raise RuntimeError("Schema-only tool; the evaluator does not call it directly.")


def _load_json(value: Any, *, name: str) -> Any:
    if isinstance(value, str):
        try:
            return json.loads(value)
        except json.JSONDecodeError as exc:
            raise ValueError(f"Invalid JSON in {name}: {value!r}") from exc
    return value


def _normalise_distribution(dist: Any) -> Dict[str, Any] | None:
    if not isinstance(dist, dict):
        return None

    try:
        values = np.asarray(
            [float(dist[label]) for label in ANSWER_LABELS],
            dtype=float,
        )
    except (KeyError, TypeError, ValueError):
        return None

    if not np.all(np.isfinite(values)) or np.any(values < 0):
        return None

    total = float(values.sum())
    if total <= 0:
        return None

    values /= total
    return {
        "yes": float(values[0]),
        "no": float(values[1]),
        "maybe": float(values[2]),
        # Preserve the oracle's exact-target signal for full-game evaluation.
        # This does not affect likelihood normalization or Blau reward.
        "solved": bool(dist.get("solved", False)),
    }


def _get_likelihoods(
    *,
    question: str,
    candidates: Sequence[str],
    cache_path: str,
    usage_tracker=None,
) -> Dict[str, Dict[str, float]]:
    candidates = list(dict.fromkeys(str(candidate) for candidate in candidates))
    if not candidates:
        return {}

    rows = tq_feedback(
        questions=[question],
        ground_truths=candidates,
        cache_path=cache_path,
        usage_tracker=usage_tracker,
        track_cache_status=usage_tracker is not None,
    )
    if not rows:
        return {}

    likelihoods = {}
    for candidate, raw_dist in zip(candidates, rows[0]):
        dist = _normalise_distribution(raw_dist)
        if dist is not None:
            likelihoods[candidate] = dist

    return likelihoods


def _answer_likelihood_vector(dist: Any) -> np.ndarray | None:
    if not isinstance(dist, dict):
        return None

    try:
        values = np.asarray(
            [float(dist[label]) for label in ANSWER_LABELS],
            dtype=float,
        )
    except (KeyError, TypeError, ValueError):
        return None

    if not np.all(np.isfinite(values)) or np.any(values < 0):
        return None

    total = float(values.sum())
    if total <= 0:
        return None

    return values / total


def _collect_active_answer_likelihoods(
    *,
    C_prev: np.ndarray,
    active_indices: Sequence[int],
    active_candidates: Sequence[str],
    likelihoods: Dict[str, Dict[str, float]],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    valid_indices = []
    weights = []
    rows = []

    for index, candidate in zip(
        active_indices,
        active_candidates,
    ):
        index = int(index)
        weight = float(C_prev[index])
        if not np.isfinite(weight) or weight <= 0:
            continue

        row = _answer_likelihood_vector(likelihoods.get(candidate))
        if row is None:
            continue

        valid_indices.append(index)
        weights.append(weight)
        rows.append(row)

    if not rows:
        return (
            np.asarray([], dtype=int),
            np.asarray([], dtype=float),
            np.empty((0, len(ANSWER_LABELS)), dtype=float),
        )

    return (
        np.asarray(valid_indices, dtype=int),
        np.asarray(weights, dtype=float),
        np.asarray(rows, dtype=float),
    )


def _blau_increment(
    C_prev: np.ndarray,
    C_new: np.ndarray,
    true_likelihood: float,
    eps: float = 1e-12,
) -> float:
    """
    Realized pointwise information gain for one observed answer.

    Let w_i = C_{t-1}(theta_i), S = sum_i w_i, and
    p_i(a) = p(a | theta_i, d). For the realized answer a sampled from the
    true theta*, the unnormalized posterior mass is

        S_a = sum_i w_i p_i(a).

    The current reward is the log posterior gain of theta*:

        IG_real(theta*, a, d)
          = log p(a | theta*, d) - log S_a + log S
          = log p(theta* | h, d, a) - log p(theta* | h)

    where the theta* prior weight cancels. C_new already contains
    w_i p_i(a), so S_a is C_new.sum().
    """
    return float(
        math.log(max(float(true_likelihood), eps))
        - math.log(max(float(C_new.sum()), eps))
        + math.log(max(float(C_prev.sum()), eps))
    )


def _expected_blau_increment(
    weights: np.ndarray,
    answer_likelihoods: np.ndarray,
    eps: float = 1e-12,
) -> float:
    """
    Expected information gain after marginalizing over possible answers.

    This is the Rao-Blackwellized expectation of the realized reward over the
    active belief support for which we queried p(a | theta, d). Define

        w_i       = C_{t-1}(theta_i)
        S         = sum_i w_i
        q_i       = w_i / S
        p_i(a)    = p(a | theta_i, d)
        p_bar(a)  = sum_i q_i p_i(a) = p(a | h, d).

    The expected one-step reward is

        EIG(d)
          = E_{theta ~ q, a ~ p(. | theta, d)}
              [ log p(a | theta, d) - log p(a | h, d) ]
          = sum_i q_i sum_a p_i(a) [log p_i(a) - log p_bar(a)]
          = H[p_bar(a)] - sum_i q_i H[p_i(a)].

    This is nonnegative mutual information I(theta; a | h, d). In the hard,
    deterministic, uniform-support special case, p_i(a) is one-hot and
    q_i = 1 / |C|. If C_a = {theta_i : answer(theta_i, d) = a}, then

        EIG(d)
          = -sum_a (|C_a| / |C|) log(|C_a| / |C|)
          = log |C| - sum_a (|C_a| / |C|) log |C_a|,

    which is the count-based formula used for hard partitions.
    """
    weights = np.asarray(weights, dtype=float)
    matrix = np.asarray(answer_likelihoods, dtype=float)
    if weights.size == 0 or matrix.size == 0:
        return 0.0
    total_weight = float(weights.sum())
    if total_weight <= 0:
        return 0.0

    q = weights / total_weight
    marginal = q @ matrix

    log_matrix = np.zeros_like(matrix)
    positive = matrix > 0
    log_matrix[positive] = np.log(np.maximum(matrix[positive], eps))
    log_marginal = np.log(np.maximum(marginal, eps))
    eig = float(np.sum(q[:, None] * matrix * (log_matrix - log_marginal)))

    if -1e-10 < eig < 0:
        eig = 0.0
    return float(eig)


def _blau_step_update_and_reward(
    *,
    C_prev: np.ndarray,
    active_indices: Sequence[int],
    active_candidates: Sequence[str],
    likelihoods: Dict[str, Dict[str, float]],
    observed_answer: str,
    true_likelihood: float,
    information_reward_mode: str,
    active_threshold: float | None = None,
    zero_inactive_hypotheses: bool = False,
) -> tuple[np.ndarray, float, bool]:
    valid_indices, active_weights, answer_likelihoods = (
        _collect_active_answer_likelihoods(
            C_prev=C_prev,
            active_indices=active_indices,
            active_candidates=active_candidates,
            likelihoods=likelihoods,
        )
    )
    if len(valid_indices) == 0:
        return C_prev.copy(), 0.0, False

    answer_index = ANSWER_LABELS.index(observed_answer)
    C_new = C_prev.copy()
    C_new[valid_indices] *= answer_likelihoods[:, answer_index]

    # Evaluation-only truncation can zero hypotheses as soon as their
    # cumulative contrastive weight crosses the active threshold. The default
    # is False so existing GRPO training behavior is unchanged unless the
    # config explicitly enables it.
    if zero_inactive_hypotheses:
        if active_threshold is None:
            raise ValueError(
                "active_threshold is required when zero_inactive_hypotheses=True."
            )
        C_new[C_new <= float(active_threshold)] = 0.0

    if information_reward_mode == "expected":
        step_reward = _expected_blau_increment(
            weights=active_weights,
            answer_likelihoods=answer_likelihoods,
        )
    elif information_reward_mode == "realized":
        step_reward = _blau_increment(
            C_prev=C_prev,
            C_new=C_new,
            true_likelihood=true_likelihood,
        )
    else:
        raise ValueError(
            "information_reward_mode must be 'realized' or 'expected'. "
            f"Got: {information_reward_mode!r}."
        )

    return C_new, float(step_reward), True


def make_multiturn_prompt(
    *,
    history,
    category: str,
    turn: int,
    max_turns: int,
    system_prompt: str | None = None,
    config: dict[str, Any] | None = None,
):
    """Build a uniform tool-call history for the policy context.

    Existing dataset history is serialized exactly like newly generated
    rollout history: assistant ``ask_question`` tool call followed by a tool
    response containing only ``Answer: <label>``.

    ``turn`` is retained in the function signature for compatibility with the
    existing dataset-preparation call, but the conversation itself represents
    the current state, so no potentially stale turn counter is shown.
    """
    del turn

    if system_prompt is None and config is not None:
        system_prompt = tq_system_prompt(config)

    messages = [
        {"role": "system", "content": system_prompt or MULTITURN_SYSTEM_PROMPT},
        {
            "role": "user",
            "content": (
                f"Goal: Identify the hidden target in at most {max_turns} turns.\n"
                f"Category: {category}."
            ),
        },
    ]

    for step in history:
        question = str(step["question"]).strip()
        answer = str(step["answer"]).strip()

        messages.append(
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {
                        "type": "function",
                        "function": {
                            "name": "ask_question",
                            "arguments": {
                                "question": question,
                            },
                        },
                    }
                ],
            }
        )
        messages.append(
            {
                "role": "tool",
                "name": "ask_question",
                "content": (f"Answer: {answer}\nStatus: continue"),
            }
        )

    return messages


def prepare_multiturn_dataset(
    dataset,
    *,
    category: str,
    max_turns: int,
    horizon: int,
    system_prompt: str | None = None,
    config: dict[str, Any] | None = None,
):
    """Replace the XML prompt and retain states with K turns remaining."""
    horizon = int(horizon)
    max_turns = int(max_turns)

    if horizon < 1:
        raise ValueError("game.rollout_horizon must be at least 1.")

    dataset = dataset.filter(lambda row: int(row["turn"]) + horizon - 1 <= max_turns)

    if len(dataset) == 0:
        raise ValueError(
            "No dataset rows have enough remaining turns for "
            f"rollout_horizon={horizon}."
        )

    def replace_prompt(row):
        history = _load_json(row["history_json"], name="history_json")
        return {
            "prompt": make_multiturn_prompt(
                history=history,
                category=tq_row_category(row, category=category, config=config),
                turn=int(row["turn"]),
                max_turns=max_turns,
                system_prompt=system_prompt,
                config=config,
            )
        }

    return dataset.map(replace_prompt)


def make_tq_environment_factory(config, usage_tracker=None):
    """Return a zero-argument TQ environment factory for ``GRPOTrainer``."""
    game_config = config.get("game")

    reward_type = normalize_reward_type(game_config.get("reward_type"))
    if reward_type not in {"blau", "llm_bed"}:
        raise ValueError(
            "Twenty Questions reward_type must be 'blau', 'llm_bed', or "
            "'asig_eig'."
        )

    horizon = int(game_config.get("rollout_horizon"))
    gamma = float(game_config.get("rollout_gamma"))
    information_reward_mode = game_config.get("information_reward_mode", "expected")
    if reward_type == "blau" and information_reward_mode not in {"realized", "expected"}:
        raise ValueError(
            "game.information_reward_mode must be 'realized' or 'expected'. "
            f"Got: {information_reward_mode!r}."
        )
    active_threshold = float(game_config.get("active_threshold"))
    zero_inactive_hypotheses = bool(
        game_config.get("zero_inactive_hypotheses", False)
    )
    likelihood_cache_path = game_config.get("likelihood_cache_path")
    missing_turn_penalty = float(game_config.get("missing_turn_penalty"))
    invalid_rollout_penalty = float(game_config.get("invalid_rollout_penalty"))

    if horizon < 1:
        raise ValueError("game.rollout_horizon must be at least 1.")
    if reward_type == "llm_bed" and horizon != 1:
        raise ValueError("reward_type='llm_bed' is K=1 only; set rollout_horizon: 1.")
    if not 0.0 <= gamma <= 1.0:
        raise ValueError("game.rollout_gamma must be in [0, 1].")
    if reward_type == "blau" and not likelihood_cache_path:
        raise ValueError("game.likelihood_cache_path is required.")
    if reward_type == "llm_bed" and likelihood_cache_path:
        raise ValueError(
            "reward_type='llm_bed' must not configure game.likelihood_cache_path."
        )

    class TwentyQuestionsEnv:
        def __init__(self):
            self.horizon = horizon
            self.gamma = gamma
            self.reward_type = reward_type
            self.information_reward_mode = information_reward_mode
            self.active_threshold = active_threshold
            self.zero_inactive_hypotheses = zero_inactive_hypotheses
            self.likelihood_cache_path = likelihood_cache_path
            self.missing_turn_penalty = missing_turn_penalty
            self.invalid_rollout_penalty = invalid_rollout_penalty
            self.step_rewards: List[float] = []
            self.step_debug: List[Dict[str, Any]] = []
            self.invalid = False
            self.last_solved = False
            self.solved = False
            # Optional Monte-Carlo reward overrides used when M>1 continuation
            # samples are averaged back onto a single root action.
            self._blau_reward_override: float | None = None
            self._format_reward_override: float | None = None

        def reset(
            self,
            true_word: str,
            history_json: str,
            contrastive_set_json: str,
            C_json: str,
            **kwargs,
        ) -> None:
            del kwargs
            self.true_word = str(true_word)
            self.history = list(_load_json(history_json, name="history_json"))
            self.contrastive_set = list(
                _load_json(contrastive_set_json, name="contrastive_set_json")
            )
            self.C = np.asarray(_load_json(C_json, name="C_json"), dtype=float)

            if len(self.C) != len(self.contrastive_set):
                raise ValueError(
                    "C_json and contrastive_set_json must have equal length."
                )

            self.step_rewards = []
            self.step_debug = []
            self.invalid = False
            self.last_solved = False
            self.solved = False
            self._blau_reward_override = None
            self._format_reward_override = None

        def _blau_step(self, question: str):
            if is_llm_bed_reward_type(self.reward_type):
                raise RuntimeError(
                    "LLM-BED rewards are computed by the trainer from current "
                    "policy logits, not by the cache-backed Blau environment."
                )

            C_prev = self.C.copy()
            active_indices = np.flatnonzero(C_prev > self.active_threshold)
            active_candidates = [
                self.contrastive_set[index] for index in active_indices
            ]

            query_candidates = list(active_candidates)
            if self.true_word not in query_candidates:
                query_candidates.append(self.true_word)

            likelihoods = _get_likelihoods(
                question=question,
                candidates=query_candidates,
                cache_path=self.likelihood_cache_path,
                usage_tracker=usage_tracker,
            )

            true_dist = likelihoods.get(self.true_word)
            if true_dist is None:
                raise ValueError(
                    "Missing true-word likelihood: "
                    f"question={question!r}, true_word={self.true_word!r}."
                )

            # `solved` is metadata returned by the oracle when the question
            # identifies the true target. It is bookkeeping only and does not
            # enter the Blau likelihood/reward computation.
            self.last_solved = bool(true_dist.get("solved", False))
            self.solved = self.solved or self.last_solved

            observed_answer = max(
                ANSWER_LABELS,
                key=lambda label: true_dist[label],
            )
            true_likelihood = float(true_dist[observed_answer])

            if not active_candidates:
                return observed_answer, 0.0

            C_new, step_reward, updated = _blau_step_update_and_reward(
                C_prev=C_prev,
                active_indices=active_indices,
                active_candidates=active_candidates,
                likelihoods=likelihoods,
                observed_answer=observed_answer,
                true_likelihood=true_likelihood,
                information_reward_mode=self.information_reward_mode,
                active_threshold=self.active_threshold,
                zero_inactive_hypotheses=self.zero_inactive_hypotheses,
            )
            if not updated:
                return observed_answer, 0.0

            self.C = C_new

            return observed_answer, float(step_reward)

        def _get_blau_reward(self, gamma_override=None) -> float:
            """Discounted K-turn information return, with no formatting/validity penalties."""
            if gamma_override is None and self._blau_reward_override is not None:
                return float(self._blau_reward_override)

            reward_gamma = (
                self.gamma if gamma_override is None else float(gamma_override)
            )
            if not 0.0 <= reward_gamma <= 1.0:
                raise ValueError("Blau reward gamma must be in [0, 1].")

            return float(
                sum(
                    (reward_gamma**turn_idx) * reward
                    for turn_idx, reward in enumerate(self.step_rewards)
                )
            )

        def _get_format_reward(self) -> float:
            """
            Return only the non-Blau formatting/validity component.

            Solving is a valid terminal event: no remaining rollout turns are
            required and therefore no missing-turn penalty is applied.
            """
            if self._format_reward_override is not None:
                return float(self._format_reward_override)

            missing_turns = (
                0 if self.solved else max(0, self.horizon - len(self.step_rewards))
            )

            return float(
                -self.missing_turn_penalty * missing_turns
                - self.invalid_rollout_penalty * int(self.invalid)
            )

        def _set_reward_overrides(self, *, blau: float, fmt: float) -> None:
            """Store an M-sample Monte-Carlo estimate for this root rollout."""
            self._blau_reward_override = float(blau)
            self._format_reward_override = float(fmt)

        def _get_reward(self) -> float:
            """Backward-compatible combined trajectory reward."""
            return float(self._get_blau_reward() + self._get_format_reward())

        def ask_question(self, question: str) -> str:
            """Ask one question about the hidden target.

            Args:
                question: An informative question that can be answered with yes, no, or maybe.

            Returns:
                The oracle answer and game status. `Status: continue` means
                the exact target has not yet been identified; `Status: solved`
                means it has.
            """

            if self.solved:
                raise ValueError("Game already solved; no further question is allowed.")

            if len(self.step_rewards) >= self.horizon:
                self.invalid = True
                raise ValueError("Rollout horizon already reached.")

            question = str(question).strip()
            if not question:
                self.invalid = True
                raise ValueError("Question must be non-empty.")

            try:
                observed_answer, step_reward = self._blau_step(question)
            except Exception:
                self.invalid = True
                raise

            self.history.append({"question": question, "answer": observed_answer})
            self.step_rewards.append(float(step_reward))
            self.step_debug.append(
                {
                    "question": question,
                    "answer": observed_answer,
                    "reward": float(step_reward),
                    "active_count": int(np.sum(self.C > self.active_threshold)),
                    "sum_C": float(np.sum(self.C)),
                    "solved": bool(self.last_solved),
                }
            )

            status = "solved" if self.last_solved else "continue"
            return f"Answer: {observed_answer}\nStatus: {status}"

        def _record_llm_bed_step(
            self,
            *,
            question: str,
            reward: float,
            debug: Dict[str, Any] | None = None,
        ) -> str:
            """Record a trainer-computed K=1 LLM-BED EIG step."""
            if not is_llm_bed_reward_type(self.reward_type):
                raise RuntimeError("_record_llm_bed_step is only for llm_bed reward.")
            if self.solved:
                raise ValueError("Game already solved; no further question is allowed.")
            if len(self.step_rewards) >= self.horizon:
                self.invalid = True
                raise ValueError("Rollout horizon already reached.")

            question = str(question).strip()
            if not question:
                self.invalid = True
                raise ValueError("Question must be non-empty.")

            self.last_solved = False
            self.solved = False
            step_reward = float(reward)
            self.step_rewards.append(step_reward)

            # K=1 LLM-BED never consumes the answer in a future policy context,
            # so this placeholder is bookkeeping only. It avoids any external
            # answerer/cache while keeping the TRL tool loop structurally valid.
            observed_answer = "maybe"
            self.history.append({"question": question, "answer": observed_answer})
            step_debug = {
                "question": question,
                "answer": observed_answer,
                "reward": step_reward,
                "active_count": None,
                "sum_C": None,
                "solved": False,
                "reward_mode": "llm_bed",
            }
            if debug:
                step_debug.update(debug)
            self.step_debug.append(step_debug)
            return "Answer: maybe\nStatus: continue"

    return TwentyQuestionsEnv


_reward_call_count = 0


def _validate_reward_batch(completions, environments):
    if len(completions) != len(environments):
        raise ValueError("completions and environments must have equal length.")


def tq_trajectory_format_reward_func(
    completions,
    environments,
    **kwargs,
):
    """
    Return only the trajectory formatting/validity reward.

    This contains missing-turn and invalid-rollout penalties and contains
    no Blau information reward.
    """
    del kwargs
    _validate_reward_batch(
        completions,
        environments,
    )

    return [float(env._get_format_reward()) for env in environments]


def tq_trajectory_blau_reward_func(
    completions,
    environments,
    **kwargs,
):
    """
    Return only the discounted K-turn Blau reward.

    This contains no formatting/validity penalties.
    """

    del kwargs
    global _reward_call_count

    _validate_reward_batch(
        completions,
        environments,
    )

    _reward_call_count += 1

    if _reward_call_count == 1:
        print(
            "\n=== FIRST MULTI-TURN TRAJECTORIES ===",
            flush=True,
        )

        for index, env in enumerate(environments[:1]):
            print(
                f"trajectory={index} "
                f"questions="
                f"{[step['question'] for step in env.step_debug]} "
                f"step_rewards={env.step_rewards} "
                f"gamma={env.gamma:.6f} "
                f"information_reward_mode={env.information_reward_mode} "
                f"information_reward={env._get_blau_reward():.6f} "
                f"total_reward={env._get_reward():.6f}",
                flush=True,
            )

    return [float(env._get_blau_reward()) for env in environments]
