"""Main entry point for Twenty Questions data building and training."""

import json
import sys
from pathlib import Path

import hydra
from omegaconf import DictConfig, OmegaConf

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


@hydra.main(version_base=None, config_path="../config", config_name="tq_base")
def main(cfg: DictConfig):
    config = OmegaConf.to_container(cfg, resolve=True)
    mode = config.get("mode", "train")
    if mode == "build_data":
        from utils.build_tq_dataset import save_training_datasets

        result = save_training_datasets(config)
        print(json.dumps(result, indent=2))
        return

    elif mode == "train":
        from utils.tq_trainer import run_training

        result = run_training(config)
        print(json.dumps(result, indent=2))
        return

    elif mode == "eval":
        raise ValueError(
            "Use the TQ evaluation scripts directly, for example "
            "scripts/evaluate_tq_checkpoints.py or "
            "scripts/evaluate_bed_llm_baseline.py."
        )

    raise ValueError(f"Unknown mode: {mode!r}")


if __name__ == "__main__":
    main()

