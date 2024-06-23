import os
import sys
import time
from pathlib import Path

import hydra
import polars as pl
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig, OmegaConf

import utils


def make_demo(cfg: DictConfig):
    input_dir = Path(cfg.dir.input_dir)
    test_dir = input_dir / "ebnerd_testset" / "ebnerd_testset" / "test"
    test_behaviors_df = pl.read_parquet(test_dir / "behaviors.parquet")
    test_history_df = pl.read_parquet(test_dir / "history.parquet")

    demo_history_df = test_history_df.sample(n=cfg.exp.n_sample, seed=cfg.exp.seed)

    demo_users = demo_history_df["user_id"].to_list()
    demo_behaivors_df = test_behaviors_df.filter(pl.col("user_id").is_in(demo_users))
    return demo_behaivors_df, demo_history_df


@hydra.main(version_base=None, config_path=".", config_name="config")
def main(cfg: DictConfig) -> None:
    runtime_choices = HydraConfig.get().runtime.choices
    exp_name = f"{Path(sys.argv[0]).parent.name}/{runtime_choices.exp}"

    print(f"exp_name: {exp_name}")
    output_path = Path(cfg.dir.preprocess_dir) / exp_name
    print(f"ouput_path: {output_path}")
    os.makedirs(output_path, exist_ok=True)

    print(OmegaConf.to_yaml(cfg))

    demo_behaivors_df, demo_history_df = make_demo(cfg)

    # Save
    os.makedirs(output_path / "test", exist_ok=True)
    demo_behaivors_df.write_parquet(output_path / "test" / "behaviors.parquet")
    demo_history_df.write_parquet(output_path / "test" / "history.parquet")


if __name__ == "__main__":
    main()
