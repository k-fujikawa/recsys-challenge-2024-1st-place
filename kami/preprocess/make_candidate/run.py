"""
候補と生成した特徴量を結合して、lightgbmで学習するためのデータセットを作成する
"""

import os
import sys
from pathlib import Path

import hydra
import polars as pl
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig, OmegaConf

import utils
from utils.data import get_data_dirs


def make_candidate(cfg: DictConfig, data_name: str = "train"):
    input_dir = Path(cfg.dir.input_dir)
    size_name = cfg.exp.size_name
    data_dirs = get_data_dirs(input_dir, size_name)

    # 候補作成
    behaviors_df = pl.read_parquet(data_dirs[data_name] / "behaviors.parquet")
    if data_name == "test":
        behaviors_df = behaviors_df.with_columns(
            pl.lit([1]).alias("article_ids_clicked")
        )
    candidate_df = (
        behaviors_df.select(
            ["impression_id", "article_ids_inview", "user_id", "article_ids_clicked"]
        )
        .explode("article_ids_inview")
        .with_columns(
            pl.col("article_ids_inview")
            .is_in(pl.col("article_ids_clicked"))
            .alias("label")
        )
        .rename({"article_ids_inview": "article_id"})
        .drop("article_ids_clicked")
    )

    candidate_df
    return candidate_df


@hydra.main(version_base=None, config_path=".", config_name="config")
def main(cfg: DictConfig) -> None:
    runtime_choices = HydraConfig.get().runtime.choices
    exp_name = f"{Path(sys.argv[0]).parent.name}/{runtime_choices.exp}"

    print(f"exp_name: {exp_name}")
    output_path = Path(cfg.dir.preprocess_dir) / exp_name
    print(f"ouput_path: {output_path}")
    os.makedirs(output_path, exist_ok=True)

    print(OmegaConf.to_yaml(cfg))

    for data_name in ["train", "validation", "test"]:
        with utils.trace(f"processing {data_name} data"):
            candidate_df = make_candidate(cfg, data_name)
            candidate_df.write_parquet(output_path / f"{data_name}_candidate.parquet")
        print(f"candidate_df: {candidate_df}")


if __name__ == "__main__":
    main()
