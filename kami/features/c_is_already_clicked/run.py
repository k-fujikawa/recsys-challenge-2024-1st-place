"""
未来のimpressionに出現するかどうかを特徴量として作成する（本来ならリークになるので注意）
"""

import itertools
import os
import sys
from pathlib import Path

import hydra
import polars as pl
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig

from utils.data import get_data_dirs

PREFIX = "c"


USE_COLUMNS = ["is_already_clicked"]


def process_df(cfg, history_df, candidate_df):
    candidate_df = candidate_df.join(
        history_df.select(["user_id", "article_id_fixed"]),
        on="user_id",
        how="left",
    )

    df = candidate_df.with_columns(
        pl.col("article_id")
        .is_in(pl.col("article_id_fixed"))
        .alias("is_already_clicked")
    )
    return df


def create_feature(cfg: DictConfig, output_path):
    input_dir = Path(cfg.dir.input_dir)
    size_name = cfg.exp.size_name
    data_dirs = get_data_dirs(input_dir, size_name)

    for data_name in ["train", "validation", "test"]:
        print(f"processing {data_name} data")
        history_df = pl.read_parquet(data_dirs[data_name] / "history.parquet")
        candidate_df = pl.read_parquet(
            Path(cfg.dir.candidate_dir) / size_name / f"{data_name}_candidate.parquet"
        )

        df = process_df(cfg, history_df, candidate_df).select(USE_COLUMNS)

        df = df.rename({col: f"{PREFIX}_{col}" for col in USE_COLUMNS})
        print(df)
        df.write_parquet(
            output_path / f"{data_name}_feat.parquet",
        )


@hydra.main(version_base=None, config_path=".", config_name="config")
def main(cfg: DictConfig) -> None:
    runtime_choices = HydraConfig.get().runtime.choices
    exp_name = f"{Path(sys.argv[0]).parent.name}/{runtime_choices.exp}"

    print(f"exp_name: {exp_name}")
    output_path = Path(cfg.dir.features_dir) / exp_name
    print(f"ouput_path: {output_path}")
    os.makedirs(output_path, exist_ok=True)

    create_feature(cfg, output_path)


if __name__ == "__main__":
    main()
