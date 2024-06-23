"""
未来のimpressionに出現するかどうかを特徴量として作成する（本来ならリークになるので注意）
"""

import os
import sys
from pathlib import Path

import hydra
import polars as pl
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig

from utils.data import get_data_dirs

PREFIX = "x"

KEY_COLUMNS = [
    "impression_id",
    "user_id",
    "article_id",
]

USE_COLUMNS = ["is_in_next_imp", "next_imp_len"]


def process_df(cfg, behaviors_df):
    behaviors_df = behaviors_df.sort(["user_id", "impression_time"])

    # 時間的に次のimpressionを取得
    before_explode_df = behaviors_df.with_columns(
        pl.col("article_ids_inview").shift(-1).over(["user_id"]).alias("next_imp")
    ).select(["impression_id", "user_id", "article_ids_inview", "next_imp"])

    # explode
    explode_df = before_explode_df.explode("article_ids_inview")

    # 次のimpressionに含まれるかを見る
    x_df = explode_df.with_columns(
        [
            pl.col("article_ids_inview")
            .is_in(pl.col("next_imp"))
            .alias("is_in_next_imp"),
            pl.col("next_imp").list.len().alias("next_imp_len"),
        ]
    ).rename({"article_ids_inview": "article_id"})

    return x_df


def create_feature(cfg: DictConfig, output_path):
    input_dir = Path(cfg.dir.input_dir)
    size_name = cfg.exp.size_name
    data_dirs = get_data_dirs(input_dir, size_name)

    for data_name in ["train", "validation", "test"]:
        print(f"processing {data_name} data")
        behaviors_df = pl.read_parquet(data_dirs[data_name] / "behaviors.parquet")
        df = process_df(cfg, behaviors_df).select(KEY_COLUMNS + USE_COLUMNS)

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
