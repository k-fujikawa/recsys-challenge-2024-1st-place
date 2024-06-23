import os
import sys
from pathlib import Path

import hydra
import polars as pl
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig

from utils.data import get_data_dirs

PREFIX = "a"

KEY_COLUMNS = [
    "article_id",
]

USE_COLUMNS = [
    "click_rank",
    "click_count",
]


def process_df(cfg, df):
    explode_df = df.explode(
        [
            "impression_time_fixed",
            "scroll_percentage_fixed",
            "article_id_fixed",
            "read_time_fixed",
        ]
    ).with_columns(
        pl.col("article_id_fixed").alias("article_id"),
    )

    # ユーザーごとに1回にする
    if cfg.exp.is_user_unique:
        explode_df = explode_df.unique(["user_id", "article_id"])

    count_df = (
        explode_df["article_id_fixed"]
        .value_counts()
        .sort("count", descending=True)
        .with_row_index("rank")
        .with_columns(
            [
                pl.col("article_id_fixed").alias("article_id"),
                pl.col("rank").alias("click_rank"),
                pl.col("count").alias("click_count"),
            ]
        )
    )
    return count_df


def create_feature(cfg: DictConfig, output_path):
    input_dir = Path(cfg.dir.input_dir)
    size_name = cfg.exp.size_name
    data_dirs = get_data_dirs(input_dir, size_name)

    for data_name in ["train", "validation", "test"]:
        print(f"processing {data_name} data")
        history_df = pl.read_parquet(data_dirs[data_name] / "history.parquet")
        df = process_df(cfg, history_df).select(KEY_COLUMNS + USE_COLUMNS)

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
