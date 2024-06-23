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


USE_COLUMNS = ["time_min_diff", "time_min_diff_rn"]


def process_df(cfg, behaviors_df, articles_df):
    candidate_df = (
        behaviors_df.select(
            ["impression_id", "article_ids_inview", "user_id", "impression_time"]
        )
        .explode("article_ids_inview")
        .rename({"article_ids_inview": "article_id"})
    )

    candidate_df = candidate_df.join(
        articles_df.select(["article_id", "published_time"]),
        on="article_id",
        how="left",
    )

    # published_time と time の差分を特徴量として追加
    df = candidate_df.with_columns(
        (pl.col("impression_time") - pl.col("published_time"))
        .dt.minutes()
        .alias("time_min_diff")
    ).with_columns(
        pl.col("time_min_diff")
        .rank()
        .over(["impression_id", "user_id"])
        .alias("time_min_diff_rn")
    )

    return df


def create_feature(cfg: DictConfig, output_path):
    input_dir = Path(cfg.dir.input_dir)
    size_name = cfg.exp.size_name
    data_dirs = get_data_dirs(input_dir, size_name)

    articles_path = Path("input/ebnerd_testset/ebnerd_testset") / "articles.parquet"
    articles_df = pl.read_parquet(articles_path)
    for data_name in ["train", "validation", "test"]:
        print(f"processing {data_name} data")
        behaviors_df = pl.read_parquet(data_dirs[data_name] / "behaviors.parquet")
        df = process_df(cfg, behaviors_df, articles_df).select(USE_COLUMNS)

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
