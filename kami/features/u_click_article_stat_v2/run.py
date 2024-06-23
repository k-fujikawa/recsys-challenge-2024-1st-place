"""
clickしたarticle の統計量を計算する
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

PREFIX = "u"

KEY_COLUMNS = [
    "user_id",
]

USE_COLUMNS = [
    "time_min_diff_click_publish_mean",
    "time_min_diff_click_publish_std",
    "total_inviews_mean",
    "total_inviews_std",
    "total_pageviews_mean",
    "total_pageviews_std",
    "total_read_time_mean",
    "total_read_time_std",
    "sentiment_score_mean",
    "sentiment_score_std",
]


def process_df(cfg, history_df, articles_df):
    explode_df = (
        history_df.explode(
            [
                "impression_time_fixed",
                "scroll_percentage_fixed",
                "article_id_fixed",
                "read_time_fixed",
            ]
        )
        .rename({"article_id_fixed": "article_id"})
        .join(articles_df, on="article_id", how="left")
    ).with_columns(
        (pl.col("impression_time_fixed") - pl.col("published_time"))
        .dt.total_minutes()
        .alias("time_min_diff_click_publish"),
    )

    group_df = explode_df.group_by("user_id").agg(
        list(
            itertools.chain(
                *[
                    [
                        pl.mean(col).alias(f"{col}_mean"),
                        pl.std(col).alias(f"{col}_std"),
                    ]
                    for col in [
                        "time_min_diff_click_publish",
                        "total_inviews",
                        "total_pageviews",
                        "total_read_time",
                        "sentiment_score",
                    ]
                ]
            )
        )
    )

    return group_df


def create_feature(cfg: DictConfig, output_path):
    input_dir = Path(cfg.dir.input_dir)
    size_name = cfg.exp.size_name
    data_dirs = get_data_dirs(input_dir, size_name)

    articles_path = Path("input/ebnerd_testset/ebnerd_testset") / "articles.parquet"
    articles_df = pl.read_parquet(articles_path)
    for data_name in ["train", "validation", "test"]:
        print(f"processing {data_name} data")
        history_df = pl.read_parquet(data_dirs[data_name] / "history.parquet")
        df = process_df(cfg, history_df, articles_df).select(KEY_COLUMNS + USE_COLUMNS)

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
