""" """

import itertools
import os
import sys
from pathlib import Path

import hydra
import polars as pl
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig

from utils.data import get_data_dirs

PREFIX = "y"

KEY_COLUMNS = [
    "user_id",
    "article_id",
]

USE_COLUMNS = [
    "transition_prob_from_first",
    "transition_count_from_first",
]


def process_df(cfg, history_df):
    # explode
    explode_df = history_df.explode(
        [
            "impression_time_fixed",
            "scroll_percentage_fixed",
            "article_id_fixed",
            "read_time_fixed",
        ]
    ).with_columns(
        pl.col("article_id_fixed").alias("article_id"),
    )
    explode_df = explode_df.sort(["user_id", "impression_time_fixed"])

    # 遷移を作成
    explode_df = explode_df.with_columns(
        pl.col("article_id").alias("from_article_id"),
        pl.col("article_id")
        .shift(-1)  # -1で順方向
        .over("user_id")
        .alias("to_article_id"),
    )

    # 同じ遷移はユーザーごとに1回にする
    if cfg.exp.is_user_unique:
        explode_df = explode_df.unique(["user_id", "from_article_id", "to_article_id"])

    transition_df = explode_df.select(["from_article_id", "to_article_id"]).filter(
        ~pl.col("to_article_id").is_null()
    )

    # 集約して確率計算
    transition_prob_df = (
        transition_df.group_by(["from_article_id", "to_article_id"])
        .agg(pl.col("from_article_id").count().alias("from_to_count"))
        .with_columns(
            pl.col("from_to_count").sum().over(["from_article_id"]).alias("from_count"),
        )
        .with_columns(
            (pl.col("from_to_count") / pl.col("from_count")).alias("transition_prob")
        )
        .sort(by=["from_article_id", "to_article_id"])
        .select(
            ["from_article_id", "to_article_id", "transition_prob", "from_to_count"]
        )
    )

    # ユーザーごとに最後に見たarticle からの遷移を考える
    last_click_df = (
        explode_df.group_by("user_id")
        .agg(pl.all().sort_by("impression_time_fixed").last())
        .select(["user_id", "from_article_id"])
    )

    # 遷移確率を結合

    y_df = (
        last_click_df.join(
            transition_prob_df, left_on="from_article_id", right_on="from_article_id"
        )
        .with_columns(
            [
                pl.col("to_article_id").alias("article_id"),
                pl.col("transition_prob").alias("transition_prob_from_first"),
                pl.col("from_to_count").alias("transition_count_from_first"),
            ]
        )
        .select(
            [
                "user_id",
                "article_id",
                "transition_prob_from_first",
                "transition_count_from_first",
            ]
        )
    )

    return y_df


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
