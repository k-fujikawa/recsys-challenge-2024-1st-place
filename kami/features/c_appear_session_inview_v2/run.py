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
from tqdm.auto import tqdm

from utils.data import get_data_dirs

PREFIX = "c_session"

ITERATION = 3
USE_COLUMNS = list(
    itertools.chain(
        *[
            [
                f"is_in_prev{i}_inview",
                f"is_in_next{i}_inview",
            ]
            for i in range(1, 1 + ITERATION)
        ]
    )
)


def process_df(cfg, behaviors_df):
    behaviors_df = behaviors_df.select(
        [
            "impression_id",
            "impression_time",
            "article_ids_inview",
            "session_id",
            "user_id",
        ]
    ).with_row_index(name="order")  # 後で戻せるように番号付与

    b_df = behaviors_df.sort(["session_id", "user_id", "impression_time"])

    tmp_df = b_df.sort(["session_id", "user_id", "impression_time"]).with_columns(
        [
            pl.col("article_ids_inview")
            .shift(1)
            .over(["session_id", "user_id"])
            .fill_null([])
            .alias("past_viewed_articles"),
            pl.col("article_ids_inview")
            .shift(-1)
            .over(["session_id", "user_id"])
            .fill_null([])
            .alias("future_viewed_articles"),
        ]
    )

    for i in tqdm(range(1, 1 + ITERATION)):
        tmp_df = tmp_df.with_columns(
            pl.col("past_viewed_articles").alias(f"past_viewed_articles_{i}"),
            pl.col("future_viewed_articles").alias(f"future_viewed_articles_{i}"),
        )

        if i == ITERATION:
            break
        tmp_df = tmp_df.with_columns(
            [
                pl.concat_list(
                    pl.col("article_ids_inview")
                    .shift(1)
                    .over(["session_id", "user_id"])
                    .fill_null([]),
                    pl.col("past_viewed_articles")
                    .shift(1)
                    .over(["session_id", "user_id"])
                    .fill_null([]),
                ).alias("past_viewed_articles"),
                pl.concat_list(
                    pl.col("article_ids_inview")
                    .shift(-1)
                    .over(["session_id", "user_id"])
                    .fill_null([]),
                    pl.col("future_viewed_articles")
                    .shift(-1)
                    .over(["session_id", "user_id"])
                    .fill_null([]),
                ).alias("future_viewed_articles"),
            ]
        )

    candidate_df = (
        tmp_df.sort("order")
        .explode("article_ids_inview")
        .rename({"article_ids_inview": "article_id"})
    )

    candidate_df = candidate_df.with_columns(
        [
            pl.col(f"past_viewed_articles_{i}")
            .list.contains(pl.col("article_id"))
            .alias(f"is_in_prev{i}_inview")
            for i in range(1, 1 + ITERATION)
        ]
        + [
            pl.col(f"future_viewed_articles_{i}")
            .list.contains(pl.col("article_id"))
            .alias(f"is_in_next{i}_inview")
            for i in range(1, 1 + ITERATION)
        ]
    )

    return candidate_df


def create_feature(cfg: DictConfig, output_path):
    input_dir = Path(cfg.dir.input_dir)
    size_name = cfg.exp.size_name
    data_dirs = get_data_dirs(input_dir, size_name)

    for data_name in ["train", "validation", "test"]:
        print(f"processing {data_name} data")
        behaviors_df = pl.read_parquet(data_dirs[data_name] / "behaviors.parquet")
        df = process_df(cfg, behaviors_df).select(USE_COLUMNS)

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
