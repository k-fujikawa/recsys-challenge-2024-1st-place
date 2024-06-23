"""
1h/24h/期間全体での impression に対して、同じarticleが何回出現するかをカウントする
user・sessionごとに、過去・未来それぞれでのカウントを行う
また、その値を同じimpression内での割合に変換した特徴量も組み込む
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

PREFIX = "c_"

FIX_WINDOWS = ["1i", "2i", "3i", "5i"]  # ["1i"]  #
USE_TIMES = ["5m", "1h", "24h"]  # ["5m"]  #

BASE_COLUMNS = list(
    itertools.chain(
        *[
            [f"{group_name}_count_past_all", f"{group_name}_count_future_all"]
            + [
                f"{group_name}_count_past_{time_str}"
                for time_str in USE_TIMES + FIX_WINDOWS
            ]
            + [
                f"{group_name}_count_future_{time_str}"
                for time_str in USE_TIMES + FIX_WINDOWS
            ]
            + [f"{group_name}_count_{time_str}" for time_str in USE_TIMES + FIX_WINDOWS]
            for group_name in ["user", "session"]
        ]
    )
)

USE_COLUMNS = BASE_COLUMNS + [col + "_ratio" for col in BASE_COLUMNS]


def process_df(cfg, behaviors_df):
    behaviors_df = behaviors_df.select(
        [
            "impression_id",
            "impression_time",
            "article_ids_inview",
            "session_id",
            "user_id",
        ]
    )
    explode_df = (
        behaviors_df.explode("article_ids_inview")
        .rename({"article_ids_inview": "article_id"})
        .with_row_index(name="order")
    )  # 後で戻せるように番号付与

    # ソートして後で使うカラムを付与
    explode_df = explode_df.sort(["impression_time"]).with_columns(
        [
            # litのままだと rolling_sum と over の組み合わせで失敗することがある
            pl.lit(1).alias("const"),
            # 逆順にrollingしたいので、時間を逆順にする
            (
                pl.col("impression_time")
                - (pl.col("impression_time") - pl.col("impression_time").min())
                - (pl.col("impression_time") - pl.col("impression_time").min())
            )
            .dt.replace_time_zone(time_zone=None)
            .alias("reverse_impression_time"),
        ]
    )

    for group_name, group_subset in [
        ["session", ["user_id", "session_id", "article_id"]],
        ["user", ["user_id", "article_id"]],
    ]:
        print(f"processing {group_name}")
        explode_df = explode_df.with_columns(
            [
                (pl.col("user_id").cum_count().over(group_subset) - 1).alias(
                    f"{group_name}_count_past_all"
                ),
                (
                    pl.col("user_id").cum_count(reverse=True).over(group_subset) - 1
                ).alias(f"{group_name}_count_future_all"),
            ]
        )

        print("past")
        explode_df = explode_df.with_columns(
            [
                (
                    pl.col("const")
                    .rolling_sum(window_size=time_str, by="impression_time")
                    .over(group_subset)
                    - 1
                ).alias(f"{group_name}_count_past_{time_str}")
                for time_str in USE_TIMES
            ]
            + [
                (
                    pl.col("const").rolling_sum(window_size=time_str).over(group_subset)
                    - 1
                ).alias(f"{group_name}_count_past_{time_str}")
                for time_str in FIX_WINDOWS
            ]
        )

        print("future")
        explode_df = explode_df.reverse()
        explode_df = explode_df.with_columns(
            [
                (
                    pl.col("const")
                    .rolling_sum(window_size=time_str, by="reverse_impression_time")
                    .over(group_subset)
                    - 1
                ).alias(f"{group_name}_count_future_{time_str}")
                for time_str in USE_TIMES
            ]
            + [
                (
                    pl.col("const").rolling_sum(window_size=time_str).over(group_subset)
                    - 1
                ).alias(f"{group_name}_count_future_{time_str}")
                for time_str in FIX_WINDOWS
            ]
        )

        # 逆順にしたので元に戻す
        explode_df = explode_df.reverse()

        # past と　futer を合わせる
        explode_df = explode_df.with_columns(
            [
                (
                    pl.col(f"{group_name}_count_past_all")
                    + pl.col(f"{group_name}_count_future_all")
                ).alias(f"{group_name}_count_all"),
            ]
            + [
                (
                    pl.col(f"{group_name}_count_past_{time_str}")
                    + pl.col(f"{group_name}_count_future_{time_str}")
                ).alias(f"{group_name}_count_{time_str}")
                for time_str in USE_TIMES + FIX_WINDOWS
            ]
        )

    # 同一 impression 内での割合を求める
    for col in BASE_COLUMNS:
        explode_df = explode_df.with_columns(
            [
                (pl.col(col) / pl.col(col).sum().over("impression_id")).alias(
                    f"{col}_ratio"
                )
            ]
        )
    explode_df = explode_df.sort(["order"])
    return explode_df


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
