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

PREFIX = "c"


USE_COLUMNS = [f"is_viewed_{i}" for i in [1, 2, 5, 10, 20]]


def process_df(cfg, behaviors_df):
    behaviors_df = behaviors_df.select(
        ["impression_id", "article_ids_inview", "user_id", "impression_time"]
    ).with_row_index(name="order")  # 後で戻せるように番号付与

    tmp_df = behaviors_df.sort(["user_id", "impression_time"]).with_columns(
        pl.col("article_ids_inview")
        .shift()
        .over("user_id")
        .fill_null([])
        .alias("viewed_articles")
    )

    for i in tqdm(range(1, 21)):  # 10回前までのimpressionを見る
        if i in [1, 2, 5, 10, 20]:
            tmp_df = tmp_df.with_columns(
                pl.col("viewed_articles").alias(f"viewed_articles_{i}")
            )
            if i == 20:
                break
        tmp_df = tmp_df.with_columns(
            pl.concat_list(
                pl.col("article_ids_inview").shift().over("user_id").fill_null([]),
                pl.col("viewed_articles").shift().over("user_id").fill_null([]),
            ).alias("viewed_articles")
        )

    candidate_df = (
        tmp_df.sort("order")
        .explode("article_ids_inview")
        .rename({"article_ids_inview": "article_id"})
    )

    candidate_df = candidate_df.with_columns(
        [
            pl.col("article_id")
            .is_in(pl.col(f"viewed_articles_{i}"))
            .alias(f"is_viewed_{i}")
            for i in [1, 2, 5, 10, 20]
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
