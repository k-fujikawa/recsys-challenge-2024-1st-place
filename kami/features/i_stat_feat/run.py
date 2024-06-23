import os
import sys
from pathlib import Path

import hydra
import polars as pl
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig

from utils.data import get_data_dirs

PREFIX = "i"

KEY_COLUMNS = [
    # user_id とセットじゃないと test でimpression_id=0 が重複しているので問題になる
    "impression_id",  # key
    "user_id",  # key
]

USE_COLUMNS = [
    "impression_times_in_1h",
    "impression_times_in_24h",
    "elapsed_time_since_last_impression",
]


def process_df(cfg, behaviors_df):
    sort_df = behaviors_df.sort(["user_id", "impression_time"])

    df_1h = sort_df.rolling(
        index_column="impression_time", period="1h", group_by="user_id"
    ).agg(
        pl.col("impression_id").count().alias("impression_times_in_1h"),
    )
    df_24h = sort_df.rolling(
        index_column="impression_time", period="24h", group_by="user_id"
    ).agg(
        pl.col("impression_id").count().alias("impression_times_in_24h"),
    )

    result_df = pl.concat(
        [
            sort_df,
            df_1h[["impression_times_in_1h"]],
            df_24h[["impression_times_in_24h"]],
        ],
        how="horizontal",
    )
    result_df = result_df.with_columns(
        (
            pl.col("impression_time")
            - pl.col("impression_time").shift(1).alias("last_impression_time")
        )
        .dt.total_minutes()
        .alias("elapsed_time_since_last_impression")
    )

    return result_df


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
