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
    "history_len",
    "scroll_percentage_fixed_min",
    "scroll_percentage_fixed_max",
    "scroll_percentage_fixed_mean",
    "scroll_percentage_fixed_sum",
    "scroll_percentage_fixed_skew",
    "scroll_percentage_fixed_std",
    "read_time_fixed_min",
    "read_time_fixed_max",
    "read_time_fixed_mean",
    "read_time_fixed_sum",
    "read_time_fixed_skew",
    "read_time_fixed_std",
]


def process_df(cfg, df):
    explode_df = df.explode(
        [
            "impression_time_fixed",
            "scroll_percentage_fixed",
            "article_id_fixed",
            "read_time_fixed",
        ]
    )

    group_df = explode_df.group_by("user_id").agg(
        [pl.len().alias("history_len")]
        + list(
            itertools.chain(
                *[
                    [
                        pl.min(col).alias(f"{col}_min"),
                        pl.max(col).alias(f"{col}_max"),
                        pl.mean(col).alias(f"{col}_mean"),
                        pl.sum(col).alias(f"{col}_sum"),
                        pl.col(col).skew().alias(f"{col}_skew"),
                        pl.std(col).alias(f"{col}_std"),
                    ]
                    for col in ["scroll_percentage_fixed", "read_time_fixed"]
                ]
            )
        )
    )

    return group_df


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
