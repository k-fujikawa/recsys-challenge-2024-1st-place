import os
import sys
from pathlib import Path

import hydra
import polars as pl
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig
from sklearn.preprocessing import OrdinalEncoder

PREFIX = "a"

KEY_COLUMNS = [
    "article_id",
]

USE_COLUMNS = [
    "total_days_last_modified_time_from_min",
    "total_seconds_last_modified_time_from_min_mod_3600",
]


def process_df(cfg, df):
    df = df.with_columns(
        [
            (pl.col("last_modified_time") - pl.col("last_modified_time").min())
            .dt.total_days()
            .alias("total_days_last_modified_time_from_min"),
            (
                (
                    pl.col("last_modified_time") - pl.col("last_modified_time").min()
                ).dt.total_seconds()
                % 3600
            ).alias("total_seconds_last_modified_time_from_min_mod_3600"),
        ]
    )
    return df


def create_feature(cfg: DictConfig, output_path):
    articles_path = Path("input/ebnerd_testset/ebnerd_testset") / "articles.parquet"
    articles_df = pl.read_parquet(articles_path)
    df = process_df(cfg, articles_df).select(KEY_COLUMNS + USE_COLUMNS)
    df = df.rename({col: f"{PREFIX}_{col}" for col in USE_COLUMNS})
    print(df)

    for data_name in ["train", "validation", "test"]:
        print(f"processing {data_name} data")
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
