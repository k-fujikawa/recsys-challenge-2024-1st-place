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


BASE_COLUMNS = [
    "total_inviews",
    "total_pageviews",
    "total_read_time",
    "sentiment_score",
]

USE_COLUMNS = list(
    itertools.chain(
        *[
            [
                f"{col}_rn",
                f"{col}_rate",
            ]
            for col in BASE_COLUMNS
        ]
    )
)


def process_df(cfg, candidate_df, articles_df):
    candidate_df = candidate_df.join(
        articles_df.select(["article_id"] + BASE_COLUMNS),
        on="article_id",
        how="left",
    )

    df = candidate_df.with_columns(
        [
            pl.col(col)
            .rank(descending=True)
            .over(
                [
                    "impression_id",
                    "user_id",
                ]
            )
            .alias(f"{col}_rn")
            for col in BASE_COLUMNS
        ]
        + [
            (
                pl.col(col)
                / pl.col(col)
                .sum()
                .over(
                    [
                        "impression_id",
                        "user_id",
                    ]
                )
            ).alias(f"{col}_rate")
            for col in BASE_COLUMNS
        ]
    )

    return df


def create_feature(cfg: DictConfig, output_path):
    size_name = cfg.exp.size_name

    articles_path = Path("input/ebnerd_testset/ebnerd_testset") / "articles.parquet"
    articles_df = pl.read_parquet(articles_path)
    for data_name in ["train", "validation", "test"]:
        print(f"processing {data_name} data")
        candidate_df = pl.read_parquet(
            Path(cfg.dir.candidate_dir) / size_name / f"{data_name}_candidate.parquet"
        )
        df = process_df(cfg, candidate_df, articles_df).select(USE_COLUMNS)

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
