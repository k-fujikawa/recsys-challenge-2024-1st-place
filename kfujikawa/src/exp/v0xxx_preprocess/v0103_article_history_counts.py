import shutil
import sys
from pathlib import Path
from typing import Optional

import polars as pl
import typer
from loguru import logger
from typing_extensions import Annotated

from exputils.const import PREPROCESS_DIR, TRAIN_START_UNIXTIME, SPLIT_INTERVAL
from exputils.utils import timer

APP = typer.Typer(pretty_exceptions_enable=False)
FILE_NAME = Path(__file__).stem
OUTPUT_DIR = PREPROCESS_DIR / FILE_NAME
ARTICLE_DIR = PREPROCESS_DIR / "v0100_articles"
USERS_DIR = PREPROCESS_DIR / "v0200_users"
IMPRESSIONS_DIR = PREPROCESS_DIR / "v0300_impressions"


def prepare_output_dir(overwrite: bool | None):
    if not OUTPUT_DIR.exists():
        pass
    elif overwrite or (overwrite is None and typer.confirm(f"Delete {OUTPUT_DIR}?")):
        logger.debug(f"Delete {OUTPUT_DIR}")
        shutil.rmtree(OUTPUT_DIR)
    else:
        logger.info(f"Skip to overwrite {OUTPUT_DIR}")
        sys.exit(0)
    return


def compute_elapsed_min_from_split_start(impression_ts):
    return (impression_ts - TRAIN_START_UNIXTIME) % SPLIT_INTERVAL // 60


def compute_features(
    lf_articles: pl.LazyFrame,
    lf_users: pl.LazyFrame,
    split_start_ts: int,
) -> pl.LazyFrame:
    lf_counts = (
        lf_users.select(
            pl.col("article_indices").alias("article_index"),
            pl.col("impression_ts"),
        )
        .explode("article_index", "impression_ts")
        .group_by("article_index")
        .agg(
            (pl.col("impression_ts") >= split_start_ts - 1 * 24 * 60 * 60)
            .sum()
            .alias("history_last_24h_counts")
            .cast(pl.Int32),
            (pl.col("impression_ts") >= split_start_ts - 1 * 60 * 60)
            .sum()
            .alias("history_last_1h_counts")
            .cast(pl.Int32),
        )
    )
    lf_outputs = (
        lf_articles.select("article_index")
        .join(lf_counts, on="article_index", how="left")
        .select(
            pl.col("article_index"),
            pl.col("history_last_24h_counts").fill_null(0),
            pl.col("history_last_1h_counts").fill_null(0),
        )
        .sort("article_index")
    )
    return lf_outputs


@APP.command()
def main(
    overwrite: Annotated[Optional[bool], typer.Option("--overwrite/--skip")] = None,
):
    prepare_output_dir(overwrite=overwrite)
    lf_articles = pl.scan_parquet(ARTICLE_DIR / "dataset.parquet")
    for i, split in enumerate(["train", "validation", "test"]):
        split_start_ts = TRAIN_START_UNIXTIME + SPLIT_INTERVAL * i
        with timer(f"Compute features ({split})"):
            lf_users = pl.scan_parquet(USERS_DIR / split / "dataset.parquet")
            lf_output = compute_features(
                lf_articles=lf_articles,
                lf_users=lf_users,
                split_start_ts=split_start_ts,
            )
            df_output = lf_output.collect(streaming=True)
            logger.info(df_output)

        with timer("Test consistency"):
            assert (
                lf_articles.select("article_index")
                .collect()
                .equals(df_output.select("article_index"))
            )

        output_path = OUTPUT_DIR / split / "dataset.parquet"
        with timer(f"Save features ({output_path})"):
            output_path.parent.mkdir(parents=True, exist_ok=True)
            df_output.write_parquet(output_path)


if __name__ == "__main__":
    APP()
