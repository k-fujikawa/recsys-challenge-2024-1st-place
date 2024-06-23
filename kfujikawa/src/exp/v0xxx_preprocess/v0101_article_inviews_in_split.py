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
    lf_impressions: pl.LazyFrame,
) -> pl.LazyFrame:
    lf_impressions = (
        lf_impressions.filter(pl.col("impression_id") != 0)
        .select(
            "article_indices_inview",
            "impression_ts",
            "read_time",
            "scroll_percentage",
        )
        .explode("article_indices_inview")
        .with_columns(
            pl.col("article_indices_inview").alias("article_index"),
            compute_elapsed_min_from_split_start(pl.col("impression_ts")).alias(
                "inview_elapsed_mins"
            ),
        )
        .group_by(["article_index", "inview_elapsed_mins"])
        .agg(
            pl.len().alias("inview_counts"),
            pl.sum("read_time").alias("read_time_sum"),
            pl.sum("scroll_percentage").alias("scroll_percentage_sum"),
            (pl.col("scroll_percentage") == 0).sum().alias("scroll_zero_counts"),
        )
        .sort("article_index", "inview_elapsed_mins")
        .group_by("article_index", maintain_order=True)
        .agg(
            pl.col("inview_elapsed_mins").cast(pl.Int16),
            pl.col("inview_counts").cast(pl.Int16),
            pl.col("read_time_sum").cast(pl.Int32),
            pl.col("scroll_percentage_sum").cast(pl.Int32),
            pl.col("scroll_zero_counts").cast(pl.Int16),
        )
    )
    lf_impressions = (
        lf_articles.select("article_index")
        .join(lf_impressions, on="article_index", how="left")
        .with_columns(
            pl.col("inview_elapsed_mins").fill_null([]),
            pl.col("inview_counts").fill_null([]),
            pl.col("read_time_sum").fill_null([]),
            pl.col("scroll_percentage_sum").fill_null([]),
            pl.col("scroll_zero_counts").fill_null([]),
        )
        .sort("article_index")
    )
    return lf_impressions


@APP.command()
def main(
    overwrite: Annotated[Optional[bool], typer.Option("--overwrite/--skip")] = None,
):
    prepare_output_dir(overwrite=overwrite)

    lf_articles = pl.scan_parquet(ARTICLE_DIR / "dataset.parquet")
    for split in ["train", "validation", "test"]:
        with timer(f"Compute features ({split})"):
            lf_impressions = pl.scan_parquet(
                IMPRESSIONS_DIR / split / "dataset.parquet"
            )
            lf_output = compute_features(
                lf_articles=lf_articles,
                lf_impressions=lf_impressions,
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
