import shutil
import sys
from pathlib import Path
from typing import Optional

import polars as pl
import typer
from loguru import logger
from typing_extensions import Annotated

from exputils.const import PREPROCESS_DIR
from exputils.utils import timer

APP = typer.Typer(pretty_exceptions_enable=False)
FILE_NAME = Path(__file__).stem
OUTPUT_DIR = PREPROCESS_DIR / FILE_NAME
ARTICLES_DIR = PREPROCESS_DIR / "v0100_articles"
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


def compute_features(
    lf_impressions: pl.LazyFrame,
) -> pl.LazyFrame:
    lf_impressions = (
        lf_impressions.sort("impression_ts")
        .group_by("user_index", maintain_order=True)
        .agg(
            pl.col("article_indices_inview").alias("user_inviews_in_split"),
            pl.col("impression_id").alias("impression_ids_in_split"),
            pl.col("impression_ts").alias("impression_ts_in_split"),
            pl.col("read_time")
            .mean()
            .cast(pl.Float32)
            .alias("read_time_mean_in_split"),
            pl.col("scroll_percentage")
            .mean()
            .cast(pl.Float32)
            .alias("scroll_percentage_mean_in_split"),
            (pl.col("scroll_percentage") == 0)
            .mean()
            .cast(pl.Float32)
            .alias("scroll_zero_ratio_in_split"),
        )
        .sort("user_index")
        .select(
            "user_index",
            "impression_ids_in_split",
            "user_inviews_in_split",
            "impression_ts_in_split",
            "read_time_mean_in_split",
            "scroll_percentage_mean_in_split",
            "scroll_zero_ratio_in_split",
        )
    )
    return lf_impressions


@APP.command()
def main(
    overwrite: Annotated[Optional[bool], typer.Option("--overwrite/--skip")] = None,
):
    prepare_output_dir(overwrite=overwrite)

    for split in ["train", "validation", "test"]:
        with timer(f"Compute features ({split})"):
            lf_impressions = pl.scan_parquet(
                IMPRESSIONS_DIR / split / "dataset.parquet"
            )
            lf_history = pl.scan_parquet(USERS_DIR / split / "dataset.parquet")
            lf_output = compute_features(
                lf_impressions=lf_impressions,
            )
            df_output = lf_output.collect(streaming=True)
            logger.info(df_output)

        with timer("Test consistency"):
            assert (
                lf_history.select("user_index")
                .collect()
                .equals(df_output.select("user_index"))
            )

        output_path = OUTPUT_DIR / split / "dataset.parquet"
        with timer(f"Save features ({output_path})"):
            output_path.parent.mkdir(parents=True, exist_ok=True)
            df_output.write_parquet(output_path)


if __name__ == "__main__":
    APP()
