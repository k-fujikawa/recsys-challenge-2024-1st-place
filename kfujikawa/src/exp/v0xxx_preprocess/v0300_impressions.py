import shutil
import sys
from pathlib import Path
from typing import Optional

import polars as pl
import typer
from loguru import logger
from typing_extensions import Annotated

from ebrec.utils._behaviors import create_binary_labels_column
from exputils.const import RAWDATA_DIRS, PREPROCESS_DIR
from exputils.utils import timer

APP = typer.Typer(pretty_exceptions_enable=False)
FILE_NAME = Path(__file__).stem
OUTPUT_DIR = PREPROCESS_DIR / FILE_NAME
ARTICLES_DIR = PREPROCESS_DIR / "v0100_articles"
USERS_DIR = PREPROCESS_DIR / "v0200_users"


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


def compute_impressions(
    lf_impressions: pl.LazyFrame,
    lf_history: pl.LazyFrame,
    lf_articles: pl.LazyFrame,
) -> pl.LazyFrame:
    lf_impressions = lf_impressions.with_row_index(
        name="impression_index"
    ).with_columns(
        pl.col("user_id").cast(pl.Int32),
        pl.col("impression_index").cast(pl.Int32),
    )
    if "article_ids_clicked" not in lf_impressions.columns:
        lf_impressions = lf_impressions.with_columns(
            pl.lit([]).cast(pl.List(pl.Int32)).alias("article_ids_clicked"),
            pl.lit(0).cast(pl.UInt16).alias("next_read_time"),
            pl.lit(0).cast(pl.UInt8).alias("next_scroll_percentage"),
        )
    lf_impressions = lf_impressions.join(
        lf_history.select("user_id", "user_index", "in_small"),
        on="user_id",
        validate="m:1",
    )
    lf_impressions_inview = (
        lf_impressions.select("impression_index", "article_ids_inview")
        .explode("article_ids_inview")
        .cast(pl.Int32)
        .join(
            lf_articles.select("article_id", "article_index"),
            left_on="article_ids_inview",
            right_on="article_id",
        )
        .group_by("impression_index", maintain_order=True)
        .agg(pl.col("article_index").alias("article_indices_inview"))
    )
    lf_impressions_click = (
        lf_impressions.select("impression_index", "article_ids_clicked")
        .explode("article_ids_clicked")
        .cast(pl.Int32)
        .join(
            lf_articles.select("article_id", "article_index"),
            left_on="article_ids_clicked",
            right_on="article_id",
        )
        .group_by("impression_index", maintain_order=True)
        .agg(pl.col("article_index").alias("article_indices_clicked"))
    )
    lf_impressions = (
        lf_impressions.join(
            lf_impressions_inview,
            on="impression_index",
            how="left",
            validate="1:1",
        )
        .join(
            lf_impressions_click,
            on="impression_index",
            how="left",
            validate="1:1",
        )
        .pipe(
            create_binary_labels_column,  # type: ignore
            clicked_col="article_indices_clicked",
            inview_col="article_indices_inview",
            shuffle=False,
            seed=123,
        )
        .with_columns(
            (pl.col("impression_time").dt.timestamp() // 10**6)
            .cast(pl.Int32)
            .alias("impression_ts"),
        )
        .sort("impression_index")
        .select(
            pl.col("impression_index"),
            pl.col("impression_id"),
            pl.col("impression_ts"),
            pl.col("impression_time"),
            pl.col("user_index"),
            pl.col("session_id"),
            pl.col("read_time").fill_null(0).cast(pl.UInt16),
            pl.col("scroll_percentage").fill_null(0).cast(pl.UInt8),
            pl.col("device_type").cast(pl.Int8),
            pl.col("is_sso_user").cast(bool),
            pl.col("gender").fill_null(-1).cast(pl.Int8),
            pl.col("postcode").fill_null(-1).cast(pl.Int8),
            pl.col("age").fill_null(-1).cast(pl.Int8),
            pl.col("is_subscriber").cast(bool),
            pl.col("next_read_time").fill_null(0).cast(pl.UInt16),
            pl.col("next_scroll_percentage").fill_null(0).cast(pl.UInt8),
            pl.col("article_indices_inview").fill_null([]),
            pl.col("article_indices_clicked").fill_null([]),
            pl.col("in_small").cast(bool),
            pl.col("labels"),
        )
    )
    return lf_impressions


@APP.command()
def main(
    overwrite: Annotated[Optional[bool], typer.Option("--overwrite/--skip")] = None,
):
    prepare_output_dir(overwrite=overwrite)

    lf_articles = pl.scan_parquet(ARTICLES_DIR / "dataset.parquet")
    for split in ["train", "validation", "test"]:
        lf_history = pl.scan_parquet(USERS_DIR / split / "dataset.parquet")
        lf_impressions = pl.scan_parquet(RAWDATA_DIRS[split] / "behaviors.parquet")
        with timer(f"Compute history ({split})"):
            lf_output = compute_impressions(
                lf_impressions=lf_impressions,
                lf_history=lf_history,
                lf_articles=lf_articles,
            )
            df_output = lf_output.collect()
            logger.info(df_output)

        with timer("Test consistency"):
            assert (
                lf_impressions.select("impression_id")
                .collect()
                .equals(df_output.select("impression_id"))
            )

        output_path = OUTPUT_DIR / split / "dataset.parquet"
        with timer(f"Output parquet: {output_path}"):
            output_path.parent.mkdir(parents=True, exist_ok=True)
            df_output.write_parquet(output_path, use_pyarrow=True)


if __name__ == "__main__":
    APP()
