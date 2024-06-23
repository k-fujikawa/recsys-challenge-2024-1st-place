import shutil
import sys
from pathlib import Path
from typing import Optional

import numpy as np
import polars as pl
import typer
from loguru import logger
from typing_extensions import Annotated

from exputils.const import DATA_DIR, RAWDATA_DIRS, PREPROCESS_DIR
from exputils.utils import timer

APP = typer.Typer(pretty_exceptions_enable=False)
FILE_NAME = Path(__file__).stem
OUTPUT_DIR = PREPROCESS_DIR / FILE_NAME
ARTICLES_DIR = PREPROCESS_DIR / "v0100_articles"


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


def compute_history(
    lf_history: pl.LazyFrame,
    lf_articles: pl.LazyFrame,
    small_user_ids: np.ndarray,
) -> pl.LazyFrame:
    lf_history = (
        lf_history.explode(
            "impression_time_fixed",
            "scroll_percentage_fixed",
            "article_id_fixed",
            "read_time_fixed",
        )
        .with_columns(
            (pl.col("impression_time_fixed").dt.timestamp() // 10**6)
            .alias("impression_ts")
            .cast(pl.UInt32),
            pl.col("scroll_percentage_fixed")
            .fill_null(0)
            .cast(pl.UInt8)
            .alias("scroll_percentage"),
            pl.col("article_id_fixed").cast(pl.Int32).alias("article_id"),
            pl.col("read_time_fixed").cast(pl.UInt16).alias("read_time"),
        )
        .join(
            lf_articles.select("article_id", "article_index"),
            on="article_id",
            how="left",
            validate="m:1",
        )
        .drop("article_id")
        .group_by("user_id", maintain_order=True)
        .agg(
            pl.col("impression_ts"),
            pl.col("article_index").alias("article_indices"),
            pl.col("scroll_percentage"),
            pl.col("read_time"),
        )
        .with_columns(pl.col("user_id").is_in(small_user_ids).alias("in_small"))
        .sort("user_id")
        .with_row_index(name="user_index")
        .select(
            pl.col("user_index").cast(pl.Int32),
            pl.col("user_id").cast(pl.Int32),
            pl.col("impression_ts"),
            pl.col("article_indices"),
            pl.col("scroll_percentage"),
            pl.col("read_time"),
            pl.col("in_small"),
        )
    )
    return lf_history


@APP.command()
def main(
    overwrite: Annotated[Optional[bool], typer.Option("--overwrite/--skip")] = None,
):
    prepare_output_dir(overwrite=overwrite)

    lf_articles = pl.scan_parquet(ARTICLES_DIR / "dataset.parquet")

    for split in ["train", "validation", "test"]:
        lf_history = pl.scan_parquet(RAWDATA_DIRS[split] / "history.parquet")
        small_user_ids = np.array([])
        if split != "test":
            small_user_ids = (
                pl.read_parquet(
                    DATA_DIR / "ebnerd_small" / split / "history.parquet",
                    columns=["user_id"],
                )
                .get_column("user_id")
                .unique()
                .to_numpy()
                .squeeze()
            )

        with timer(f"Compute history ({split})"):
            lf_output = compute_history(
                lf_history=lf_history,
                lf_articles=lf_articles,
                small_user_ids=small_user_ids,
            )
            df_output = lf_output.collect(streaming=True)
            logger.info(df_output)

        with timer("Test consistency"):
            df_original = (
                lf_history.sort("user_id")
                .select("scroll_percentage_fixed")
                .collect(streaming=True)
            )
            x1 = df_original["scroll_percentage_fixed"].to_numpy()
            x2 = df_output["scroll_percentage"].to_numpy()
            for i in range(len(x1)):
                assert (np.where(np.isnan(x1[i]), 0, x1[i]) == x2[i]).all()

        output_path = OUTPUT_DIR / split / "dataset.parquet"
        with timer(f"Output parquet: {output_path}"):
            output_path.parent.mkdir(parents=True, exist_ok=True)
            df_output.write_parquet(output_path, use_pyarrow=True)


if __name__ == "__main__":
    APP()
