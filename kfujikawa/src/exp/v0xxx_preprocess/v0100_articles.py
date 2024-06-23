import shutil
import sys
from pathlib import Path
from typing import Optional

import polars as pl
import typer
from loguru import logger
from typing_extensions import Annotated

from exputils.const import RAWDATA_DIRS, PREPROCESS_DIR
from exputils.utils import timer

APP = typer.Typer(pretty_exceptions_enable=False)
FILE_NAME = Path(__file__).stem
OUTPUT_DIR = PREPROCESS_DIR / FILE_NAME


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


def compute_articles(
    lf_articles: pl.LazyFrame,
) -> pl.LazyFrame:
    lf_articles = lf_articles.sort("article_id")
    lf_articles = lf_articles.with_row_index(name="article_index")
    lf_articles = lf_articles.with_columns(
        (pl.col("published_time").dt.timestamp() // 10**6)
        .cast(pl.Int32)
        .alias("published_ts"),
        pl.col("published_time").cast(pl.Date).alias("published_date"),
        pl.col("published_time").dt.weekday().alias("published_weekday"),
        pl.col("total_inviews").fill_null(0).cast(pl.Int32),
        pl.col("total_pageviews").fill_null(0).cast(pl.Int32),
        pl.col("total_read_time").fill_null(0).cast(pl.Int32),
        pl.col("article_index").cast(pl.Int32),
        pl.col("article_id").cast(pl.Int32),
        pl.col("image_ids").cast(pl.List(pl.Int32)),
    )
    lf_articles = lf_articles.drop(["last_modified_time"])
    lf_articles = lf_articles.select("article_index", pl.exclude("article_index"))
    lf_articles = lf_articles.sort("article_index")
    return lf_articles


@APP.command()
def main(
    overwrite: Annotated[Optional[bool], typer.Option("--overwrite/--skip")] = None,
):
    prepare_output_dir(overwrite=overwrite)

    with timer("Compute articles"):
        lf_articles = pl.scan_parquet(RAWDATA_DIRS["articles"])
        lf_output = compute_articles(lf_articles)
        df_output = lf_output.collect(streaming=True)
        logger.info(df_output)

    output_path = OUTPUT_DIR / "dataset.parquet"
    with timer(f"Output parquet: {output_path}"):
        output_path.parent.mkdir(parents=True, exist_ok=True)
        df_output.write_parquet(output_path, use_pyarrow=True)


if __name__ == "__main__":
    APP()
