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
ARTICLE_DIR = PREPROCESS_DIR / "v0100_articles"
IMPRESSIONS_DIR = PREPROCESS_DIR / "v0300_impressions"


"""
Max (ner_clusters): 147
Max (entity_groups): 7
Max (topics): 61
Max (image_ids): 2
Max (category): 12
Max (subcategory): 34
"""


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


def _compute_metadata_ids(
    lf: pl.LazyFrame,
    input_col: str,
    output_col: str,
    dtype: pl.DataType,
    min_inview_count: int = 10_000,
    min_article_count: int = 50,
) -> pl.LazyFrame:
    if isinstance(lf.schema[input_col], pl.List):
        lf = lf.explode(input_col)
    lf = (
        lf.group_by(input_col)
        .agg(
            pl.col("inview_count").sum(),
            (pl.col("inview_count") > 0).sum().alias("article_count"),
        )
        .filter(
            (pl.col("inview_count") >= min_inview_count)
            & (pl.col("article_count") >= min_article_count)
        )
        .sort(["inview_count", input_col], descending=True)
        .with_row_index(name=output_col, offset=1)
        .select(input_col, pl.col(output_col).cast(dtype))
        .fill_null(0)
    )
    return lf


def compute_features(
    lf_articles: pl.LazyFrame,
    lf_impressions: pl.LazyFrame,
) -> pl.LazyFrame:
    lf_articles = (
        lf_articles.join(
            lf_impressions.filter(pl.col("impression_id") != 0)
            .select(pl.col("article_indices_inview").alias("article_index"))
            .explode("article_index")
            .group_by("article_index")
            .len("inview_count"),
            on="article_index",
            how="left",
        )
        .fill_null(0)
        .collect()
        .lazy()
    )
    for col, dtype in [
        ("ner_clusters", pl.UInt16),
        ("entity_groups", pl.UInt8),
        ("topics", pl.UInt8),
        ("image_ids", pl.UInt8),
        ("category", pl.UInt8),
        ("subcategory", pl.UInt8),
    ]:
        category_id_col = f"{col}_ids"
        explode = isinstance(lf_articles.schema[col], pl.List)
        lf_categories = _compute_metadata_ids(
            lf=lf_articles,
            input_col=col,
            output_col=category_id_col,
            dtype=dtype,
        )
        lf_categories = lf_categories.collect().lazy()
        logger.info(f"Max ({col}): {lf_categories.collect()[category_id_col].max()}")
        lf_articles = lf_articles.join(
            (lf_articles.explode(col) if explode else lf_articles)
            .select("article_index", col)
            .join(lf_categories, on=col)
            .select("article_index", category_id_col)
            .unique(maintain_order=True)
            .group_by("article_index")
            .agg(category_id_col),
            on="article_index",
            how="left",
        )
    lf_articles = lf_articles.select(
        pl.col("article_index"),
        pl.col("ner_clusters_ids").fill_null([0]),
        pl.col("entity_groups_ids").fill_null([0]),
        pl.col("topics_ids").fill_null([0]),
        pl.col("image_ids_ids").fill_null([0]),
        pl.col("category_ids").fill_null([0]),
        pl.col("subcategory_ids").fill_null([0]),
    ).sort("article_index")
    return lf_articles


@APP.command()
def main(
    overwrite: Annotated[Optional[bool], typer.Option("--overwrite/--skip")] = None,
):
    prepare_output_dir(overwrite=overwrite)

    with timer("Compute articles"):
        lf_articles = pl.scan_parquet(ARTICLE_DIR / "dataset.parquet")
        lf_impressions = pl.scan_parquet(IMPRESSIONS_DIR / "train" / "dataset.parquet")
        lf_output = compute_features(lf_articles, lf_impressions)
        df_output = lf_output.collect(streaming=True)
        logger.info(df_output)

    with timer("Test consistency"):
        assert (
            lf_articles.select("article_index")
            .collect()
            .equals(df_output.select("article_index"))
        )

    output_path = OUTPUT_DIR / "dataset.parquet"
    with timer(f"Output parquet: {output_path}"):
        output_path.parent.mkdir(parents=True, exist_ok=True)
        df_output.write_parquet(output_path, use_pyarrow=True)


if __name__ == "__main__":
    APP()
