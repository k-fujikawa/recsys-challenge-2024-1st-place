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
        lf_impressions.select("impression_index", "user_index")
        .join(
            lf_impressions.group_by("user_index", maintain_order=True).len(
                "num_impressions"
            ),
            on="user_index",
        )
        .sort("impression_index")
        .select(
            pl.col("impression_index"),
            pl.col("num_impressions").cast(pl.Int16),
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
            lf_output = compute_features(
                lf_impressions=lf_impressions,
            )
            df_output = lf_output.collect(streaming=True)
            logger.info(df_output)

        with timer("Test consistency"):
            assert (
                lf_impressions.select("impression_index")
                .collect()
                .equals(df_output.select("impression_index"))
            )

        output_path = OUTPUT_DIR / split / "dataset.parquet"
        with timer(f"Save features ({output_path})"):
            output_path.parent.mkdir(parents=True, exist_ok=True)
            df_output.write_parquet(output_path)


if __name__ == "__main__":
    APP()
