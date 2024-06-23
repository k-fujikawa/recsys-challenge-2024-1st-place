#! /usr/bin/env python

from pathlib import Path
from typing import Iterable

import numpy as np
import polars as pl
import typer
from loguru import logger
from tqdm.auto import tqdm
from polars.testing import assert_series_equal

from ebrec.utils._python import zip_submission_file
from exputils.const import DATA_DIR
from exputils.utils import timer

APP = typer.Typer(pretty_exceptions_enable=False)
NUM_TEST_IMPRESSIONS = 13536710
FAKE_PREDICTION_PATH = DATA_DIR / "ebnerd" / "predictions.txt"


def write_submission_file(
    impression_ids: Iterable[int],
    prediction_scores: Iterable[any],
    path: Path = Path("predictions.txt"),
    rm_file: bool = True,
    filename_zip: str = None,
) -> None:
    path = Path(path)
    with open(path, "w") as f:
        for i, (impr_index, preds) in enumerate(
            tqdm(zip(impression_ids, prediction_scores), total=len(impression_ids)),
        ):
            # preds = rank_predictions_by_score(preds)
            preds = "[" + ",".join([str(i) for i in preds]) + "]"
            line = " ".join([str(impr_index), preds]) + "\n"
            f.write(line)
            if i == 0:
                print(f"Example: {line}")
    zip_submission_file(path=path, rm_file=rm_file, filename_zip=filename_zip)


def make_fake_prediction(
    preds_df: pl.DataFrame,
    rate: float = 0.147059,
    seed: int = 42,
) -> pl.DataFrame:
    np.random.seed(seed)
    with timer(f"Load: {FAKE_PREDICTION_PATH}"):
        random_df = (
            pl.scan_csv(
                FAKE_PREDICTION_PATH,
                has_header=False,
                new_columns=["impression_id", "rank"],
                separator=" ",
                low_memory=True,
            )
            .with_columns(
                pl.col("rank")
                .str.strip_chars("[]")
                .str.split(",")
                .cast(pl.List(pl.UInt8))
            )
            .collect(streaming=True)
        )

    preds_df = preds_df.with_columns(
        pl.Series(name="random_rank", values=random_df["rank"]),
    ).select(
        pl.col("impression_id"),
        pl.when(pl.lit(np.random.uniform(low=0.0, high=1.0, size=len(preds_df))) < rate)
        .then(pl.col("random_rank"))
        .otherwise(pl.col("rank"))
        .alias("rank"),
        pl.col("rank").list.len().alias("rank_len"),
        pl.col("random_rank").list.len().alias("random_rank_len"),
    )
    assert_series_equal(
        preds_df["rank_len"],
        preds_df["random_rank_len"],
        check_names=False,
    )
    preds_df = preds_df.drop(["rank_len", "random_rank_len"])
    return preds_df


@APP.command()
def make(
    prediction_path: Path = typer.Argument(help="Path to the prediction file."),
    fake: bool = False,
):
    parts = prediction_path.parts
    with timer(f"Load: {prediction_path}"):
        df = pl.read_parquet(prediction_path, columns=["impression_id", "rank"])
    if fake:
        df = make_fake_prediction(preds_df=df)
    assert len(df) == NUM_TEST_IMPRESSIONS
    logger.info(df)
    filename_zip = f"{'FAKE_' if fake else ''}{parts[-4][:48]}_{parts[-3][:7]}.zip"
    logger.info(f"Output: {filename_zip}")
    write_submission_file(
        impression_ids=df["impression_id"].to_numpy(),
        prediction_scores=df["rank"].to_numpy(),
        path=prediction_path.with_name("predictions.txt"),
        filename_zip=filename_zip,
        rm_file=True,
    )


if __name__ == "__main__":
    APP()
