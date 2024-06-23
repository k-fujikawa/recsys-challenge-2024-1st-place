from pathlib import Path

import numpy as np
import polars as pl
import sklearn.metrics
import typer
from loguru import logger
from tqdm import tqdm

from exputils.const import KAMI_DIR, KFUJIKAWA_DIR, PREPROCESS_DIR


APP = typer.Typer(pretty_exceptions_enable=False)
FILE_DIR = Path(__file__).parent.name
FILE_NAME = Path(__file__).stem
OUTPUT_DIR = KFUJIKAWA_DIR / FILE_DIR / FILE_NAME
KAMI_WEIGHTS = {
    KAMI_DIR / "015_train_third/large067_001": 2,
    KAMI_DIR / "016_catboost/large067": 1,
}
KAMI_SUFFIXES = {
    "validation": [
        "validation_result_first.parquet",
    ],
    "test": [
        "test_result_third.parquet",
    ],
}
KF_WEIGHTS = {
    KFUJIKAWA_DIR / "v1xxx_training/v1174_111_L8_128d_smpl3_drophist": 2,
    KFUJIKAWA_DIR / "v1xxx_training/v1170_111_L8_128d": 1,
}
KF_SUFFIXES = {
    "validation": [
        "fold_0/predictions/validation.parquet",
    ],
    "test": [
        "fold_2/predictions/test.parquet",
    ],
}
"""
AUC: 0.8791009544954949
"""


def _weighted_append(src_df, trg_df, weight):
    if src_df.is_empty():
        return trg_df
    else:
        assert (src_df["index"] == trg_df["index"]).all()
        return src_df.with_columns(pl.col("pred") + trg_df["pred"] * weight)


def _load_exploaded_predictions(path: str, pred_col: str = "pred"):
    return (
        pl.scan_parquet(path)
        .with_row_index()
        .select(
            "index",
            "impression_id",
            pl.col("pred").cast(pl.List(pl.Float64)),
        )
        .explode("pred")
        .collect()
    )


def _load_weighted_predictions(
    weights: dict[Path, float],
    suffixes: list[str],
):
    exploded_df = pl.DataFrame()
    for prefix, weight in tqdm(weights.items(), desc="Load predictions"):
        weight = weight / len(weights) / len(suffixes)
        for suffix in suffixes:
            _df = _load_exploaded_predictions(f"{prefix}/{suffix}")
            exploded_df = _weighted_append(exploded_df, _df, weight)
    # exploded_df = exploded_df.with_columns(pl.col("pred").rank() / pl.len())
    return exploded_df


def load_predictions(split: str):
    kami_preds_df = _load_weighted_predictions(
        weights=KAMI_WEIGHTS,
        suffixes=KAMI_SUFFIXES[split],
    )
    kf_preds_df = _load_weighted_predictions(
        weights=KF_WEIGHTS,
        suffixes=KF_SUFFIXES[split],
    )
    ensembled_df = pl.DataFrame(
        {
            "index": kami_preds_df["index"],
            "impression_id": kami_preds_df["impression_id"],
            "pred": (
                kami_preds_df["pred"] * sum(KAMI_WEIGHTS.values())
                + kf_preds_df["pred"] * sum(KF_WEIGHTS.values())
            ),
        }
    )
    # ensembled_df = ensembled_df.with_columns(pl.col("pred").rank() / pl.len())
    ensembled_df = ensembled_df.with_columns(
        (1 / (1 + np.exp(-pl.col("pred")))).alias("pred").cast(pl.Float32)
    )
    ensembled_df = (
        ensembled_df.group_by("index", maintain_order=True)
        .agg(
            pl.col("impression_id").first(),
            pl.col("pred"),
            (pl.col("pred").arg_sort(descending=True).arg_sort() + 1)
            .cast(pl.UInt8)
            .alias("rank"),
            (pl.col("pred").rank(method="min", descending=True) == 1)
            .cast(pl.Int8)
            .alias("pseudo_labels"),
            (pl.col("pred").max() - pl.col("pred").sort(descending=True).get(2)).alias(
                "pred_diff_top2"
            ),
        )
        .select(pl.exclude("index"))
    )
    return ensembled_df


@APP.command()
def valid():
    logger.debug(f"KF weight: {KF_WEIGHTS}")
    logger.debug(f"Kami suffixes: {KAMI_WEIGHTS}")
    logger.info("Load labeled dataset")
    labels_df = (
        pl.scan_parquet(
            PREPROCESS_DIR / "v0300_impressions" / "validation" / "dataset.parquet",
        )
        .with_row_index()
        .select("index", "impression_id", "labels", "in_small")
        .collect()
    )

    logger.info("Load predictions")
    preds_df = load_predictions("validation")

    output_path = OUTPUT_DIR / "validation.parquet"
    logger.info(f"Write predictions: {output_path}")
    logger.debug(preds_df)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    preds_df.write_parquet(output_path, use_pyarrow=True)

    logger.info("Evaluate")
    num_samples, accum_auc = 0, 0
    valid_df = labels_df.with_columns(preds_df["pred"]).filter(pl.col("in_small"))
    pbar = tqdm(valid_df.iter_rows(named=True), total=len(valid_df))
    for sample in pbar:
        accum_auc += sklearn.metrics.roc_auc_score(sample["labels"], sample["pred"])
        num_samples += 1
        pbar.set_postfix({"auc": accum_auc / num_samples})
        pbar.update()
    pbar.close()
    logger.info(f"AUC: {accum_auc / num_samples}")

    accuracy_df = (
        labels_df.with_columns(preds_df["pred"])
        .select("pred", "labels")
        .explode(["labels", "pred"])
        .with_columns(
            (pl.col("labels") == (pl.col("pred") > 0.5)).alias("is_correct"),
        )
    )
    output_path = OUTPUT_DIR / "accuracy.parquet"
    logger.info(f"Write accuracy: {output_path}")
    logger.debug(accuracy_df)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    accuracy_df.write_parquet(output_path, use_pyarrow=True)


@APP.command()
def test():
    logger.info("Load predictions")
    preds_df = load_predictions("test")

    output_path = OUTPUT_DIR / "test.parquet"
    logger.info(f"Write predictions: {output_path}")
    logger.debug(preds_df)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    preds_df.write_parquet(output_path, use_pyarrow=True)


if __name__ == "__main__":
    APP()
