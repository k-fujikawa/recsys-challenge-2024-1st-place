import os
import zipfile
import tqdm
import numpy as np
import polars as pl
from typing import Iterable
from pathlib import Path
from sklearn import metrics
import multiprocessing

BASE_DIR = Path(__file__).parents[1].resolve()
DATA_DIR = BASE_DIR / "input"


def read_target_df(data_period="train", data_size="small"):
    # (impression_id, article_ids_inview, target)
    if data_period in ["train", "validation", "test"]:
        df_behavior = pl.read_parquet(
            DATA_DIR
            / "ekstra_original"
            / f"ebnerd_{data_size}"
            / data_period
            / "behaviors.parquet"
        )
    else:
        raise ValueError(f"Invalid data_period: {data_period}")

    df_target = (
        df_behavior.explode("article_ids_inview")
        .with_columns(
            pl.col("article_ids_inview")
            .is_in(pl.col("article_ids_clicked"))
            .alias("target")
            .cast(pl.Int8)
        )
        .rename({"article_ids_inview": "target_article_id"})[
            [
                "impression_id",
                "user_id",
                "target",
                "target_article_id",
                "impression_time",
            ]
        ]
    )
    return df_target


def read_preds(pred_dir_path: str, data_period: str, pred_type: str):
    # read 1st stage preds
    assert data_period in ["validation", "test"]
    assert pred_type in ["first", "second", "third"]
    if data_period == "validation":
        assert pred_type == "first"

    pred_user = pred_dir_path.split("/")[0]

    if pred_user == "kami":
        pred_name = "pred_" + pred_dir_path.split("/")[3]
        pred_df = pl.read_parquet(
            os.path.join(
                os.path.dirname(__file__),
                "../../",
                pred_dir_path,
                f"{data_period}_result_{pred_type}.parquet",
            )
        )
    if pred_user == "kfujikawa":
        pred_name = "pred_" + pred_dir_path.split("/")[4]
        folder_name = {"first": "fold_0", "second": "fold_1", "third": "fold_2"}[
            pred_type
        ]
        pred_df = pl.read_parquet(
            os.path.join(
                os.path.dirname(__file__),
                "../../",
                pred_dir_path,
                folder_name,
                "predictions",
                f"{data_period}.parquet",
            )
        )
    pred_df = pred_df.rename({"pred": pred_name})
    return pred_df, pred_name


def read_behavior_preds(model_path_list: list, data_period: str, pred_type: str):
    behaivior_df = pl.read_parquet(
        DATA_DIR / f"ekstra_original/ebnerd_large/{data_period}/behaviors.parquet",
        columns=["impression_id", "user_id", "article_ids_inview"],
    )

    for model_path in model_path_list:
        pred_df, pred_name = read_preds(model_path, data_period, pred_type)
        pred_df = pred_df.with_columns(behaivior_df["user_id"])
        behaivior_df = behaivior_df.join(
            pred_df[["impression_id", pred_name, "user_id"]],
            on=["impression_id", "user_id"],
        )
    preds_cols = [col for col in behaivior_df.columns if col.startswith("pred_")]
    return behaivior_df, preds_cols


# calc competition metric (multiprocessing)
def get_impression_auc(tmp_target_pred_ar):
    target_list = tmp_target_pred_ar[0]
    pred_list = tmp_target_pred_ar[1]
    return metrics.roc_auc_score(target_list, pred_list)


def get_impression_auc_mean(pred_df, target_col="target", pred_col="pred"):
    target_pred_ar = (
        pl.DataFrame(pred_df)
        .group_by("impression_id")
        .agg(
            [
                pl.col(target_col).alias("target_list"),
                pl.col(pred_col).alias("pred_list"),
            ]
        )[:, 1:3]
        .to_numpy()
    )
    with multiprocessing.Pool(multiprocessing.cpu_count()) as pool:
        auc_list = list(
            tqdm.tqdm(
                pool.imap_unordered(get_impression_auc, target_pred_ar),
                total=len(target_pred_ar),
            )
        )
    return np.mean(auc_list)


def write_submission_file(
    impression_ids: Iterable[int],
    prediction_scores: Iterable[any],
    path: Path = Path("predictions.txt"),
    rm_file: bool = True,
    filename_zip: str = None,
) -> None:
    """
    We align the submission file similar to MIND-format for users who are familar.

    Reference:
        https://github.com/recommenders-team/recommenders/blob/main/examples/00_quick_start/nrms_MIND.ipynb

    Example:
    >>> impression_ids = [237, 291, 320]
    >>> prediction_scores = [[0.2, 0.1, 0.3], [0.1, 0.2], [0.4, 0.2, 0.1, 0.3]]
    >>> write_submission_file(impression_ids, prediction_scores, path="predictions.txt", rm_file=False)
    ## Output file:
        237 [0.2,0.1,0.3]
        291 [0.1,0.2]
        320 [0.4,0.2,0.1,0.3]
    """
    path = Path(path)
    with open(path, "w") as f:
        for impr_index, preds in tqdm.tqdm(zip(impression_ids, prediction_scores)):
            preds = "[" + ",".join([str(i) for i in preds]) + "]"
            f.write(" ".join([str(impr_index), preds]) + "\n")
    # =>
    zip_submission_file(path=path, rm_file=rm_file, filename_zip=filename_zip)


def zip_submission_file(
    path: Path,
    filename_zip: str = None,
    verbose: bool = True,
    rm_file: bool = True,
) -> None:
    """
    Compresses a specified file into a ZIP archive within the same directory.

    Args:
        path (Path): The directory path where the file to be zipped and the resulting zip file will be located.
        filename_input (str, optional): The name of the file to be compressed. Defaults to the path.name.
        filename_zip (str, optional): The name of the output ZIP file. Defaults to "prediction.zip".
        verbose (bool, optional): If set to True, the function will print the process details. Defaults to True.
        rm_file (bool, optional): If set to True, the original file will be removed after compression. Defaults to True.

    Returns:
        None: This function does not return any value.
    """
    path = Path(path)
    if filename_zip:
        path_zip = path.parent.joinpath(filename_zip)
    else:
        path_zip = path.with_suffix(".zip")

    if path_zip.suffix != ".zip":
        raise ValueError(f"suffix for {path_zip.name} has to be '.zip'")
    if verbose:
        print(f"Zipping {path} to {path_zip}")
    f = zipfile.ZipFile(path_zip, "w", zipfile.ZIP_DEFLATED)
    f.write(path, arcname=path.name)
    f.close()
    if rm_file:
        path.unlink()


def rank_predictions_by_score(
    arr: Iterable[float],
) -> list[np.ndarray]:
    """
    Converts the prediction scores based on their ranking (1 for highest score,
    2 for second highest, etc.), effectively ranking prediction scores for each row.

    Reference:
        https://github.com/recommenders-team/recommenders/blob/main/examples/00_quick_start/nrms_MIND.ipynb

    >>> prediction_scores = [[0.2, 0.1, 0.3], [0.1, 0.2], [0.4, 0.2, 0.1, 0.3]]
    >>> [rank_predictions_by_score(row) for row in prediction_scores]
        [
            array([2, 3, 1]),
            array([2, 1]),
            array([1, 3, 4, 2])
        ]
    """
    return np.argsort(np.argsort(arr)[::-1]) + 1
