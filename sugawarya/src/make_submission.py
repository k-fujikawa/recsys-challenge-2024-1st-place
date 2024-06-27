import tqdm
import multiprocessing
import polars as pl
from utils import BASE_DIR, write_submission_file, rank_predictions_by_score


def get_rank(arr):
    _idx = arr[0]
    _impression_id = arr[1]
    _user_id = arr[2]
    _pred = arr[3]
    return [_idx, _impression_id, _user_id, list(rank_predictions_by_score(_pred))]


def main():
    df1 = pl.read_parquet(BASE_DIR / "output" / "test_stacking.parquet")
    df2 = pl.read_parquet(BASE_DIR / "output" / "test_weighted_mean.parquet")
    pred_df = df1.with_columns(df2["pred_weighted_mean"]).with_columns(
        (pl.col("stacking_pred") + pl.col("pred_weighted_mean")).alias("pred")
    )

    pred_df = pred_df.group_by(["impression_id", "user_id"], maintain_order=True).agg(
        pl.col("pred"), pl.col("article_ids_inview")
    )
    pred_df_pd = pred_df[["impression_id", "user_id", "pred"]].to_pandas()

    with multiprocessing.Pool(multiprocessing.cpu_count()) as p:
        tmp_lst = list(
            tqdm.tqdm(
                p.imap_unordered(get_rank, pred_df_pd.reset_index().values),
                total=len(pred_df_pd),
            )
        )
    pred_df = pl.DataFrame(
        tmp_lst, schema=["idx_num", "impression_id", "user_id", "prediction_scores"]
    ).sort("idx_num")

    impression_ids = pred_df["impression_id"].to_list()
    prediction_scores = pred_df["prediction_scores"].to_list()
    write_submission_file(
        impression_ids=impression_ids,
        prediction_scores=prediction_scores,
        path="predictions.txt",
        filename_zip=BASE_DIR / "output" / "v999_final_submission.zip",
    )


if __name__ == "__main__":
    main()
