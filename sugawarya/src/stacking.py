import polars as pl
import numpy as np
import argparse

import tqdm
import itertools
import lightgbm as lgb
from sklearn.model_selection import GroupKFold
from utils import BASE_DIR, read_target_df, read_behavior_preds


def feature_engineering(explode_pred_df, raw_pred_cols):
    df = (
        explode_pred_df.join(
            explode_pred_df.group_by(["impression_id", "user_id"]).agg(
                pl.len().alias("impression_count")
            ),
            on=["impression_id", "user_id"],
        )
        .join(
            explode_pred_df.group_by(["article_ids_inview", "user_id"]).agg(
                pl.len().alias("article_count")
            ),
            on=["article_ids_inview", "user_id"],
        )
        .join(
            explode_pred_df.group_by(["user_id"]).agg(pl.len().alias("user_count")),
            on=["user_id"],
        )
        .with_columns(
            explode_pred_df[raw_pred_cols].mean_horizontal().alias("pred_mean")
        )
    )

    agg_expr_list1 = list(
        np.concatenate(
            [
                [
                    pl.col(raw_pred_col).mean().alias(f"{raw_pred_col}_mean"),
                    pl.col(raw_pred_col).max().alias(f"{raw_pred_col}_max"),
                    pl.col(raw_pred_col).min().alias(f"{raw_pred_col}_min"),
                    pl.col(raw_pred_col).std().alias(f"{raw_pred_col}_std"),
                ]
                for raw_pred_col in raw_pred_cols
            ]
        )
    )
    expr_list2 = list(
        np.concatenate(
            [
                [
                    (
                        (pl.col(raw_pred_col) - pl.col(f"{raw_pred_col}_mean"))
                        / pl.col(f"{raw_pred_col}_std")
                    ).alias(f"{raw_pred_col}_zscore"),
                    (
                        (pl.col(raw_pred_col) - pl.col(f"{raw_pred_col}_min"))
                        / (
                            pl.col(f"{raw_pred_col}_max")
                            - pl.col(f"{raw_pred_col}_min")
                        )
                    ).alias(f"{raw_pred_col}_normed_in_impression"),
                ]
                for raw_pred_col in raw_pred_cols
            ]
        )
    )
    df = df.join(
        df.group_by("impression_id").agg(agg_expr_list1), on="impression_id"
    ).with_columns(expr_list2)

    expr_list1 = [
        pl.col(pred_col).rank().over("impression_id").alias(f"{pred_col}_rank")
        for pred_col in raw_pred_cols
    ] + [
        pl.col(pred_col)
        .rank(descending=True)
        .over("impression_id")
        .alias(f"{pred_col}_rank_desc")
        for pred_col in raw_pred_cols
    ]
    expr_list2 = [
        (pl.col(f"{pred_col}_rank") / pl.col("impression_count")).alias(
            f"{pred_col}_normedrank"
        )
        for pred_col in raw_pred_cols
    ] + [
        (pl.col(f"{pred_col}_rank_desc") / pl.col("impression_count")).alias(
            f"{pred_col}_normedrank_desc"
        )
        for pred_col in raw_pred_cols
    ]

    expr_list3 = []

    for raw_pred_col1, raw_pred_col2 in itertools.combinations(raw_pred_cols, 2):
        for col1, col2 in [
            (raw_pred_col1, raw_pred_col2),
            (f"{raw_pred_col1}_rank", f"{raw_pred_col2}_rank"),
            (f"{raw_pred_col1}_rank_desc", f"{raw_pred_col2}_rank_desc"),
            (f"{raw_pred_col1}_normedrank", f"{raw_pred_col2}_normedrank"),
            (f"{raw_pred_col1}_normedrank_desc", f"{raw_pred_col2}_normedrank_desc"),
            (f"{raw_pred_col1}_zscore", f"{raw_pred_col2}_zscore"),
            (
                f"{raw_pred_col1}_normed_in_impression",
                f"{raw_pred_col2}_normed_in_impression",
            ),
        ]:
            expr_list3.append(
                (pl.col(col1) - pl.col(col2)).alias(f"feat_{col1}_{col2}_diff")
            )
            expr_list3.append(
                (pl.col(col1) / pl.col(col2)).alias(f"feat_{col1}_{col2}_ratio")
            )
            expr_list3.append(
                pl.max_horizontal(pl.col(col1), pl.col(col2)).alias(
                    f"feat_{col1}_{col2}_max"
                )
            )
            expr_list3.append(
                pl.min_horizontal(pl.col(col1), pl.col(col2)).alias(
                    f"feat_{col1}_{col2}_min"
                )
            )

    df = df.with_columns(expr_list1).with_columns(expr_list2).with_columns(expr_list3)
    return df


def stacking_training(df):
    gkf = GroupKFold(n_splits=4)
    rv_cols = ["impression_id", "target", "article_ids_inview", "user_id"]
    X_cols = [col for col in df.columns if col not in rv_cols]
    lgb_params = {
        "objective": "lambdarank",
        "metric": "ndcg",
        "ndcg_at": [5, 10, 20],
        "learning_rate": 0.1,
        "feature_fraction": 0.8,
        "bagging_fraction": 0.8,
        "bagging_freq": 1,
        "seed": 19930820,
        "max_bin": 1024,
        "verbose": 1,
    }
    pred_ar = np.zeros(len(df))
    models = []
    for fold, (train_idx, valid_idx) in enumerate(
        gkf.split(df, groups=df["impression_id"])
    ):
        train_df = df[train_idx]
        valid_df = df[valid_idx]
        # data sampling
        train_df = train_df.filter(
            train_df["impression_id"].is_in(
                train_df["impression_id"].unique().sort()[::10]
            )
        )

        tr_D = lgb.Dataset(
            train_df[X_cols].to_numpy().astype(np.float32),
            train_df["target"].to_numpy().astype(np.float32),
            group=train_df.group_by("impression_id")
            .len()
            .sort("impression_id")["len"]
            .to_numpy(),
        )
        va_D = lgb.Dataset(
            valid_df[X_cols].to_numpy().astype(np.float32),
            valid_df["target"].to_numpy().astype(np.float32),
            group=valid_df.group_by("impression_id")
            .len()
            .sort("impression_id")["len"]
            .to_numpy(),
        )
        model = lgb.train(
            lgb_params,
            tr_D,
            valid_sets=[va_D],
            num_boost_round=10000,
            callbacks=[lgb.early_stopping(50), lgb.log_evaluation(50)],
        )

        pred_ar[valid_idx] = model.predict(
            valid_df[X_cols].to_numpy().astype(np.float32)
        )
        models.append(model)
    return models, X_cols


def main(model_paths, debug=False):
    #######################################
    # training stacking model
    #######################################
    behaivior_df, pred_cols = read_behavior_preds(model_paths, "validation", "first")

    explode_pred_df = behaivior_df.explode(["article_ids_inview"] + pred_cols)
    if debug:
        explode_pred_df = explode_pred_df.filter(
            explode_pred_df["impression_id"].is_in(
                explode_pred_df["impression_id"].unique().sort()[::1000]
            )
        )
    df = feature_engineering(explode_pred_df, pred_cols)

    target_df = read_target_df("validation", "large")
    target_df = target_df[["impression_id", "target", "target_article_id"]].rename(
        {"target_article_id": "article_ids_inview"}
    )
    df = df.join(target_df, on=["impression_id", "article_ids_inview"])
    models, X_cols = stacking_training(df)

    #######################################
    # inference
    #######################################
    behaivior_df, pred_cols = read_behavior_preds(model_paths, "test", "third")
    explode_pred_df = behaivior_df.explode(["article_ids_inview"] + pred_cols)
    if debug:
        explode_pred_df = explode_pred_df.filter(
            explode_pred_df["impression_id"].is_in(
                explode_pred_df["impression_id"].unique().sort()[::1000]
            )
        )
    df = feature_engineering(explode_pred_df, pred_cols)
    pred_ar = np.zeros(len(df))
    for model in tqdm.tqdm(models, total=len(models)):
        pred_ar += model.predict(df[X_cols].to_numpy().astype(np.float32)) / len(models)
    pred_df = df[["impression_id", "user_id", "article_ids_inview"]].with_columns(
        pl.Series(pred_ar).alias("stacking_pred")
    )

    # save
    pred_df.write_parquet(BASE_DIR / "output" / "test_stacking.parquet")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()
    kami_models = [
        "kami/output/experiments/015_train_third/large067_001",
        "kami/output/experiments/016_catboost/large067",
    ]
    kfujikawa_models = [
        "kfujikawa/data/kfujikawa/v1xxx_training/v1157_111_fix_past_v2",
        "kfujikawa/data/kfujikawa/v1xxx_training/v1170_111_L8_128d",
        "kfujikawa/data/kfujikawa/v1xxx_training/v1184_111_PL_bert_L4_256d",
    ]
    main(kami_models + kfujikawa_models, args.debug)
