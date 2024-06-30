import polars as pl

import optuna
import argparse
from utils import (
    BASE_DIR,
    read_target_df,
    get_impression_auc_mean,
    read_behavior_preds,
)


def optimize_weight_by_optuna(explode_pred_df, pred_cols):
    # data sampling
    mini_df = explode_pred_df.filter(
        explode_pred_df["impression_id"].is_in(
            explode_pred_df["impression_id"].unique().sort()[::100]
        )
    )
    mini_df = mini_df.join(
        mini_df["impression_id"].value_counts(), on="impression_id"
    ).with_columns((1 / pl.col("count")).alias("weight"))

    def objective(trial):
        tmp_params = {}
        for i, pred_name in enumerate(pred_cols):
            tmp_params[pred_name] = trial.suggest_float(pred_name, 0, 1)

        local_df = mini_df.clone()  # Create a local copy of mini_df
        pred_plse = pl.Series("pred_ensemble", [0] * len(local_df), pl.Float32)
        for pred_name, weight in tmp_params.items():
            pred_plse += local_df[pred_name] * weight

        local_df = local_df.with_columns(pred_plse)
        score = get_impression_auc_mean(
            local_df, target_col="target", pred_col="pred_ensemble"
        )
        return score

    # optimize weight with Optuna
    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=200)

    # 結果の表示
    print("Best value: {} (params: {})\n".format(study.best_value, study.best_params))
    return study.best_params


def main(model_paths, debug=False):
    #######################################
    # get best weight with validation preds
    #######################################
    behaivior_df, pred_cols = read_behavior_preds(model_paths, "validation", "first")
    target_df = read_target_df("validation", "large")
    target_df = target_df[["impression_id", "target", "target_article_id"]].rename(
        {"target_article_id": "article_ids_inview"}
    )
    explode_pred_df = behaivior_df.explode(["article_ids_inview"] + pred_cols)
    # add target
    explode_pred_df = target_df.join(
        explode_pred_df, on=["impression_id", "article_ids_inview"]
    )
    if debug:
        explode_pred_df = explode_pred_df.filter(
            explode_pred_df["impression_id"].is_in(
                explode_pred_df["impression_id"].unique().sort()[::1000]
            )
        )
    best_params = optimize_weight_by_optuna(explode_pred_df, pred_cols)
    print(f"{best_params=}")

    #######################################
    # weighted mean with test preds
    #######################################
    behaivior_df, pred_cols = read_behavior_preds(model_paths, "test", "third")
    explode_pred_df = behaivior_df.explode(["article_ids_inview"] + pred_cols)
    if debug:
        explode_pred_df = explode_pred_df.filter(
            explode_pred_df["impression_id"].is_in(
                explode_pred_df["impression_id"].unique().sort()[::1000]
            )
        )

    pred_ensemble = pl.Series(
        "pred_weighted_mean", [0] * len(explode_pred_df), pl.Float32
    )
    for pred_name, weight in best_params.items():
        pred_ensemble += explode_pred_df[pred_name] * weight
    explode_pred_df = explode_pred_df.with_columns(pred_ensemble)
    explode_pred_df.write_parquet(BASE_DIR / "output" / "test_weighted_mean.parquet")


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
