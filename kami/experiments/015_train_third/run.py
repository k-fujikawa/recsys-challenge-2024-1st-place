"""
1st: trainで学習、validationで評価、testで予測
2nd: validationで学習、testで予測 (1stのイテレーション数を使う)
"""

import gc
import os
import pickle
import random
import sys
from pathlib import Path

import hydra
import lightgbm as lgb
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import polars as pl
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig, OmegaConf
from tqdm.auto import tqdm

import utils
import wandb
from ebrec.evaluation import MetricEvaluator
from ebrec.evaluation import MultiprocessingAucScore as AucScore
from ebrec.evaluation import MultiprocessingMrrScore as MrrScore
from ebrec.evaluation import MultiprocessingNdcgScore as NdcgScore
from ebrec.utils._python import write_submission_file
from utils.logger import get_logger
from wandb.lightgbm import log_summary, wandb_callback

logger = None

group_cols = ["impression_id", "user_id"]


def get_need_cols(cfg: DictConfig, cols: list) -> pl.DataFrame:
    need_cols = ["impression_id", "user_id", "article_id", "label"] + [
        col for col in cols if col not in cfg.exp.lgbm.unuse_cols
    ]
    if cfg.exp.article_stats_cols is False:
        need_cols = [
            col for col in need_cols if col not in cfg.leak_features.article_stats_cols
        ]
    if cfg.exp.past_impression_cols is False:
        need_cols = [
            col
            for col in need_cols
            if col not in cfg.leak_features.past_impression_cols
        ]
    if cfg.exp.future_impression_cols is False:
        need_cols = [
            col
            for col in need_cols
            if col not in cfg.leak_features.future_impression_cols
        ]
    return need_cols


def process_df(cfg: DictConfig, df: pl.DataFrame) -> pl.DataFrame:
    mul_cols_dict = cfg.exp.lgbm.mul_cols_dict
    if mul_cols_dict is not None and len(mul_cols_dict) > 0:
        df = df.with_columns(
            [
                (pl.col(cols[0]) * pl.col(cols[1])).alias(name)
                for name, cols in mul_cols_dict.items()
            ]
        )

    div_cols_dict = cfg.exp.lgbm.div_cols_dict
    if div_cols_dict is not None and len(div_cols_dict) > 0:
        df = df.with_columns(
            [
                (pl.col(cols[0]) / pl.col(cols[1])).alias(name)
                for name, cols in div_cols_dict.items()
            ]
        )

    norm_cols_dict = cfg.exp.lgbm.norm_cols_dict
    if norm_cols_dict is not None and len(norm_cols_dict) > 0:
        df = df.with_columns(
            [
                (pl.col(cols[0]) / pl.col(cols[1])).alias(name)
                for name, cols in norm_cols_dict.items()
            ]
        ).drop([cols[0] for cols in norm_cols_dict.values()])

    need_cols = get_need_cols(cfg, df.columns)
    df = df[need_cols]
    return df


def train_and_valid(
    cfg: DictConfig, train_df: pl.DataFrame, validation_df: pl.DataFrame
) -> lgb.Booster:
    unuse_cols = cfg.exp.lgbm.unuse_cols
    feature_cols = [col for col in train_df.columns if col not in unuse_cols]
    logger.info(f"{len(feature_cols)=} {feature_cols=}")
    label_col = cfg.exp.lgbm.label_col

    if cfg.exp.lgbm.params.two_rounds:
        # bool特徴をintに変換
        bool_cols = [
            col
            for col in [label_col] + feature_cols
            if train_df[col].dtype == pl.Boolean
        ]
        train_df = train_df.with_columns(
            [train_df[col].cast(pl.Int8) for col in bool_cols]
        )
        validation_df = validation_df.with_columns(
            [validation_df[col].cast(pl.Int8) for col in bool_cols]
        )

        with utils.trace("write csv"):
            # テキストファイルへ書き出し
            train_df[[label_col] + feature_cols].write_csv(
                "tmp_train.csv", include_header=False
            )
            validation_df[[label_col] + feature_cols].write_csv(
                "tmp_validation.csv", include_header=False
            )

    print("make lgb.Dataset")
    lgb_train_dataset = lgb.Dataset(
        "tmp_train.csv"
        if cfg.exp.lgbm.params.two_rounds
        else train_df[feature_cols].to_numpy().astype(np.float32),
        label=np.array(train_df[label_col]),
        feature_name=feature_cols,
    )
    lgb_valid_dataset = lgb.Dataset(
        "tmp_validation.csv"
        if cfg.exp.lgbm.params.two_rounds
        else validation_df[feature_cols].to_numpy().astype(np.float32),
        label=np.array(validation_df[label_col]),
        feature_name=feature_cols,
    )
    """
    if cfg.exp.lgbm.params.two_rounds:
        lgb_train_dataset.construct()
        lgb_valid_dataset.construct()
    """

    if cfg.exp.lgbm.params.objective == "lambdarank":
        print("make train group")
        train_group = (
            train_df.select(group_cols)
            .group_by(group_cols, maintain_order=True)
            .len()["len"]
            .to_list()
        )
        print("make validation group")
        valid_group = (
            validation_df.select(group_cols)
            .group_by(group_cols, maintain_order=True)
            .len()["len"]
            .to_list()
        )
        print("set group")
        lgb_train_dataset.set_group(train_group)
        lgb_valid_dataset.set_group(valid_group)
        cfg.exp.lgbm.params["ndcg_eval_at"] = cfg.exp.lgbm.ndcg_eval_at

    print("train")
    bst = lgb.train(
        OmegaConf.to_container(cfg.exp.lgbm.params, resolve=True),
        lgb_train_dataset,
        num_boost_round=cfg.exp.lgbm.num_boost_round,
        valid_sets=[lgb_valid_dataset],
        valid_names=["valid"],
        categorical_feature=cfg.exp.lgbm.cat_cols,
        callbacks=[
            wandb_callback(),
            lgb.early_stopping(
                stopping_rounds=cfg.exp.lgbm.early_stopping_round,
                verbose=True,
                first_metric_only=cfg.exp.lgbm.params.first_metric_only,
            ),
            lgb.log_evaluation(cfg.exp.lgbm.verbose_eval),
        ],
    )
    log_summary(bst, save_model_checkpoint=True)
    logger.info(
        f"best_itelation: {bst.best_iteration}, train: {bst.best_score['train']}, valid: {bst.best_score['valid']}"
    )
    return bst


def train_only(cfg: DictConfig, train_df: pl.DataFrame, iteration: int) -> lgb.Booster:
    unuse_cols = cfg.exp.lgbm.unuse_cols
    feature_cols = [col for col in train_df.columns if col not in unuse_cols]
    label_col = cfg.exp.lgbm.label_col

    if cfg.exp.lgbm.params.two_rounds:
        # bool特徴をintに変換
        bool_cols = [
            col
            for col in [label_col] + feature_cols
            if train_df[col].dtype == pl.Boolean
        ]
        train_df = train_df.with_columns(
            [train_df[col].cast(pl.Int8) for col in bool_cols]
        )
        # テキストファイルへ書き出し
        train_df[[label_col] + feature_cols].write_csv(
            "tmp_train.csv", include_header=False
        )
    print("make lgb.Dataset")
    lgb_train_dataset = lgb.Dataset(
        "tmp_train.csv"
        if cfg.exp.lgbm.params.two_rounds
        else train_df[feature_cols].to_numpy().astype(np.float32),
        label=np.array(train_df[label_col]),
        feature_name=feature_cols,
    )

    if cfg.exp.lgbm.params.objective == "lambdarank":
        print("make train group")
        train_group = (
            train_df.select(group_cols)
            .group_by(group_cols, maintain_order=True)
            .len()["len"]
            .to_list()
        )
        print("set group")
        lgb_train_dataset.set_group(train_group)
        cfg.exp.lgbm.params["ndcg_eval_at"] = cfg.exp.lgbm.ndcg_eval_at

    print("train")
    bst = lgb.train(
        OmegaConf.to_container(cfg.exp.lgbm.params, resolve=True),
        lgb_train_dataset,
        num_boost_round=iteration,
        valid_sets=[lgb_train_dataset],
        valid_names=["train"],
        categorical_feature=cfg.exp.lgbm.cat_cols,
        callbacks=[
            lgb.log_evaluation(cfg.exp.lgbm.verbose_eval),
        ],
    )
    logger.info(
        f"best_itelation: {bst.best_iteration}, train: {bst.best_score['train']}, valid: {bst.best_score['valid']}"
    )
    return bst


def predict(
    cfg: DictConfig, bst: lgb.Booster, test_df: pd.DataFrame, num_iteration: int
) -> pd.DataFrame:
    unuse_cols = cfg.exp.lgbm.unuse_cols
    feature_cols = [col for col in test_df.columns if col not in unuse_cols]
    # batch size で分割して予測
    batch_size = 100000
    y_pred = np.zeros(len(test_df))
    for i in tqdm(range(0, len(test_df), batch_size)):
        y_pred[i : i + batch_size] = bst.predict(
            test_df[feature_cols][i : i + batch_size], num_iteration=num_iteration
        )
    return y_pred


def save_model(cfg: DictConfig, bst: lgb.Booster, output_path: Path, name: int) -> None:
    with open(output_path / f"model_dict_{name}.pkl", "wb") as f:
        pickle.dump({"model": bst}, f)

    # save feature importance
    fig, ax = plt.subplots(figsize=(10, 20))
    ax = lgb.plot_importance(bst, importance_type="gain", ax=ax, max_num_features=100)
    fig.tight_layout()
    fig.savefig(output_path / f"importance_{name}.png")

    # importance を log に出力
    importance_df = pd.DataFrame(
        {
            "feature": bst.feature_name(),
            "importance": bst.feature_importance(importance_type="gain"),
        }
    )
    importance_df = importance_df.sort_values("importance", ascending=False)
    # 省略せずに表示
    pd.set_option("display.max_rows", None)
    logger.info(importance_df)
    logger.info(importance_df["feature"].to_list())


def make_result_df(df: pl.DataFrame, pred: np.ndarray):
    assert len(df) == len(pred)
    return (
        df.select(["impression_id", "user_id", "article_id"])
        .with_columns(pl.Series(name="pred", values=pred))
        .with_columns(
            pl.col("pred")
            .rank(method="ordinal", descending=True)
            .over(["impression_id", "user_id"])
            .alias("rank")
        )
        .group_by(["impression_id", "user_id"], maintain_order=True)
        .agg(pl.col("rank"), pl.col("pred"))
        .select(["impression_id", "rank", "pred"])
    )


def first_stage(cfg: DictConfig, output_path) -> None:
    print("first_stage")
    dataset_path = Path(cfg.exp.dataset_path)

    size_name = cfg.exp.size_name
    if "train" in cfg.exp.first_modes:
        with utils.trace("load datasets"):
            train_df = pl.read_parquet(
                str(dataset_path / size_name / "train_dataset.parquet"),
                
            )
            if cfg.exp.sampling_rate:
                print(f"{train_df.shape=}")
                random.seed(cfg.exp.seed)
                train_impression_ids = sorted(
                    train_df["impression_id"].unique().to_list()
                )
                use_train_impression_ids = random.sample(
                    train_impression_ids,
                    int(len(train_impression_ids) * cfg.exp.sampling_rate),
                )
                train_df = train_df.filter(
                    pl.col("impression_id").is_in(use_train_impression_ids)
                )
                print(f"{train_df.shape=}")
                gc.collect()
            validation_df = pl.read_parquet(
                str(dataset_path / size_name / "validation_dataset.parquet"),
                
            )
            if cfg.exp.sampling_rate:
                # validation
                print(f"{validation_df.shape=}")
                random.seed(cfg.exp.seed)
                validation_impression_ids = sorted(
                    validation_df["impression_id"].unique().to_list()
                )
                use_validation_impression_ids = random.sample(
                    validation_impression_ids,
                    int(len(validation_impression_ids) * cfg.exp.sampling_rate),
                )
                validation_df = validation_df.filter(
                    pl.col("impression_id").is_in(use_validation_impression_ids)
                )
                print(f"{validation_df.shape=}")
                gc.collect()

            train_df = process_df(cfg, train_df)
            gc.collect()
            validation_df = process_df(cfg, validation_df)
            gc.collect()

        with utils.trace("train and valid"):
            bst = train_and_valid(cfg, train_df, validation_df)
            save_model(cfg, bst, output_path, name="first_stage")

        del train_df, validation_df
        gc.collect()

    if "predict" in cfg.exp.first_modes:
        bst = None
        with open(output_path / "model_dict_first_stage.pkl", "rb") as f:
            bst = pickle.load(f)["model"]

        with utils.trace("predict validation"):
            validation_df = pl.read_parquet(
                str(dataset_path / size_name / "validation_dataset.parquet"),
                
            )
            validation_df = process_df(cfg, validation_df)
            y_valid_pred = predict(
                cfg, bst, validation_df, num_iteration=bst.best_iteration
            )
            validation_result_df = make_result_df(validation_df, y_valid_pred)
            validation_result_df.write_parquet(
                output_path / "validation_result_first.parquet"
            )
            print(f"{validation_result_df=}")
            del validation_df, validation_result_df
            gc.collect()
        with utils.trace("predict test"):
            test_df = pl.read_parquet(
                str(dataset_path / size_name / "test_dataset.parquet"),
                
            )
            print(test_df.shape)
            test_df = process_df(cfg, test_df)
            y_pred = predict(cfg, bst, test_df, num_iteration=bst.best_iteration)
            test_result_df = make_result_df(test_df, y_pred)
            test_result_df.write_parquet(output_path / "test_result_first.parquet")

            print(f"{test_result_df=}")
            impression_ids = test_result_df["impression_id"].to_list()
            prediction_scores = test_result_df["rank"].to_list()
            write_submission_file(
                impression_ids=impression_ids,
                prediction_scores=prediction_scores,
                path=output_path / "predictions.txt",
                filename_zip=f"predictions_{size_name}.zip",
            )
            del test_df, test_result_df
            gc.collect()

    if "eval" in cfg.exp.first_modes:
        with utils.trace("load datasets"):
            validation_df = pl.read_parquet(
                str(dataset_path / size_name / "validation_dataset.parquet"),
                
            )
            validation_df = process_df(cfg, validation_df)
            validation_result_df = pl.read_parquet(
                output_path / "validation_result_first.parquet",
                
            )
        with utils.trace("prepare eval validation"):
            labels = (
                validation_df.select(["impression_id", "user_id", "label"])
                .group_by(["impression_id", "user_id"], maintain_order=True)
                .agg(pl.col("label").cast(pl.Int32))["label"]
                .to_list()
            )
            predictions = validation_result_df.with_columns(
                pl.col("rank").list.eval(1 / pl.element())
            )["rank"].to_list()

        with utils.trace("eval validation"):
            metric_functions = []
            if cfg.exp.use_auc:
                metric_functions = [AucScore()]
            metric_functions += [NdcgScore(k=10)]

            metrics = MetricEvaluator(
                labels=labels,
                predictions=predictions,
                metric_functions=metric_functions,
            )
            metrics.evaluate()
            result_dict = metrics.evaluations

            logger.info(result_dict)
            wandb.log(result_dict)


def second_stage(cfg: DictConfig, output_path) -> None:
    print("second_stage")
    dataset_path = Path(cfg.exp.dataset_path)
    size_name = cfg.exp.size_name

    if "train" in cfg.exp.second_modes:
        with utils.trace("load datasets"):
            validation_df = pl.read_parquet(
                str(dataset_path / size_name / "validation_dataset.parquet"),
                
            )
            if cfg.exp.sampling_rate:
                random.seed(cfg.exp.seed)
                validation_impression_ids = sorted(
                    validation_df["impression_id"].unique().to_list()
                )
                use_validation_impression_ids = random.sample(
                    validation_impression_ids,
                    int(len(validation_impression_ids) * cfg.exp.sampling_rate),
                )
                validation_df = validation_df.filter(
                    pl.col("impression_id").is_in(use_validation_impression_ids)
                )
                gc.collect()
            validation_df = process_df(cfg, validation_df)

        with utils.trace("train only"):
            first_bst = None
            with open(output_path / "model_dict_first_stage.pkl", "rb") as f:
                first_bst = pickle.load(f)["model"]
            iteration = first_bst.best_iteration
            bst = train_only(cfg, validation_df, iteration)
            save_model(cfg, bst, output_path, name="second_stage")

        del validation_df
        gc.collect()

    if "predict" in cfg.exp.second_modes:
        first_bst = None
        with open(output_path / "model_dict_first_stage.pkl", "rb") as f:
            first_bst = pickle.load(f)["model"]
        iteration = first_bst.best_iteration
        bst = None
        with open(output_path / "model_dict_second_stage.pkl", "rb") as f:
            bst = pickle.load(f)["model"]

        with utils.trace("predict test"):
            test_df = pl.read_parquet(
                str(dataset_path / size_name / "test_dataset.parquet"),
                
            )
            test_df = process_df(cfg, test_df)
            y_pred = predict(cfg, bst, test_df, num_iteration=iteration)
            test_result_df = make_result_df(test_df, y_pred)
            test_result_df.write_parquet(output_path / "test_result_second.parquet")

            print(f"{test_result_df=}")
            impression_ids = test_result_df["impression_id"].to_list()
            prediction_scores = test_result_df["rank"].to_list()
            write_submission_file(
                impression_ids=impression_ids,
                prediction_scores=prediction_scores,
                path=output_path / "predictions.txt",
                filename_zip=f"predictions_{size_name}_second.zip",
            )
            del test_df, test_result_df
            gc.collect()


def third_stage(cfg: DictConfig, output_path) -> None:
    print("third_stage")
    dataset_path = Path(cfg.exp.dataset_path)
    size_name = cfg.exp.size_name

    if "train" in cfg.exp.third_modes:
        with utils.trace("load datasets"):
            train_df = pl.read_parquet(
                str(dataset_path / size_name / "train_dataset.parquet"),
                
            )
            if cfg.exp.sampling_rate:
                print(f"{train_df.shape=}")
                random.seed(cfg.exp.seed)
                train_impression_ids = sorted(
                    train_df["impression_id"].unique().to_list()
                )
                use_train_impression_ids = random.sample(
                    train_impression_ids,
                    int(len(train_impression_ids) * cfg.exp.sampling_rate),
                )
                train_df = train_df.filter(
                    pl.col("impression_id").is_in(use_train_impression_ids)
                )
                print(f"{train_df.shape=}")
                gc.collect()
            train_df = process_df(cfg, train_df)
            validation_df = pl.read_parquet(
                str(dataset_path / size_name / "validation_dataset.parquet"),
                
            )
            if cfg.exp.sampling_rate:
                random.seed(cfg.exp.seed)
                validation_impression_ids = sorted(
                    validation_df["impression_id"].unique().to_list()
                )
                use_validation_impression_ids = random.sample(
                    validation_impression_ids,
                    int(len(validation_impression_ids) * cfg.exp.sampling_rate),
                )
                validation_df = validation_df.filter(
                    pl.col("impression_id").is_in(use_validation_impression_ids)
                )
                gc.collect()
            validation_df = process_df(cfg, validation_df)

        train_df = pl.concat([train_df, validation_df])

        with utils.trace("train only"):
            first_bst = None
            with open(output_path / "model_dict_first_stage.pkl", "rb") as f:
                first_bst = pickle.load(f)["model"]
            iteration = first_bst.best_iteration
            bst = train_only(cfg, train_df, iteration)
            save_model(cfg, bst, output_path, name="third_stage")

        del train_df, validation_df
        gc.collect()

    if "predict" in cfg.exp.third_modes:
        first_bst = None
        with open(output_path / "model_dict_first_stage.pkl", "rb") as f:
            first_bst = pickle.load(f)["model"]
        iteration = first_bst.best_iteration
        bst = None
        with open(output_path / "model_dict_third_stage.pkl", "rb") as f:
            bst = pickle.load(f)["model"]

        with utils.trace("predict test"):
            test_df = pl.read_parquet(
                str(dataset_path / size_name / "test_dataset.parquet"),
                
            )
            test_df = process_df(cfg, test_df)
            y_pred = predict(cfg, bst, test_df, num_iteration=iteration)
            test_result_df = make_result_df(test_df, y_pred)
            test_result_df.write_parquet(output_path / "test_result_third.parquet")

            print(f"{test_result_df=}")
            impression_ids = test_result_df["impression_id"].to_list()
            prediction_scores = test_result_df["rank"].to_list()
            write_submission_file(
                impression_ids=impression_ids,
                prediction_scores=prediction_scores,
                path=output_path / "predictions.txt",
                filename_zip=f"predictions_{size_name}_third.zip",
            )
            del test_df, test_result_df
            gc.collect()


@hydra.main(version_base=None, config_path=".", config_name="config")
def main(cfg: DictConfig) -> None:
    runtime_choices = HydraConfig.get().runtime.choices
    exp_name = f"{Path(sys.argv[0]).parent.name}/{runtime_choices.exp}"

    print(f"exp_name: {exp_name}")
    output_path = Path(cfg.dir.exp_dir) / exp_name
    print(f"ouput_path: {output_path}")
    os.makedirs(output_path, exist_ok=True)

    global logger
    logger = get_logger(__name__, file_path=output_path / "run.log")

    logger.info(f"exp_name: {exp_name}")
    logger.info(f"ouput_path: {output_path}")
    logger.info(OmegaConf.to_yaml(cfg))

    wandb.init(
        project="recsys2024",
        name=exp_name,
        config=OmegaConf.to_container(cfg.exp, resolve=True),
        mode="disabled" if cfg.debug or cfg.exp.size_name == "demo" else "online",
    )

    first_stage(cfg, output_path)
    second_stage(cfg, output_path)
    third_stage(cfg, output_path)


if __name__ == "__main__":
    main()
