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
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import polars as pl
from catboost import CatBoostRanker, Pool
from cloudpathlib import CloudPath
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

logger = None

group_cols = ["impression_id", "user_id"]


def get_need_cols(cfg: DictConfig, cols: list) -> pl.DataFrame:
    need_cols = ["impression_id", "user_id", "article_id", "label"] + [
        col for col in cols if col not in cfg.exp.catboost.unuse_cols
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
    mul_cols_dict = cfg.exp.catboost.mul_cols_dict
    if mul_cols_dict is not None and len(mul_cols_dict) > 0:
        df = df.with_columns(
            [
                (pl.col(cols[0]) * pl.col(cols[1])).alias(name)
                for name, cols in mul_cols_dict.items()
            ]
        )

    div_cols_dict = cfg.exp.catboost.div_cols_dict
    if div_cols_dict is not None and len(div_cols_dict) > 0:
        df = df.with_columns(
            [
                (pl.col(cols[0]) / pl.col(cols[1])).alias(name)
                for name, cols in div_cols_dict.items()
            ]
        )

    norm_cols_dict = cfg.exp.catboost.norm_cols_dict
    if norm_cols_dict is not None and len(norm_cols_dict) > 0:
        df = df.with_columns(
            [
                (pl.col(cols[0]) / pl.col(cols[1])).alias(name)
                for name, cols in norm_cols_dict.items()
            ]
        ).drop([cols[0] for cols in norm_cols_dict.values()])

    need_cols = get_need_cols(cfg, df.columns)
    df = df[need_cols]
    if cfg.debug:
        df = df.head(1000)
    return df


def train_and_valid(
    cfg: DictConfig, train_df: pl.DataFrame, validation_df: pl.DataFrame
):
    unuse_cols = cfg.exp.catboost.unuse_cols
    feature_cols = [col for col in train_df.columns if col not in unuse_cols]
    logger.info(f"{len(feature_cols)=} {feature_cols=}")
    label_col = cfg.exp.catboost.label_col

    train_group = (
        train_df.with_row_index("new_id")
        .with_columns(pl.col("new_id").first().over(group_cols).alias("new_id"))[
            "new_id"
        ]
        .to_list()
    )
    valid_group = (
        validation_df.with_row_index("new_id")
        .with_columns(pl.col("new_id").first().over(group_cols).alias("new_id"))[
            "new_id"
        ]
        .to_list()
    )

    train_pool = Pool(
        data=train_df[feature_cols].to_numpy(),
        label=train_df[label_col].to_numpy(),
        group_id=train_group,
        cat_features=list(cfg.exp.catboost.cat_cols),
    )
    valid_pool = Pool(
        data=validation_df[feature_cols].to_numpy(),
        label=validation_df[label_col].to_numpy(),
        group_id=valid_group,
        cat_features=list(cfg.exp.catboost.cat_cols),
    )

    model = CatBoostRanker(
        **OmegaConf.to_container(cfg.exp.catboost.params, resolve=True),
    )

    model.fit(
        train_pool,
        eval_set=valid_pool,
        use_best_model=True,
        verbose=cfg.exp.catboost.verbose_eval,
    )
    wandb.log(
        {
            "best_iteration": model.get_best_iteration(),
        }
    )

    return model


def train_only(cfg: DictConfig, train_df: pl.DataFrame, iteration: int):
    unuse_cols = cfg.exp.catboost.unuse_cols
    feature_cols = [col for col in train_df.columns if col not in unuse_cols]
    logger.info(f"{len(feature_cols)=} {feature_cols=}")
    label_col = cfg.exp.catboost.label_col

    train_group = (
        train_df.with_row_index("new_id")
        .with_columns(pl.col("new_id").first().over(group_cols).alias("new_id"))[
            "new_id"
        ]
        .to_list()
    )
    train_pool = Pool(
        data=train_df[feature_cols].to_numpy(),
        label=train_df[label_col].to_numpy(),
        group_id=train_group,
        cat_features=list(cfg.exp.catboost.cat_cols),
    )

    params = OmegaConf.to_container(cfg.exp.catboost.params, resolve=True)
    params["iterations"] = iteration
    model = CatBoostRanker(**params)

    model.fit(
        train_pool,
        use_best_model=True,
        verbose=cfg.exp.catboost.verbose_eval,
    )
    wandb.log(
        {
            "best_iteration": model.get_best_iteration(),
        }
    )

    return model


def predict(cfg: DictConfig, bst, test_df: pd.DataFrame) -> pd.DataFrame:
    unuse_cols = cfg.exp.catboost.unuse_cols
    feature_cols = [col for col in test_df.columns if col not in unuse_cols]
    # batch size で分割して予測
    batch_size = 100000
    y_pred = np.zeros(len(test_df))
    for i in tqdm(range(0, len(test_df), batch_size)):
        y_pred[i : i + batch_size] = bst.predict(
            test_df[feature_cols][i : i + batch_size].to_numpy()
        )
    return y_pred


def save_model(cfg: DictConfig, bst, output_path: Path, name: int) -> None:
    with open(output_path / f"model_dict_{name}.pkl", "wb") as f:
        pickle.dump({"model": bst}, f)


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
    dataset_path = CloudPath(cfg.exp.dataset_gcs_path)

    size_name = cfg.exp.size_name
    if "train" in cfg.exp.first_modes:
        with utils.trace("load datasets"):
            train_df = pl.read_parquet(
                str(dataset_path / size_name / "train_dataset.parquet"),
                retries=3,
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
                retries=3,
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
                retries=3,
            )
            validation_df = process_df(cfg, validation_df)
            y_valid_pred = predict(cfg, bst, validation_df)
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
                retries=3,
            )
            print(test_df.shape)
            test_df = process_df(cfg, test_df)
            y_pred = predict(cfg, bst, test_df)
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
                retries=3,
            )
            validation_df = process_df(cfg, validation_df)
            validation_result_df = pl.read_parquet(
                output_path / "validation_result_first.parquet",
                retries=3,
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
    dataset_path = CloudPath(cfg.exp.dataset_gcs_path)
    size_name = cfg.exp.size_name

    if "train" in cfg.exp.second_modes:
        with utils.trace("load datasets"):
            validation_df = pl.read_parquet(
                str(dataset_path / size_name / "validation_dataset.parquet"),
                retries=3,
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
            iteration = first_bst.get_best_iteration()
            bst = train_only(cfg, validation_df, iteration)
            save_model(cfg, bst, output_path, name="second_stage")

        del validation_df
        gc.collect()

    if "predict" in cfg.exp.second_modes:
        first_bst = None
        with open(output_path / "model_dict_first_stage.pkl", "rb") as f:
            first_bst = pickle.load(f)["model"]
        bst = None
        with open(output_path / "model_dict_second_stage.pkl", "rb") as f:
            bst = pickle.load(f)["model"]

        with utils.trace("predict test"):
            test_df = pl.read_parquet(
                str(dataset_path / size_name / "test_dataset.parquet"),
                retries=3,
            )
            test_df = process_df(cfg, test_df)
            y_pred = predict(cfg, bst, test_df)
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
    dataset_path = CloudPath(cfg.exp.dataset_gcs_path)
    size_name = cfg.exp.size_name

    if "train" in cfg.exp.third_modes:
        with utils.trace("load datasets"):
            train_df = pl.read_parquet(
                str(dataset_path / size_name / "train_dataset.parquet"),
                retries=3,
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
                retries=3,
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
            iteration = first_bst.get_best_iteration()
            bst = train_only(cfg, train_df, iteration)
            save_model(cfg, bst, output_path, name="third_stage")

        del train_df, validation_df
        gc.collect()

    if "predict" in cfg.exp.third_modes:
        first_bst = None
        with open(output_path / "model_dict_first_stage.pkl", "rb") as f:
            first_bst = pickle.load(f)["model"]
        bst = None
        with open(output_path / "model_dict_third_stage.pkl", "rb") as f:
            bst = pickle.load(f)["model"]

        with utils.trace("predict test"):
            test_df = pl.read_parquet(
                str(dataset_path / size_name / "test_dataset.parquet"),
                retries=3,
            )
            test_df = process_df(cfg, test_df)
            y_pred = predict(cfg, bst, test_df)
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
