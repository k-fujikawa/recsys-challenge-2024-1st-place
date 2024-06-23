"""
候補と生成した特徴量を結合して、lightgbmで学習するためのデータセットを作成する
"""

import gc
import os
import sys
from pathlib import Path

import gcsfs
import hydra
import polars as pl
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig, OmegaConf

import utils
from utils.data import get_data_dirs

os.environ["POLARS_CONCURRENCY_BUDGET"] = (
    "5"  # 大きいとlarge で失敗することがあるため小さめに設定したが関係ないかも
)


def create_dataset(cfg: DictConfig, data_name: str = "train"):
    size_name = cfg.exp.size_name
    result_df = pl.read_parquet(
        str(
            Path(cfg.exp.candidate_path)
            / size_name
            / f"{data_name}_candidate.parquet"
        ),
        
    )

    # 特徴量結合
    for feat_dir in cfg.exp.features.c:
        with utils.trace(f"feat_dir: {feat_dir}"):
            path = str(Path(feat_dir) / size_name / f"{data_name}_feat.parquet")
            feat_df = pl.read_parquet(
                path,
                
            )
            # 横方向に結合
            assert result_df.shape[0] == feat_df.shape[0]
            result_df = pl.concat([result_df, feat_df], how="horizontal")
            unuse_cols = [col for col in cfg.exp.unuse_cols if col in result_df.columns]
            result_df = result_df.drop(unuse_cols)

    for feat_dir in cfg.exp.features.factor_paths:
        with utils.trace(f"feat_dir: {feat_dir}"):
            user_factor = pl.read_parquet(
                str(Path(feat_dir) / size_name / "user_feat.parquet"),
                
            )
            result_df = result_df.join(user_factor, how="left", on=["user_id"])
            article_factor = pl.read_parquet(
                str(Path(feat_dir) / size_name / "article_feat.parquet"),
                
            )
            result_df = result_df.join(article_factor, how="left", on=["article_id"])
            unuse_cols = [col for col in cfg.exp.unuse_cols if col in result_df.columns]
            result_df = result_df.drop(unuse_cols)

    for feat_dir in cfg.exp.features.y:
        with utils.trace(f"feat_dir: {feat_dir}"):
            path = str(Path(feat_dir) / size_name / f"{data_name}_feat.parquet")
            feat_df = pl.read_parquet(
                path,
                
            )
            result_df = result_df.join(
                feat_df, how="left", on=["user_id", "article_id"]
            )
            unuse_cols = [col for col in cfg.exp.unuse_cols if col in result_df.columns]
            result_df = result_df.drop(unuse_cols)

    for feat_dir in cfg.exp.features.a:
        with utils.trace(f"feat_dir: {feat_dir}"):
            path = str(Path(feat_dir) / size_name / f"{data_name}_feat.parquet")
            feat_df = pl.read_parquet(
                path,
                
            )
            result_df = result_df.join(feat_df, how="left", on=["article_id"])
            unuse_cols = [col for col in cfg.exp.unuse_cols if col in result_df.columns]
            result_df = result_df.drop(unuse_cols)

    for feat_dir in cfg.exp.features.i:
        with utils.trace(f"feat_dir: {feat_dir}"):
            path = str(Path(feat_dir) / size_name / f"{data_name}_feat.parquet")
            feat_df = pl.read_parquet(
                path,
                
            )
            result_df = result_df.join(
                feat_df, how="left", on=["impression_id", "user_id"]
            )
            unuse_cols = [col for col in cfg.exp.unuse_cols if col in result_df.columns]
            result_df = result_df.drop(unuse_cols)

    for feat_dir in cfg.exp.features.u:
        with utils.trace(f"feat_dir: {feat_dir}"):
            path = str(Path(feat_dir) / size_name / f"{data_name}_feat.parquet")
            feat_df = pl.read_parquet(
                path,
                
            )
            result_df = result_df.join(feat_df, how="left", on=["user_id"])
            unuse_cols = [col for col in cfg.exp.unuse_cols if col in result_df.columns]
            result_df = result_df.drop(unuse_cols)

    for feat_dir in cfg.exp.features.x:
        with utils.trace(f"feat_dir: {feat_dir}"):
            path = str(Path(feat_dir) / size_name / f"{data_name}_feat.parquet")
            feat_df = pl.read_parquet(
                path,
                
            )
            result_df = result_df.join(
                feat_df, how="left", on=["impression_id", "user_id", "article_id"]
            )
            unuse_cols = [col for col in cfg.exp.unuse_cols if col in result_df.columns]
            result_df = result_df.drop(unuse_cols)
    # drop unuse_cols
    unuse_cols = [col for col in cfg.exp.unuse_cols if col in result_df.columns]
    result_df = result_df.drop(unuse_cols)

    # float64 -> float32
    float_cols = [
        col for col in result_df.columns if result_df[col].dtype == pl.Float64
    ]
    result_df = result_df.with_columns(
        [result_df[col].cast(pl.Float32) for col in float_cols]
    )
    return result_df


@hydra.main(version_base=None, config_path=".", config_name="config")
def main(cfg: DictConfig) -> None:
    runtime_choices = HydraConfig.get().runtime.choices
    exp_name = f"{Path(sys.argv[0]).parent.name}/{runtime_choices.exp}"

    print(f"exp_name: {exp_name}")
    output_path = Path(cfg.dir.preprocess_dir) / exp_name
    print(f"ouput_path: {output_path}")
    os.makedirs(output_path, exist_ok=True)

    print(OmegaConf.to_yaml(cfg))

    fs = gcsfs.GCSFileSystem()

    for data_name in [
        "train",
        "validation",
        "test",
    ]:
        destination = (
            f"{cfg.exp.base_path}/preprocess/{exp_name}/{data_name}_dataset.parquet"
        )
        print(destination)
        with utils.trace(f"processing {data_name} data"):
            dataset_df = create_dataset(cfg, data_name)
            print(f"dataset_df: {dataset_df}")

            dataset_df.write_parquet(output_path / f"{data_name}_dataset.parquet")
            """
            with fs.open(destination, mode="wb") as f:
                dataset_df.write_parquet(f)
            """
            del dataset_df
            gc.collect()


if __name__ == "__main__":
    main()
