import itertools
import os
import sys
from pathlib import Path

import hydra
import numpy as np
import polars as pl
from hydra.core.hydra_config import HydraConfig
from implicit.cpu.als import AlternatingLeastSquares
from omegaconf import DictConfig
from scipy.sparse import csr_matrix

from utils.data import get_data_dirs


def process_df(cfg, behaviors_df):
    base_df = (
        behaviors_df.select(["impression_id", "article_ids_inview"])
        .explode("article_ids_inview")
        .rename({"article_ids_inview": "article_id"})
    )

    # user
    impression_id_list = base_df["impression_id"].unique(maintain_order=True).to_list()
    new_impression_id_list = np.arange(len(impression_id_list))
    impression_id_map = dict(zip(impression_id_list, new_impression_id_list))

    # article
    article_id_list = base_df["article_id"].unique(maintain_order=True).to_list()
    new_article_id_list = np.arange(len(article_id_list))
    article_id_map = dict(zip(article_id_list, new_article_id_list))

    # map
    base_df = base_df.with_columns(
        [
            pl.col("impression_id")
            .replace(impression_id_map)
            .alias("new_impression_id"),
            pl.col("article_id").replace(article_id_map).alias("new_article_id"),
        ]
    )

    count_df = base_df.group_by(["new_impression_id", "new_article_id"]).len()
    matrix = csr_matrix(
        (
            count_df["len"].to_numpy(),
            (
                count_df["new_impression_id"].to_numpy(),
                count_df["new_article_id"].to_numpy(),
            ),
        ),
    )
    model = AlternatingLeastSquares(
        factors=cfg.exp.n_factors,
        regularization=cfg.exp.regularization,
        alpha=1.0,
        use_native=True,
        use_cg=True,
        iterations=cfg.exp.iterations,
        calculate_training_loss=True,
    )
    model.fit(matrix, show_progress=True)
    impression_df = pl.DataFrame(
        [pl.Series(name="impression_id", values=impression_id_list, dtype=pl.UInt32)]
        + [
            pl.Series(name=f"impression_factor_{i}", values=model.user_factors[:, i])
            for i in range(cfg.exp.n_factors)
        ]
    )
    article_df = pl.DataFrame(
        [pl.Series(name="article_id", values=article_id_list, dtype=pl.Int32)]
        + [
            pl.Series(name=f"article_factor_{i}", values=model.item_factors[:, i])
            for i in range(cfg.exp.n_factors)
        ]
    )
    return impression_df, article_df


def create_feature(cfg: DictConfig, output_path):
    input_dir = Path(cfg.dir.input_dir)
    size_name = cfg.exp.size_name
    data_dirs = get_data_dirs(input_dir, size_name)

    behaviors_df = pl.concat(
        [
            pl.read_parquet(data_dirs[data_name] / "behaviors.parquet").with_columns(
                pl.lit(data_name).alias("data_split"),
            )
            for data_name in data_dirs.keys()
        ],
        how="diagonal",
    )
    impression_df, article_df = process_df(cfg, behaviors_df)

    print(impression_df)
    impression_df.write_parquet(
        output_path / "impression_feat.parquet",
    )
    print(article_df)
    article_df.write_parquet(
        output_path / "article_feat.parquet",
    )


@hydra.main(version_base=None, config_path=".", config_name="config")
def main(cfg: DictConfig) -> None:
    runtime_choices = HydraConfig.get().runtime.choices
    exp_name = f"{Path(sys.argv[0]).parent.name}/{runtime_choices.exp}"

    print(f"exp_name: {exp_name}")
    output_path = Path(cfg.dir.features_dir) / exp_name
    print(f"ouput_path: {output_path}")
    os.makedirs(output_path, exist_ok=True)

    create_feature(cfg, output_path)


if __name__ == "__main__":
    main()
