import os
import sys
from pathlib import Path

import hydra
import numpy as np
import polars as pl
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig
from sklearn.feature_extraction.text import TfidfVectorizer

from utils.data import get_data_dirs

PREFIX = "c"

USE_COLUMNS = [
    "category_tfidf_sim",
    "category_tfidf_sim_rn",
]


def process_df(cfg, articles_df, history_df, candidate_df):
    # 集約してuserごとのcategory出現をテキスト化
    user_df = (
        history_df.select(["user_id", "article_id_fixed"])
        .explode("article_id_fixed")
        .join(
            articles_df.select(["article_id", "category"]),
            left_on="article_id_fixed",
            right_on="article_id",
            how="left",
        )
        .group_by("user_id")
        .agg(pl.col("category").cast(pl.String))
    )

    #
    article_text_list = articles_df["category"].cast(pl.String).to_list()
    user_text_list = user_df["category"].list.join(separator=" ").to_list()

    vectorizer = TfidfVectorizer()
    article_matrix = vectorizer.fit_transform(article_text_list)
    user_matrix = vectorizer.transform(user_text_list)

    article_id_map = dict(
        zip(
            articles_df["article_id"].to_list(),
            articles_df.with_row_index()["index"].to_list(),
        )
    )
    user_id_map = dict(
        zip(user_df["user_id"].to_list(), user_df.with_row_index()["index"].to_list())
    )

    candidate_df = candidate_df.with_columns(
        [
            pl.col("article_id").replace(article_id_map).alias("article_rn"),
            pl.col("user_id").replace(user_id_map).alias("user_rn"),
        ]
    )

    similarity = np.asarray(
        article_matrix[candidate_df["article_rn"].to_list()]
        .multiply(user_matrix[candidate_df["user_rn"].to_list()])
        .sum(axis=1)
    ).flatten()

    df = (
        candidate_df.with_columns(
            pl.Series(name="category_tfidf_sim", values=similarity)
        )
        .with_columns(
            pl.col("category_tfidf_sim")
            .rank(descending=True)
            .over("user_rn")
            .alias("category_tfidf_sim_rn")
        )
        .select(USE_COLUMNS)
    )

    return df


def create_feature(cfg: DictConfig, output_path):
    input_dir = Path(cfg.dir.input_dir)
    size_name = cfg.exp.size_name
    data_dirs = get_data_dirs(input_dir, size_name)

    articles_path = Path("input/ebnerd_testset/ebnerd_testset") / "articles.parquet"
    articles_df = pl.read_parquet(articles_path)

    for data_name in ["train", "validation", "test"]:
        print(f"processing {data_name} data")
        history_df = pl.read_parquet(data_dirs[data_name] / "history.parquet")
        candidate_df = pl.read_parquet(
            Path(cfg.dir.candidate_dir) / size_name / f"{data_name}_candidate.parquet"
        )

        df = process_df(cfg, articles_df, history_df, candidate_df)

        df = df.rename({col: f"{PREFIX}_{col}" for col in USE_COLUMNS})
        print(df)
        df.write_parquet(
            output_path / f"{data_name}_feat.parquet",
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
