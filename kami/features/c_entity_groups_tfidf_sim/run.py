import os
import sys
from pathlib import Path

import hydra
import numpy as np
import polars as pl
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig
from sklearn.feature_extraction.text import TfidfVectorizer
from tqdm.auto import tqdm

from utils.data import get_data_dirs

PREFIX = "c"

USE_COLUMNS = [
    "entity_groups_tfidf_sim",
    "entity_groups_tfidf_sim_rn",
]


def process_df(cfg, articles_df, history_df, candidate_df):
    articles_df = articles_df.select(["article_id", "entity_groups"]).with_columns(
        pl.col("entity_groups").list.join(separator=" ")
    )
    # 集約してuserごとのentity_groups出現をテキスト化
    print("make user_df")
    user_df = (
        history_df.select(["user_id", "article_id_fixed"])
        .explode("article_id_fixed")
        .join(
            articles_df,
            left_on="article_id_fixed",
            right_on="article_id",
            how="left",
        )
        .group_by("user_id")
        .agg(pl.col("entity_groups").cast(pl.String))
    )

    article_text_list = articles_df["entity_groups"].cast(pl.String).to_list()
    user_text_list = user_df["entity_groups"].list.join(separator=" ").to_list()

    print("TfidfVectorizer")
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

    print("similarity")

    batch_size = 10000
    article_rns = candidate_df["article_rn"].to_numpy()
    user_rns = candidate_df["user_rn"].to_numpy()
    similarity = []
    for i in tqdm(range(0, len(article_rns), batch_size)):
        similarity.append(
            np.asarray(
                article_matrix[article_rns[i : i + batch_size]]
                .multiply(user_matrix[user_rns[i : i + batch_size]])
                .sum(axis=1)
            ).flatten()
        )
    similarity = np.concatenate(similarity)

    print("make df")
    df = (
        candidate_df.with_columns(
            pl.Series(name="entity_groups_tfidf_sim", values=similarity)
        )
        .with_columns(
            pl.col("entity_groups_tfidf_sim")
            .rank(descending=True)
            .over("user_rn")
            .alias("entity_groups_tfidf_sim_rn")
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
