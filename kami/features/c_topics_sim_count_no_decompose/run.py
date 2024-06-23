import os
import sys
from pathlib import Path

import hydra
import numpy as np
import polars as pl
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig
from sklearn.decomposition import NMF, LatentDirichletAllocation, TruncatedSVD
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.preprocessing import normalize
from tqdm.auto import tqdm

from utils.data import get_data_dirs

PREFIX = "c"

USE_COLUMNS = [
    "topics_count_svd_sim",
    "topics_count_svd_rn",
]


def process_df(cfg, article_embeddings, articles_df, history_df, candidate_df):
    # map
    article_id_map = dict(
        zip(
            articles_df["article_id"].to_list(),
            articles_df.with_row_index()["index"].to_list(),
        )
    )
    user_id_map = dict(
        zip(
            history_df["user_id"].to_list(),
            history_df.with_row_index()["index"].to_list(),
        )
    )
    map_history_df = (
        history_df.select(["user_id", "article_id_fixed"])
        .rename({"article_id_fixed": "article_id"})
        .with_columns(
            pl.col("article_id")
            .list.eval(pl.element().replace(article_id_map))
            .alias("new_article_id")
        )
    )

    # user embedding
    user_embeddings = []
    for new_article_id_list in tqdm(map_history_df["new_article_id"].to_list()):
        user_embeddings.append(article_embeddings[new_article_id_list].mean(axis=0))
    user_embeddings = np.array(user_embeddings)
    user_embeddings = normalize(user_embeddings, norm="l2")

    candidate_df = candidate_df.with_columns(
        [
            pl.col("article_id").replace(article_id_map).alias("article_rn"),
            pl.col("user_id").replace(user_id_map).alias("user_rn"),
        ]
    )

    # 要素積の合計を類似度とする（consin similarity）
    print(f"{article_embeddings.shape=}, {user_embeddings.shape=}")
    similarity = np.asarray(
        (
            article_embeddings[candidate_df["article_rn"].to_list()]
            * user_embeddings[candidate_df["user_rn"].to_list()]
        ).sum(axis=1)
    ).flatten()

    df = (
        candidate_df.with_columns(
            pl.Series(name="topics_count_svd_sim", values=similarity)
        )
        .with_columns(
            pl.col("topics_count_svd_sim")
            .rank(descending=True)
            .over("user_rn")
            .alias("topics_count_svd_rn")
        )
        .select(USE_COLUMNS)
    )

    return df


def create_feature(cfg: DictConfig, output_path):
    input_dir = Path(cfg.dir.input_dir)
    size_name = cfg.exp.size_name
    data_dirs = get_data_dirs(input_dir, size_name)

    articles_path = Path("input/ebnerd_testset/ebnerd_testset") / "articles.parquet"
    articles_df = pl.read_parquet(articles_path).with_columns(
        pl.concat_str(
            [
                pl.col("topics").list.join(separator=" "),
            ],
            separator=" ",
        ).alias("target_col"),
    )

    print("make article embeddings")
    vectorizer = CountVectorizer()
    article_matrix = vectorizer.fit_transform(articles_df["target_col"].to_list())
    article_embeddings = np.asarray(normalize(article_matrix, norm="l2").todense())

    for data_name in ["train", "validation", "test"]:
        print(f"processing {data_name} data")
        history_df = pl.read_parquet(data_dirs[data_name] / "history.parquet")
        candidate_df = pl.read_parquet(
            Path(cfg.dir.candidate_dir) / size_name / f"{data_name}_candidate.parquet"
        )

        df = process_df(cfg, article_embeddings, articles_df, history_df, candidate_df)

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
