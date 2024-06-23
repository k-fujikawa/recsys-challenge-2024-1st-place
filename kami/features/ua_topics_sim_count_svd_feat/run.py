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

n_components = 10


def process_df(cfg, article_embeddings, articles_df, history_df):
    # user
    user_id_list = history_df["user_id"].unique(maintain_order=True).to_list()

    # article
    article_id_list = articles_df["article_id"].unique(maintain_order=True).to_list()
    new_article_id_list = np.arange(len(article_id_list))
    article_id_map = dict(zip(article_id_list, new_article_id_list))

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
    user_embeddings = normalize(np.array(user_embeddings), norm="l2")

    user_df = pl.DataFrame(
        [
            pl.Series(
                name="user_id", values=user_id_list, dtype=history_df["user_id"].dtype
            )
        ]
        + [
            pl.Series(name=f"topic_user_emb_{i}", values=user_embeddings[:, i])
            for i in range(n_components)
        ]
    )
    article_embedding_df = pl.DataFrame(
        [
            pl.Series(
                name="article_id",
                values=article_id_list,
                dtype=articles_df["article_id"].dtype,
            )
        ]
        + [
            pl.Series(name=f"topic_article_emb_{i}", values=article_embeddings[:, i])
            for i in range(n_components)
        ]
    )
    return user_df, article_embedding_df


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
    decomposer = TruncatedSVD(n_components=n_components)
    article_embeddings = normalize(decomposer.fit_transform(article_matrix), norm="l2")

    for data_name in ["train", "validation", "test"]:
        print(f"processing {data_name} data")
        history_df = pl.read_parquet(data_dirs[data_name] / "history.parquet")

        user_df, article_embedding_df = process_df(
            cfg, article_embeddings, articles_df, history_df
        )

        print(user_df)
        user_df.write_parquet(
            output_path / "user_feat.parquet",
        )
        print(article_embedding_df)
        article_embedding_df.write_parquet(
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
