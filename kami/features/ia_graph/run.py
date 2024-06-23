import itertools
import os
import sys
from pathlib import Path

import hydra
import igraph as ig
import numpy as np
import polars as pl
from hydra.core.hydra_config import HydraConfig
from implicit.cpu.als import AlternatingLeastSquares
from omegaconf import DictConfig
from scipy.sparse import csr_matrix

import utils
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
    new_article_id_list = np.arange(len(article_id_list)) + len(impression_id_list)
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

    edges = [
        (row["new_impression_id"], row["new_article_id"])
        for row in count_df.iter_rows(named=True)
    ]
    weights = count_df["len"].to_numpy()

    graph = ig.Graph(edges=edges)

    # 計算量：https://igraph.org/c/doc/igraph-Structural.html

    # authority_score
    # https://python.igraph.org/en/stable/api/igraph.GraphBase.html#authority_score
    # Time complexity: depends on the input graph, usually it is O(|V|), the number of vertices.
    with utils.trace("authority_score"):
        authority_score = graph.authority_score(weights=weights)

    # hub_score
    # https://python.igraph.org/en/stable/api/igraph.GraphBase.html#hub_score
    # Time complexity: depends on the input graph, usually it is O(|V|), the number of vertices.
    with utils.trace("hub_score"):
        hub_score = graph.hub_score(weights=weights)

    # coreness
    # https://python.igraph.org/en/stable/api/igraph.GraphBase.html#coreness
    # Time complexity: O(|E|) where |E| is the number of edges in the graph.
    with utils.trace("coreness"):
        coreness = graph.coreness(mode="all")

    # eigenvector_centrality
    # https://python.igraph.org/en/stable/api/igraph.GraphBase.html#eigenvector_centrality
    # Time complexity: depends on the input graph, usually it is O(|V|+|E|).
    with utils.trace("eigenvector_centrality"):
        eigenvector_centrality = graph.eigenvector_centrality(
            directed=False, weights=weights
        )

    impression_df = pl.DataFrame(
        [
            pl.Series(name="impression_id", values=impression_id_list, dtype=pl.UInt32),
            pl.Series(
                name="i_authority_score",
                values=authority_score[: len(impression_id_list)],
            ),
            pl.Series(name="i_hub_score", values=hub_score[: len(impression_id_list)]),
            pl.Series(name="i_coreness", values=coreness[: len(impression_id_list)]),
            pl.Series(
                name="i_eigenvector_centrality",
                values=eigenvector_centrality[: len(impression_id_list)],
            ),
        ]
    )
    article_df = pl.DataFrame(
        [
            pl.Series(name="article_id", values=article_id_list, dtype=pl.Int32),
            pl.Series(
                name="a_authority_score",
                values=authority_score[len(impression_id_list) :],
            ),
            pl.Series(name="a_hub_score", values=hub_score[len(impression_id_list) :]),
            pl.Series(name="a_coreness", values=coreness[len(impression_id_list) :]),
            pl.Series(
                name="a_eigenvector_centrality",
                values=eigenvector_centrality[len(impression_id_list) :],
            ),
        ]
    )
    return impression_df, article_df


def create_feature(cfg: DictConfig, output_path):
    input_dir = Path(cfg.dir.input_dir)
    size_name = cfg.exp.size_name
    data_dirs = get_data_dirs(input_dir, size_name)

    for data_name in ["train", "validation", "test"]:
        print(f"processing {data_name} data")
        behaviors_df = pl.read_parquet(data_dirs[data_name] / "behaviors.parquet")
        impression_df, article_df = process_df(cfg, behaviors_df)

        print(impression_df)
        impression_df.write_parquet(
            output_path / f"{data_name}_impression_feat.parquet",
        )
        print(article_df)
        article_df.write_parquet(
            output_path / f"{data_name}_article_feat.parquet",
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
