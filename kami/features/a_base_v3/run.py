import os
import sys
from pathlib import Path

import hydra
import polars as pl
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig
from sklearn.preprocessing import OrdinalEncoder

PREFIX = "a"

KEY_COLUMNS = [
    "article_id",
]

USE_COLUMNS = [
    "premium",
    "category_article_type",
    "total_inviews",
    "total_pageviews",
    "total_read_time",
    "sentiment_score",
    "ordinal_sentiment_label",
    "title_len",
    "title_word_count",
    "subtitle_len",
    "subtitle_word_count",
]


def process_df(cfg, df):
    enc = OrdinalEncoder()
    print(df[["article_type"]].to_numpy().shape)
    enc.fit(df[["article_type"]].to_numpy())
    article_types = enc.transform(df[["article_type"]]).flatten()

    sentiment_labels = [["Negative", "Neutral", "Positive"]]
    enc = OrdinalEncoder(categories=sentiment_labels)
    enc.fit(df[["sentiment_label"]].to_numpy())
    sentiment_labels = enc.transform(df[["sentiment_label"]]).flatten()

    df = df.with_columns(
        [
            pl.Series(
                name="category_article_type", values=article_types, dtype=pl.Int32
            ),
            pl.Series(
                name="ordinal_sentiment_label", values=sentiment_labels, dtype=pl.Int32
            ),
            pl.Series(
                name="title_len", values=df["title"].str.len_chars(), dtype=pl.Int32
            ),
            pl.Series(
                name="title_word_count",
                values=df["title"].str.split(" ").list.len(),
                dtype=pl.Int32,
            ),
            pl.Series(
                name="subtitle_len",
                values=df["subtitle"].str.len_chars(),
                dtype=pl.Int32,
            ),
            pl.Series(
                name="subtitle_word_count",
                values=df["subtitle"].str.split(" ").list.len(),
                dtype=pl.Int32,
            ),
        ]
    )
    return df


def create_feature(cfg: DictConfig, output_path):
    articles_path = Path("input/ebnerd_testset/ebnerd_testset") / "articles.parquet"
    articles_df = pl.read_parquet(articles_path)
    df = process_df(cfg, articles_df).select(KEY_COLUMNS + USE_COLUMNS)
    df = df.rename({col: f"{PREFIX}_{col}" for col in USE_COLUMNS})
    print(df)

    for data_name in ["train", "validation", "test"]:
        print(f"processing {data_name} data")
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
