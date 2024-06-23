import os
import sys
from pathlib import Path

import hydra
import polars as pl
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig

from utils.data import get_data_dirs

PREFIX = "i"

KEY_COLUMNS = [
    # user_id とセットじゃないと test でimpression_id=0 が重複しているので問題になる
    "impression_id",  # key
    "user_id",  # key
]

USE_COLUMNS = [
    "read_time",
    "scroll_percentage",
    "device_type",
    "is_sso_user",
    "gender",
    "postcode",
    "age",
    "is_subscriber",
    # processed
    "num_article_ids_inview",
]


def process_df(cfg, df):
    df = df.with_columns(
        pl.col("article_ids_inview").list.len().alias("num_article_ids_inview")
    )
    return df


def create_feature(cfg: DictConfig, output_path):
    input_dir = Path(cfg.dir.input_dir)
    size_name = cfg.exp.size_name
    data_dirs = get_data_dirs(input_dir, size_name)

    for data_name in ["train", "validation", "test"]:
        print(f"processing {data_name} data")
        behaviors_df = pl.read_parquet(data_dirs[data_name] / "behaviors.parquet")
        df = process_df(cfg, behaviors_df).select(KEY_COLUMNS + USE_COLUMNS)

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
