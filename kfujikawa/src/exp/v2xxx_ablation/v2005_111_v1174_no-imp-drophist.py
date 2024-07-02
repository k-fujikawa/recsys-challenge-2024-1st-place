#! /usr/bin/env python

from __future__ import annotations

import functools
import inspect
import math
import os
import shutil
import sys
import typing
import yaml
from collections import Counter
from dataclasses import asdict, dataclass, field, fields
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Callable

import lightning.pytorch as L
import numpy as np
import polars as pl
import pytorch_toolbelt.losses
import scipy
import torch
import torch.nn.functional as F
import torch.utils.data
import torchmetrics
import typer
from hydra_slayer import Registry
from loguru import logger
from omegaconf import OmegaConf, MISSING, SI
from torch.utils.data.dataloader import default_collate
from tqdm.auto import tqdm

from exputils.const import KFUJIKAWA_DIR
from exputils.utils import timer
from exputils.torch import CollateSequences, pad_sequences
from exputils.lightning import TQDMProgressBarEx, SlackNotificationCallback


APP = typer.Typer(pretty_exceptions_enable=False)
FILE_DIR = Path(__file__).parent.name
FILE_NAME = Path(__file__).stem
REGISTRY = Registry()
PREPROCESSED_DIR = KFUJIKAWA_DIR / "v0xxx_preprocess"
PROJECT_NAME = "recsys-challenge-2024"
MAX_HISTORY_LENGTH = 10
SPLIT_INTERVAL = 60 * 60 * 24 * 7
USE_FUTURE_IMP = FILE_NAME.split("_")[1][0] == "1"
USE_FUTURE_ARTICLE_STATS = FILE_NAME.split("_")[1][1] == "1"
USE_PAST_IMP = FILE_NAME.split("_")[1][2] == "1"
USE_PSEUDO_LABEL = FILE_NAME.split("_")[2] == "PL"
PSEUDO_LABEL_DIR = KFUJIKAWA_DIR / "v8xxx_ensemble" / "v8003_015_v1157_avg"
OmegaConf.register_new_resolver("eval", eval, replace=True)
os.environ["TOKENIZERS_PARALLELISM"] = "false"
sys.dont_write_bytecode = True
torch.set_float32_matmul_precision("high")

FOLD_SPLIT_MAPPING = {
    0: {
        "train": ["train", "pl_validation"] if USE_PSEUDO_LABEL else ["train"],
        "validation": ["validation"],
        "test": ["test"],
    },
    1: {
        "train": ["validation", "pl_test"] if USE_PSEUDO_LABEL else ["validation"],
        "validation": ["validation"],
        "test": ["test"],
    },
    2: {
        "train": ["train", "validation", "pl_test"]
        if USE_PSEUDO_LABEL
        else ["train", "validation"],
        "validation": ["validation"],
        "test": ["test"],
    },
}

# =============================================================================
#                                  CONFIG
# =============================================================================


@dataclass
class CommonConfig:
    suffix: str = ""
    exp_id: str = SI(FILE_NAME + "${.suffix}")
    output_root_dir: Path = KFUJIKAWA_DIR / FILE_DIR
    output_dir: Path = SI("${.output_root_dir}/${.exp_id}")
    fold_output_dir: Path = SI("${.output_dir}/fold_${common.fold}")
    seed: int = 0
    fold: int = 0
    batch_size: int = 64
    max_epochs: int = 5
    num_epochs: int = 4
    ckpt_epochs: int = 3
    max_samples: int | None = None
    num_workers: int = min(
        8, (os.cpu_count() or 1) // max(1, torch.cuda.device_count())
    )
    progress_update_interval: int = 1
    validation_interval: int = 2
    overwrite: bool = False
    debug: bool = False
    verbose: bool = True
    resume: str | None = None
    wandb: bool = False
    slack: bool = SI("${.wandb}")
    use_future_imp: bool = USE_FUTURE_IMP
    use_future_article_stats: bool = USE_FUTURE_ARTICLE_STATS
    use_past_imp: bool = USE_PAST_IMP
    use_pseudo_label: bool = USE_PSEUDO_LABEL
    num_features: int = MISSING


@dataclass
class DatasetConfig:
    _target_: str = "Dataset"
    _var_: str | None = None
    fold: int = SI("${common.fold}")
    split: str = MISSING
    small: bool = False
    debug: bool = SI("${common.debug}")


@dataclass
class DataLoadersConfig:
    @dataclass
    class DataLoaderConfig:
        @dataclass
        class BatchSamplerConfig:
            _target_: str = "BatchSampler"
            dataset: dict = MISSING
            batch_size: int = SI("${common.batch_size}")
            num_workers: int = SI("${common.num_workers}")
            max_samples: int | None = SI("${common.max_samples}")
            max_sample_per_user: int = 10**8
            drop_last: bool = False
            shuffle: bool = False

        _target_: str = "torch.utils.data.DataLoader"
        dataset: DatasetConfig = MISSING
        batch_size: int = 1
        num_workers: int = SI("${common.num_workers}")
        batch_sampler: BatchSamplerConfig | None = None
        collate_fn: dict = MISSING

    train: DataLoaderConfig = DataLoaderConfig(
        dataset=DatasetConfig(
            _var_="train_dataset",
            split="train",
        ),
        batch_sampler=DataLoaderConfig.BatchSamplerConfig(
            dataset={"_var_": "train_dataset"},
            max_sample_per_user=3,
            shuffle=True,
            drop_last=True,
        ),
        collate_fn={"_target_": "DataCollator"},
    )
    validation_small: DataLoaderConfig = DataLoaderConfig(
        dataset=DatasetConfig(
            _var_="valid_small_dataset",
            split="validation",
            small=True,
        ),
        batch_sampler=DataLoaderConfig.BatchSamplerConfig(
            dataset={"_var_": "valid_small_dataset"},
        ),
        collate_fn={"_target_": "DataCollator"},
    )
    validation: DataLoaderConfig = DataLoaderConfig(
        dataset=DatasetConfig(
            _var_="valid_dataset",
            split="validation",
            small=False,
        ),
        batch_sampler=DataLoaderConfig.BatchSamplerConfig(
            dataset={"_var_": "valid_dataset"},
        ),
        collate_fn={"_target_": "DataCollator"},
    )
    test: DataLoaderConfig = DataLoaderConfig(
        dataset=DatasetConfig(
            _var_="test_dataset",
            split="test",
        ),
        batch_sampler=DataLoaderConfig.BatchSamplerConfig(
            dataset={"_var_": "test_dataset"},
        ),
        collate_fn={"_target_": "DataCollator"},
    )


@dataclass
class ModelConfig:
    @dataclass
    class EmbeddingConfig:
        _target_: str = "Embedding"
        _mode_: str = "call"
        sinusoidal_dim: int = 128
        hidden_dim: int = 128
        output_dim: int = 128

    @dataclass
    class EncoderConfig:
        @dataclass
        class TransformerEncoderLayerConfig:
            @dataclass
            class Activation:
                _target_: str = "torch.nn.GELU"

            _target_: str = "TransformerEncoderLayer"
            d_model: int = SI("${model.embedding.output_dim}")
            nhead: int = SI("${eval: ${.d_model} // 32}")
            dim_feedforward: int = SI("${eval: ${.d_model} * 2}")
            dropout: float = 0.1
            activation: Activation = Activation()
            b2t_connection: bool = False
            batch_first: bool = True
            norm_first: bool = True

        _target_: str = "torch.nn.TransformerEncoder"
        encoder_layer: TransformerEncoderLayerConfig = TransformerEncoderLayerConfig()
        num_layers: int = 8

    @dataclass
    class PredictionHeadConfig:
        _target_: str = "PredictionHead"
        hidden_dim: int = SI("${model.embedding.output_dim}")
        num_hidden_layers: int = 1
        output_dim: int = 1

    @dataclass
    class MetricsMeterConfig:
        _target_: str = "MetricsMeter"
        loss_weights: dict = field(
            default_factory=lambda: {
                "bce_loss": 1.0,
            },
        )

    @dataclass
    class OptimizerConfig:
        _target_: str = "torch.optim.AdamW"
        _mode_: str = "partial"
        lr: float = 1e-3
        weight_decay: float = 1e-5
        eps: float = 1e-6

    @dataclass
    class SchedulerConfig:
        _target_: str = "torch.optim.lr_scheduler.CosineAnnealingLR"
        _mode_: str = "partial"
        T_max: int = SI("${common.max_epochs}")
        eta_min: float = 1e-6

    _target_: str = "Model"
    embedding: EmbeddingConfig = EmbeddingConfig()
    encoder: EncoderConfig = EncoderConfig()
    prediction_head: PredictionHeadConfig = PredictionHeadConfig()
    metrics_meter: MetricsMeterConfig = MetricsMeterConfig()
    optimizer_factory: OptimizerConfig = OptimizerConfig()
    scheduler_factory: SchedulerConfig = SchedulerConfig()


@dataclass
class TrainerConfig:
    _target_: str = "lightning.Trainer"
    accelerator: str = "auto"
    devices: list[int] = field(default_factory=lambda: [0])
    accumulate_grad_batches: int = 1
    deterministic: bool = False
    benchmark: bool = False
    precision: str = "16-mixed"
    max_epochs: int = SI("${common.num_epochs}")
    gradient_clip_val: float = 1.0


@dataclass
class ExperimentConfig:
    common: CommonConfig = CommonConfig()
    model: ModelConfig = ModelConfig()
    dataloaders: DataLoadersConfig = DataLoadersConfig()
    trainer: TrainerConfig = TrainerConfig()

    @classmethod
    def create(
        cls,
        override_dotlists: list[str] | None = None,
    ) -> ExperimentConfig:
        dotlist_config = OmegaConf.from_dotlist(list(override_dotlists or []))
        config = OmegaConf.merge(OmegaConf.structured(cls()), dotlist_config)
        config.common.num_features = len(
            FeatureExtractionPipeline.get_fields(InputFeatureField)
        )

        if config.common.debug:
            config.common.max_epochs = 2
            config.common.num_epochs = 2
            config.common.exp_id = "debug"
            config.common.overwrite = True
            config.common.num_workers = 0
            config = OmegaConf.merge(config, dotlist_config)

        return typing.cast(ExperimentConfig, OmegaConf.to_object(config))


# =============================================================================
#                                  FEATURE
# =============================================================================


@dataclass
class ArticleFeature:
    published_ts: np.ndarray
    total_inviews: np.ndarray
    total_pageviews: np.ndarray
    total_read_time: np.ndarray
    sentiment_score: np.ndarray
    ner_clusters_ids: np.ndarray
    entity_groups_ids: np.ndarray
    topics_ids: np.ndarray
    image_ids_ids: np.ndarray
    category_ids: np.ndarray
    subcategory_ids: np.ndarray
    inview_elapsed_mins: np.ndarray
    inview_counts: np.ndarray
    read_time_sum: np.ndarray
    scroll_percentage_sum: np.ndarray
    scroll_zero_counts: np.ndarray


def create_article_feature(
    fold: int,
    split: str,
    debug: bool,
) -> dict[str, ArticleFeature]:
    outputs = {}

    for _split in FOLD_SPLIT_MAPPING[fold][split]:
        _split = _split.replace("pl_", "")
        with timer(f"Load article data: ({_split})"):
            df = pl.concat(
                [
                    pl.scan_parquet(
                        PREPROCESSED_DIR / "v0100_articles" / "dataset.parquet",
                    ).select(
                        pl.col("article_index"),
                        pl.col("published_ts"),
                        pl.col("total_inviews"),
                        pl.col("total_pageviews"),
                        pl.col("total_read_time"),
                        pl.col("sentiment_score"),
                    ),
                    pl.scan_parquet(
                        PREPROCESSED_DIR
                        / "v0101_article_inviews_in_split_v2"
                        / _split
                        / "dataset.parquet",
                    ).select(
                        "inview_elapsed_mins",
                        "inview_counts",
                        "read_time_sum",
                        "scroll_percentage_sum",
                        "scroll_zero_counts",
                    ),
                    pl.scan_parquet(
                        PREPROCESSED_DIR
                        / "v0102_article_metadata_id_v2"
                        / "dataset.parquet"
                    ).select(
                        pl.col("ner_clusters_ids"),
                        pl.col("entity_groups_ids"),
                        pl.col("topics_ids"),
                        pl.col("image_ids_ids"),
                        pl.col("category_ids"),
                        pl.col("subcategory_ids"),
                    ),
                ],
                how="horizontal",
            )
            df = df.sort("article_index")
            df = df.collect(streaming=True)

        assert (df["article_index"].to_numpy() == np.arange(len(df))).all()
        outputs[_split] = ArticleFeature(
            **{
                col.name: df[col.name].to_numpy()
                for col in fields(ArticleFeature)
                if col.name in df.columns
            },
        )
    return outputs


@dataclass
class UserFeature:
    article_indices: np.ndarray
    impression_ts: np.ndarray
    scroll_percentage: np.ndarray
    read_time: np.ndarray
    impression_ids_in_split: np.ndarray
    user_inviews_in_split: np.ndarray
    impression_ts_in_split: np.ndarray


def create_user_feature(
    fold: int,
    split: str,
    debug: bool,
) -> dict[str, UserFeature]:
    outputs = {}
    n_rows = 10000 if debug else None
    for _split in FOLD_SPLIT_MAPPING[fold][split]:
        _split = _split.replace("pl_", "")
        with timer(f"Load user data ({_split})"):
            df = pl.concat(
                [
                    pl.scan_parquet(
                        PREPROCESSED_DIR / "v0200_users" / _split / "dataset.parquet",
                        n_rows=n_rows,
                    ).select(
                        pl.col("user_index"),
                        pl.col("article_indices"),
                        pl.col("impression_ts"),
                        pl.col("scroll_percentage"),
                        pl.col("read_time"),
                    ),
                    pl.scan_parquet(
                        PREPROCESSED_DIR
                        / "v0201_user_inviews_in_split"
                        / _split
                        / "dataset.parquet",
                        n_rows=n_rows,
                    ).select(
                        pl.col("impression_ids_in_split"),
                        pl.col("user_inviews_in_split"),
                        pl.col("impression_ts_in_split"),
                    ),
                ],
                how="horizontal",
            )
            df = df.sort("user_index")
            df = df.collect(streaming=True)

        assert (df["user_index"].to_numpy() == np.arange(len(df))).all()
        outputs[_split] = UserFeature(
            **{col.name: df[col.name].to_numpy() for col in fields(UserFeature)}
        )
    return outputs


@dataclass
class ImpressionFeature:
    is_pseudo_label: bool
    impression_index: np.ndarray
    impression_id: np.ndarray
    user_index: np.ndarray
    article_indices_inview: np.ndarray
    impression_ts: np.ndarray
    num_impressions: np.ndarray
    is_subscriber: np.ndarray
    device_type: np.ndarray
    read_time: np.ndarray
    scroll_percentage: np.ndarray
    labels: np.ndarray
    label_confidence: np.ndarray
    ts_min: int


def create_impression_feature(
    fold: int,
    split: str,
    small: bool,
    debug: bool,
) -> dict[str, ImpressionFeature]:
    outputs = {}
    n_rows = 1_000_000 if debug else None
    for _split in FOLD_SPLIT_MAPPING[fold][split]:
        is_pseudo_label = _split.startswith("pl_")
        _split = _split.replace("pl_", "")
        with timer(f"Load impression data ({_split})"):
            df = pl.concat(
                [
                    pl.scan_parquet(
                        PREPROCESSED_DIR
                        / "v0300_impressions"
                        / _split
                        / "dataset.parquet",
                        n_rows=n_rows,
                    ).select(
                        pl.col("impression_id"),
                        pl.col("user_index"),
                        pl.col("article_indices_inview"),
                        pl.col("impression_ts"),
                        pl.col("is_subscriber"),
                        pl.col("device_type"),
                        pl.col("read_time"),
                        pl.col("scroll_percentage"),
                        pl.col("labels"),
                        pl.lit(1.0).cast(pl.Float32).alias("label_confidence"),
                        pl.col("in_small"),
                    ),
                    pl.scan_parquet(
                        PREPROCESSED_DIR
                        / "v0301_imp_counts_per_user"
                        / _split
                        / "dataset.parquet",
                        n_rows=n_rows,
                    ).select(
                        pl.col("num_impressions"),
                    ),
                ],
                how="horizontal",
            )
            impression_ts_min = int(
                df.select("impression_ts").min().collect().to_numpy().squeeze()
            )
            if is_pseudo_label:
                _df = (
                    pl.scan_parquet(
                        PSEUDO_LABEL_DIR / f"{_split}.parquet",
                        n_rows=n_rows,
                    )
                    .select(
                        pl.col("pred").alias("labels"),
                        pl.col("pred_diff_top2").alias("label_confidence"),
                    )
                    .collect()
                )
                df = df.with_columns(_df)

            df = (
                df.filter(pl.col("in_small") if small and not debug else pl.lit(True))
                .filter(pl.col("user_index") < 10000 if debug else pl.lit(True))
                .with_row_index("impression_index")
            )
            df = df.collect(streaming=True)

        if not small and not debug:
            assert (df["impression_index"].to_numpy() == np.arange(len(df))).all()
        outputs[_split] = ImpressionFeature(
            is_pseudo_label=is_pseudo_label,
            ts_min=impression_ts_min,
            **{
                col.name: df[col.name].to_numpy()
                for col in fields(ImpressionFeature)
                if col.name in df.columns
            },
        )
    return outputs


# =============================================================================
#                             FEATURE FIELDS
# =============================================================================


class ArticleMetadataCounts(Enum):
    topics = 61
    category = 12
    subcategory = 34
    entity_groups = 7
    image_ids = 2
    ner_clusters = 147


@dataclass
class FeatureField:
    name: str
    collate_fn: Callable | None = default_collate
    max_value: int = -1
    padding_idx: int = -1
    dtype: type | None = None


@dataclass
class MetadataField(FeatureField):
    collate_fn: Callable = default_collate


@dataclass
class LabelField(FeatureField):
    collate_fn: Callable = default_collate
    dtype: type = np.float32


@dataclass
class InputFeatureField(FeatureField):
    pass


@dataclass
class SinusoidalImpressionFeatureField(InputFeatureField):
    collate_fn: Callable = default_collate
    dtype: type = np.float32
    padding_idx: int = 0


@dataclass
class NumericalImpressionFeatureField(InputFeatureField):
    collate_fn: Callable = default_collate
    dtype: type = np.float32


@dataclass
class CategoricalImpressionFeatureField(InputFeatureField):
    collate_fn: Callable = default_collate
    dtype: type = np.int64


@dataclass
class SinusoidalArticleFeatureField(InputFeatureField):
    collate_fn: Callable = CollateSequences(pad_value=0)
    dtype: type = np.float32
    padding_idx: int = 0


@dataclass
class NumericalArticleFeatureField(InputFeatureField):
    collate_fn: Callable = CollateSequences(pad_value=0)
    dtype: type = np.float32


@dataclass
class CategoricalArticleFeatureField(InputFeatureField):
    collate_fn: Callable = CollateSequences(pad_value=0)
    dtype: type = np.int64


@dataclass
class TextTokenArticleFeatureField(InputFeatureField):
    collate_fn: Callable = CollateSequences(pad_value=0)
    dtype: type = np.int64


@dataclass
class CategoricalArticleSimilarityFeatureField(InputFeatureField):
    collate_fn: Callable = CollateSequences(pad_value=0)
    dtype: type = np.int64


@dataclass
class CategoricalArticleHistorySimilarityFeatureField(InputFeatureField):
    pair_col: str = ""
    collate_fn: Callable = CollateSequences(pad_value=0)
    dtype: type = np.int64


@dataclass
class PreTrainedArticleHistorySimilarityFeatureField(InputFeatureField):
    name: str | None = None
    collate_fn: Callable | None = None
    df_path: Path = Path("")
    embedding_col: str = ""
    use_proj: bool = False


# =============================================================================
#                           FEATURE EXTRACTION
# =============================================================================


def compute_unique_features(
    x: np.ndarray,
    **kwargs,
):
    _, _idx, _inv, _counts = np.unique(
        x,
        return_index=True,
        return_inverse=True,
        return_counts=True,
    )
    _sidx = np.sort(_idx)
    return {
        "values": x[_sidx],
        "counts": _counts[_inv][_sidx],
        "idx": _idx,
        "sidx": _sidx,
        "inv": _inv,
        **{k: v[_sidx] for k, v in kwargs.items()},
    }


def compute_unique_inv_features(
    x: np.ndarray,
    y: np.ndarray,
    maxlen: int | None = None,
    fill_value: float = 0,
    **kwargs,
):
    xy = np.concatenate([x, y])
    _, _idx, _inv = np.unique(
        xy,
        return_index=True,
        return_inverse=True,
    )
    _sidx = np.sort(_idx)
    outputs = {"values": xy[_sidx][:maxlen]}
    for k, v in kwargs.items():
        assert y.shape == v.shape
        outputs[k] = np.full(len(_inv), fill_value, dtype=v.dtype)
        outputs[k][_idx[_inv][len(x) :]] = v
        outputs[k] = outputs[k][_sidx][:maxlen]
    return outputs


def compute_rank(
    x: np.ndarray,
    method: str = "average",
    descending: bool = False,
    normalize: bool = False,
    log_scale: bool = True,
):
    factor = -1 if descending else 1
    data = scipy.stats.rankdata(x * factor, method=method).astype(np.float16)

    if log_scale:
        data = np.log1p(data)

    if normalize:
        data = data - data.min()
        data = data / max(1, data.max())
        data = data + 1
    return data


def compute_matched_article_feature_counts(
    feature_ids: np.ndarray,
    candidate_article_indices: np.ndarray,
    reference_article_indices: np.ndarray,
    max_feature_id: int,
):
    if len(reference_article_indices) == 0:
        return np.zeros(len(candidate_article_indices), dtype=np.float32)
    ids = np.concatenate(feature_ids[reference_article_indices])
    ids = ids[ids > 0]
    counts = np.bincount(ids, minlength=max_feature_id + 1)
    counts = counts / max(1, counts.max())
    return np.array([counts[xs].max() for xs in feature_ids[candidate_article_indices]])


def compute_ts_agg_features(
    min_ts: int,
    max_ts: int,
    src_ts: np.ndarray,
    ref_ts: np.ndarray,
    agg: Callable,
):
    return np.array(
        [agg(r[(min_ts <= s) & (s < max_ts)]) for s, r in zip(src_ts, ref_ts)],
        dtype=np.float32,
    )


class FeatureExtractor:
    is_available: bool

    @property
    def fields(self) -> list[FeatureField]:
        raise NotImplementedError()

    def __call__(self, features: dict) -> dict:
        raise NotImplementedError()


# -----------------------------------------------------------------------------
#                         IMPRESSION-LEVEL FEATURES
# -----------------------------------------------------------------------------


class MetaFieldFeatureExtractor(FeatureExtractor):
    is_available = True

    @property
    def fields(self) -> list[FeatureField]:
        return [
            MetadataField("impression_id"),
            MetadataField("user_index"),
        ]

    def __call__(
        self,
        impressions: ImpressionFeature,
        impression_index: int,
    ) -> dict:
        outputs = {}
        outputs["impression_id"] = int(impressions.impression_id[impression_index])
        outputs["user_index"] = impressions.user_index[impression_index]
        return outputs


class LabelFeatureExtractor(FeatureExtractor):
    is_available = True

    @property
    def fields(self) -> list[FeatureField]:
        return [
            LabelField("labels", collate_fn=CollateSequences(pad_value=-1)),
            LabelField("is_pseudo_label"),
            LabelField("label_confidence"),
        ]

    def __call__(
        self,
        impressions: ImpressionFeature,
        impression_index: int,
    ) -> dict:
        outputs = {}
        outputs["labels"] = impressions.labels[impression_index]
        outputs["is_pseudo_label"] = np.bool_(impressions.is_pseudo_label)
        outputs["label_confidence"] = impressions.label_confidence[impression_index]
        return outputs


class ImpressionFeatureExtractor(FeatureExtractor):
    is_available = True

    @property
    def fields(self) -> list[FeatureField]:
        return [
            SinusoidalImpressionFeatureField("elapsed_ts_from_day_start"),
            SinusoidalImpressionFeatureField("read_time"),
            SinusoidalImpressionFeatureField("scroll_percentage"),
            CategoricalImpressionFeatureField("device_type", max_value=4),
        ]

    def __call__(
        self,
        impressions: ImpressionFeature,
        impression_index: int,
    ) -> dict:
        outputs = {}
        outputs["imp_ts_inview"] = impressions.impression_ts[impression_index]
        outputs["elapsed_ts_from_day_start"] = (
            outputs["imp_ts_inview"] % (60**2) / (60**2)
        )
        outputs["read_time"] = impressions.read_time[impression_index]
        outputs["device_type"] = impressions.device_type[impression_index] + 1
        outputs["scroll_percentage"] = impressions.scroll_percentage[impression_index]
        outputs["inview_elapsed_min"] = (
            (outputs["imp_ts_inview"] - impressions.ts_min) % SPLIT_INTERVAL // 60
        )
        return outputs


class HistoryImpressionFeatureExtractor(FeatureExtractor):
    is_available = True

    @property
    def fields(self) -> list[FeatureField]:
        return [
            SinusoidalImpressionFeatureField("elapsed_ts_from_history"),
            SinusoidalImpressionFeatureField("num_history_articles"),
        ]

    def __call__(
        self,
        users: UserFeature,
        user_index: int,
        imp_ts_inview: int,
        split: str,
    ) -> dict:
        outputs = {}

        history_mask = np.full(len(users.article_indices[user_index]), True)
        # drop_prob = 0.3
        # if split == "train" and np.random.rand() < drop_prob:
        #     history_mask = np.random.choice(
        #         [True, False],
        #         size=len(history_mask),
        #         p=[1 - drop_prob, drop_prob],
        #     )
        #     history_mask[-1] = True

        outputs["history_mask"] = history_mask
        outputs["num_history_articles"] = np.log1p(
            len(users.impression_ts[user_index][history_mask])
        )

        if outputs["num_history_articles"] == 0:
            outputs["elapsed_ts_from_history"] = 0
            outputs["history_read_time_mean"] = 0
            outputs["history_scroll_percentage_mean"] = 0
            outputs["history_scroll_zero_ratio"] = 0
            return outputs

        outputs["elapsed_ts_from_history"] = np.clip(
            np.log1p(
                (imp_ts_inview - users.impression_ts[user_index][history_mask].max())
                / 60**2
            ),
            0,
            5,
        )
        outputs["history_read_time_mean"] = users.read_time[user_index][
            history_mask
        ].mean()
        outputs["history_scroll_percentage_mean"] = users.scroll_percentage[user_index][
            history_mask
        ].mean()
        outputs["history_scroll_zero_ratio"] = (
            users.scroll_percentage[user_index][history_mask] == 0
        ).mean()

        return outputs


class PastImpressionFeatureExtractor(FeatureExtractor):
    is_available = USE_PAST_IMP

    @property
    def fields(self) -> list[FeatureField]:
        return [
            SinusoidalImpressionFeatureField("elapsed_ts_from_past"),
            SinusoidalImpressionFeatureField("num_past_articles"),
        ]

    def __call__(
        self,
        users: UserFeature,
        user_index: int,
        imp_ts_inview: int,
        impression_id: int,
    ) -> dict:
        outputs = {}
        impression_ids_in_split = users.impression_ids_in_split[user_index]
        impression_ts_in_split = users.impression_ts_in_split[user_index]
        article_indices_in_split = users.user_inviews_in_split[user_index]
        is_current_impression = impression_ids_in_split == impression_id
        past_indices = (
            (is_current_impression[::-1].cumsum() == 1)[::-1]
            & ~is_current_impression
            & (impression_ids_in_split != 0)
        )
        assert is_current_impression.sum() == 1
        outputs["num_past_articles"] = np.log1p(past_indices.sum())

        if outputs["num_past_articles"] == 0:
            outputs["past_article_indices"] = np.array([], dtype=np.int32)
            outputs["past_article_ts"] = np.array([], dtype=np.int32)
            outputs["elapsed_ts_from_past"] = np.float32(0)
            return outputs

        outputs["past_article_indices"] = article_indices_in_split[past_indices][::-1]
        outputs["past_article_ts"] = impression_ts_in_split[past_indices][::-1]
        outputs["elapsed_ts_from_past"] = np.clip(
            np.log1p(
                (imp_ts_inview - impression_ts_in_split[past_indices].max()) / 60**2
            ),
            0,
            3,
        )
        return outputs


class FutureImpressionFeatureExtractor(FeatureExtractor):
    is_available = USE_FUTURE_IMP

    @property
    def fields(self) -> list[FeatureField]:
        return [
            SinusoidalImpressionFeatureField("elapsed_ts_from_future"),
            SinusoidalImpressionFeatureField("num_future_articles"),
        ]

    def __call__(
        self,
        users: UserFeature,
        user_index: int,
        imp_ts_inview: int,
        impression_id: int,
    ) -> dict:
        outputs = {}
        impression_ids_in_split = users.impression_ids_in_split[user_index]
        impression_ts_in_split = users.impression_ts_in_split[user_index]
        article_indices_in_split = users.user_inviews_in_split[user_index]
        is_current_impression = impression_ids_in_split == impression_id
        future_indices = (
            (is_current_impression.cumsum() == 1)
            & ~is_current_impression
            & (impression_ids_in_split != 0)
        )
        assert is_current_impression.sum() == 1

        outputs["num_future_articles"] = np.log1p(future_indices.sum())
        outputs["future_past_article_indices"] = article_indices_in_split
        if outputs["num_future_articles"] == 0:
            outputs["elapsed_ts_from_future"] = np.float32(0)
            outputs["future_article_indices"] = np.array([], dtype=np.int32)
            outputs["future_article_ts"] = np.array([], dtype=np.int32)
            return outputs

        outputs["future_article_indices"] = article_indices_in_split[future_indices]
        outputs["future_article_ts"] = impression_ts_in_split[future_indices]
        outputs["elapsed_ts_from_future"] = np.clip(
            np.log1p(
                (impression_ts_in_split[future_indices].min() - imp_ts_inview) / 60**2
            ),
            0,
            3,
        )

        return outputs


# -----------------------------------------------------------------------------
#                          ARTICLE-LEVEL FEATURES
# -----------------------------------------------------------------------------


class ArticleFeatureExtractor(FeatureExtractor):
    is_available = True

    @property
    def fields(self) -> list[FeatureField]:
        return [
            FeatureField(
                "inview_article_indices",
                collate_fn=CollateSequences(pad_value=-1),
            ),
            FeatureField(
                "is_inview",
                collate_fn=CollateSequences(pad_value=0),
            ),
            SinusoidalArticleFeatureField(
                "article_sentiment_score",
            ),
            SinusoidalArticleFeatureField(
                "article_published_ts_diff",
            ),
            SinusoidalArticleFeatureField(
                "inview_article_ts_ranks_desc",
            ),
            CategoricalArticleSimilarityFeatureField(
                "article_topics_ids",
                max_value=ArticleMetadataCounts.topics.value,
            ),
            CategoricalArticleSimilarityFeatureField(
                "article_category_ids",
                max_value=ArticleMetadataCounts.category.value,
            ),
        ]

    def __call__(
        self,
        impressions: ImpressionFeature,
        impression_index: int,
        articles: ArticleFeature,
        imp_ts_inview: int,
    ) -> dict:
        outputs = {}
        outputs["inview_article_indices"] = impressions.article_indices_inview[
            impression_index
        ]
        outputs["num_inview"] = len(outputs["inview_article_indices"])
        outputs["is_inview"] = np.ones(outputs["num_inview"], dtype=np.int32)
        outputs["inview_article_ts_ranks"] = np.zeros(
            outputs["num_inview"], dtype=np.int32
        )
        outputs["inview_article_ts_ranks_desc"] = compute_rank(
            articles.published_ts[outputs["inview_article_indices"]],
        )
        outputs["article_sentiment_score"] = articles.sentiment_score[
            outputs["inview_article_indices"]
        ]
        outputs["article_published_ts_diff"] = np.clip(
            np.log1p(
                np.abs(
                    imp_ts_inview
                    - articles.published_ts[outputs["inview_article_indices"]]
                )
                / 60**2,
            ),
            0,
            5,
        )
        for col in [
            "topics",
            "category",
        ]:
            outputs[f"article_{col}_ids"] = np.stack(
                pad_sequences(
                    getattr(articles, f"{col}_ids")[outputs["inview_article_indices"]],
                    min_shape=[1],
                    backend="numpy",
                    dtype=np.int32,
                )
            )

        return outputs


class HistoryArticleFeatureExtractor(FeatureExtractor):
    is_available = True

    @property
    def fields(self) -> list[FeatureField]:
        return [
            FeatureField(
                "history_article_indices",
                collate_fn=CollateSequences(pad_value=-1),
            ),
            SinusoidalArticleFeatureField(
                "history_article_ranks",
            ),
            SinusoidalArticleFeatureField(
                "history_article_log_counts",
            ),
            SinusoidalArticleFeatureField(
                "history_article_log_ranks",
            ),
            SinusoidalArticleFeatureField(
                "history_imp_ts_diff",
            ),
            SinusoidalArticleFeatureField(
                "history_matched_topics_normed_counts_rank",
            ),
            SinusoidalArticleFeatureField(
                "history_matched_category_normed_counts_rank",
            ),
            CategoricalArticleHistorySimilarityFeatureField(
                "article_history_topics_ids",
                max_value=ArticleMetadataCounts.topics.value,
                pair_col="article_topics_ids",
            ),
            CategoricalArticleHistorySimilarityFeatureField(
                "article_history_category_ids",
                max_value=ArticleMetadataCounts.category.value,
                pair_col="article_category_ids",
            ),
        ]

    def __call__(
        self,
        users: UserFeature,
        user_index: int,
        articles: ArticleFeature,
        imp_ts_inview: int,
        inview_article_indices: np.ndarray,
        num_inview: int,
        history_mask: np.ndarray,
    ) -> dict:
        outputs = {}

        history_article_indices = users.article_indices[user_index][history_mask]
        history_unique_articles = compute_unique_features(
            history_article_indices,
            ranks=np.arange(len(history_article_indices)) + 1,
            imp_ts=users.impression_ts[user_index][history_mask],
            readtime=users.read_time[user_index][history_mask],
            scroll_percentage=users.scroll_percentage[user_index][history_mask],
        )

        if len(history_article_indices) == 0:
            raise NotImplementedError()

        inview_features = compute_unique_inv_features(
            x=inview_article_indices,
            y=history_unique_articles["values"],
            maxlen=num_inview,
            history_article_ranks=history_unique_articles["ranks"],
            history_article_counts=history_unique_articles["counts"],
            history_imp_ts=history_unique_articles["imp_ts"],
            history_readtime=history_unique_articles["readtime"],
            history_scroll_percentage=history_unique_articles["scroll_percentage"],
        )
        outputs["history_article_indices"] = history_unique_articles["values"]
        outputs["history_article_log_counts"] = np.log1p(
            np.clip(inview_features["history_article_counts"], 0, 3)
        )
        outputs["history_article_ranks"] = inview_features["history_article_ranks"]
        outputs["history_article_log_ranks"] = np.log1p(
            np.clip(inview_features["history_article_ranks"], 0, 20)
        )
        outputs["history_imp_ts_diff"] = np.clip(
            np.log1p((imp_ts_inview - inview_features["history_imp_ts"]) / 60**2),
            0,
            5,
        )
        outputs["history_article_readtime"] = inview_features["history_readtime"]
        outputs["history_article_scroll_percentage"] = inview_features[
            "history_scroll_percentage"
        ]

        for col in [
            "category",
            "topics",
        ]:
            outputs[f"article_history_{col}_ids"] = np.concatenate(
                getattr(articles, f"{col}_ids")[history_article_indices].tolist(),
                dtype=np.int32,
            )
            outputs[f"history_matched_{col}_normed_counts"] = (
                compute_matched_article_feature_counts(
                    feature_ids=getattr(articles, f"{col}_ids"),
                    candidate_article_indices=inview_article_indices,
                    reference_article_indices=history_unique_articles["values"],
                    max_feature_id=ArticleMetadataCounts[col].value,
                )
            )
            outputs[f"history_matched_{col}_normed_counts_rank"] = compute_rank(
                outputs[f"history_matched_{col}_normed_counts"],
            )

        return outputs


class PastArticleFeatureExtractor(FeatureExtractor):
    is_available = USE_PAST_IMP

    @property
    def fields(self) -> list[FeatureField]:
        return [
            SinusoidalArticleFeatureField(
                "past_article_log_counts",
            ),
            SinusoidalArticleFeatureField(
                "past_article_log_ranks",
            ),
        ]

    def __call__(
        self,
        inview_article_indices: np.ndarray,
        num_inview: int,
        past_article_indices: np.ndarray,
    ) -> dict:
        outputs = {}

        if len(past_article_indices) == 0:
            outputs["past_article_log_counts"] = np.array([0] * num_inview)
            outputs["past_article_log_ranks"] = np.array([0] * num_inview)
            return outputs

        flatten_past_article_indices = np.concatenate(past_article_indices)
        flatten_past_article_ranks = np.concatenate(
            [[i + 1] * len(x) for i, x in enumerate(past_article_indices)]
        )
        past_unique_articles = compute_unique_features(
            x=flatten_past_article_indices,
            ranks=flatten_past_article_ranks,
        )
        inview_features = compute_unique_inv_features(
            x=inview_article_indices,
            y=past_unique_articles["values"],
            maxlen=num_inview,
            past_article_ranks=past_unique_articles["ranks"],
            past_article_counts=past_unique_articles["counts"],
        )

        outputs["past_article_log_counts"] = np.log1p(
            np.clip(inview_features["past_article_counts"], 0, 5)
        )
        outputs["past_article_log_ranks"] = np.log1p(
            np.clip(inview_features["past_article_ranks"], 0, 5)
        )

        return outputs


class FutureArticleFeatureExtractor(FeatureExtractor):
    is_available = USE_FUTURE_IMP

    @property
    def fields(self) -> list[FeatureField]:
        return [
            SinusoidalArticleFeatureField(
                "future_article_log_counts",
            ),
            SinusoidalArticleFeatureField(
                "future_article_log_ranks",
            ),
        ]

    def __call__(
        self,
        inview_article_indices: np.ndarray,
        num_inview: int,
        future_article_indices: np.ndarray,
    ) -> dict:
        outputs = {}

        if len(future_article_indices) == 0:
            outputs["future_article_log_counts"] = np.array([0] * num_inview)
            outputs["future_article_log_ranks"] = np.array([0] * num_inview)
            return outputs

        flatten_future_article_indices = np.concatenate(future_article_indices)
        flatten_future_article_ranks = np.concatenate(
            [[i + 1] * len(x) for i, x in enumerate(future_article_indices)]
        )
        future_unique_articles = compute_unique_features(
            x=flatten_future_article_indices,
            ranks=flatten_future_article_ranks,
        )
        inview_features = compute_unique_inv_features(
            x=inview_article_indices,
            y=future_unique_articles["values"],
            maxlen=num_inview,
            future_article_ranks=future_unique_articles["ranks"],
            future_article_counts=future_unique_articles["counts"],
        )

        outputs["future_article_log_counts"] = np.log1p(
            np.clip(inview_features["future_article_counts"], 0, 5)
        )
        outputs["future_article_log_ranks"] = np.log1p(
            np.clip(inview_features["future_article_ranks"], 0, 5)
        )

        return outputs


class GlobalPastArticleFeatureExtractor(FeatureExtractor):
    is_available = USE_PAST_IMP and not USE_FUTURE_IMP

    @property
    def fields(self) -> list[FeatureField]:
        fields = []
        ts_prev, ts_next = -11, -1
        fields += [
            SinusoidalArticleFeatureField(
                f"global_article_{ts_prev}m_{ts_next}m_counts",
            ),
            SinusoidalArticleFeatureField(
                f"global_article_{ts_prev}m_{ts_next}m_normed_counts",
            ),
            SinusoidalArticleFeatureField(
                f"global_article_{ts_prev}m_{ts_next}m_counts_rank",
            ),
            SinusoidalArticleFeatureField(
                f"global_article_{ts_prev}m_{ts_next}m_readtime_sum",
            ),
            SinusoidalArticleFeatureField(
                f"global_article_{ts_prev}m_{ts_next}m_readtime_sum_rank",
            ),
            SinusoidalArticleFeatureField(
                f"global_article_{ts_prev}m_{ts_next}m_readtime_mean",
            ),
        ]

        ts_prev, ts_next = -61, -1
        fields += [
            SinusoidalArticleFeatureField(
                f"global_article_{ts_prev}m_{ts_next}m_normed_counts",
            ),
            SinusoidalArticleFeatureField(
                f"global_article_{ts_prev}m_{ts_next}m_counts_rank",
            ),
            SinusoidalArticleFeatureField(
                f"global_article_{ts_prev}m_{ts_next}m_readtime_sum_rank",
            ),
            SinusoidalArticleFeatureField(
                f"global_article_{ts_prev}m_{ts_next}m_readtime_mean",
            ),
            SinusoidalArticleFeatureField(
                f"global_article_{ts_prev}m_{ts_next}m_scroll_mean",
            ),
            SinusoidalArticleFeatureField(
                f"global_article_{ts_prev}m_{ts_next}m_scroll_zero_ratio",
            ),
        ]

        return fields

    def __call__(
        self,
        articles: ArticleFeature,
        inview_article_indices: np.ndarray,
        inview_elapsed_min: int,
    ) -> dict:
        outputs = {}

        inview_counts = articles.inview_counts[inview_article_indices]
        read_time_sum = articles.read_time_sum[inview_article_indices]
        inview_elapsed_mins = articles.inview_elapsed_mins[inview_article_indices]
        scroll_percentage_sum = articles.scroll_percentage_sum[inview_article_indices]
        scroll_zero_counts = articles.scroll_zero_counts[inview_article_indices]

        # Aggregation features for (-10m, +0m)
        ts_prev, ts_next = -11, -1
        prefix = f"global_article_{ts_prev}m_{ts_next}m"
        counts = compute_ts_agg_features(
            min_ts=inview_elapsed_min + ts_prev,
            max_ts=inview_elapsed_min + ts_next,
            src_ts=inview_elapsed_mins,
            ref_ts=inview_counts,
            agg=np.sum,
        )
        readtimes = compute_ts_agg_features(
            min_ts=inview_elapsed_min + ts_prev,
            max_ts=inview_elapsed_min + ts_next,
            src_ts=inview_elapsed_mins,
            ref_ts=read_time_sum,
            agg=np.sum,
        )
        outputs[f"{prefix}_counts"] = counts
        outputs[f"{prefix}_normed_counts"] = counts / counts.max().clip(1)
        outputs[f"{prefix}_counts_rank"] = compute_rank(counts)
        outputs[f"{prefix}_readtime_sum"] = readtimes
        outputs[f"{prefix}_readtime_mean"] = readtimes / counts.clip(1)
        outputs[f"{prefix}_readtime_sum_rank"] = compute_rank(readtimes)

        # Aggregation features for (-60m, +0m)
        ts_prev, ts_next = -61, -1
        prefix = f"global_article_{ts_prev}m_{ts_next}m"
        counts = compute_ts_agg_features(
            min_ts=inview_elapsed_min + ts_prev,
            max_ts=inview_elapsed_min + ts_next,
            src_ts=inview_elapsed_mins,
            ref_ts=inview_counts,
            agg=np.sum,
        )
        readtimes = compute_ts_agg_features(
            min_ts=inview_elapsed_min + ts_prev,
            max_ts=inview_elapsed_min + ts_next,
            src_ts=inview_elapsed_mins,
            ref_ts=read_time_sum,
            agg=np.sum,
        )
        scroll_percentages = compute_ts_agg_features(
            min_ts=inview_elapsed_min + ts_prev,
            max_ts=inview_elapsed_min + ts_next,
            src_ts=inview_elapsed_mins,
            ref_ts=scroll_percentage_sum,
            agg=np.sum,
        )
        scroll_zeros = compute_ts_agg_features(
            min_ts=inview_elapsed_min + ts_prev,
            max_ts=inview_elapsed_min + ts_next,
            src_ts=inview_elapsed_mins,
            ref_ts=scroll_zero_counts,
            agg=np.sum,
        )
        outputs[f"{prefix}_normed_counts"] = counts / counts.max().clip(1)
        outputs[f"{prefix}_counts_rank"] = compute_rank(counts)
        outputs[f"{prefix}_readtime_sum_rank"] = compute_rank(readtimes)
        outputs[f"{prefix}_readtime_mean"] = readtimes / counts.clip(1)
        outputs[f"{prefix}_scroll_mean"] = scroll_percentages / counts.clip(1)
        outputs[f"{prefix}_scroll_zero_ratio"] = scroll_zeros / counts.clip(1)

        return outputs


class GlobalFutureArticleFeatureExtractor(FeatureExtractor):
    is_available = USE_FUTURE_IMP

    @property
    def fields(self) -> list[FeatureField]:
        fields = []
        # 10m
        ts_prev, ts_next = -5, 5
        fields += [
            SinusoidalArticleFeatureField(
                f"global_article_{ts_prev}m_{ts_next}m_counts",
            ),
            SinusoidalArticleFeatureField(
                f"global_article_{ts_prev}m_{ts_next}m_normed_counts",
            ),
            SinusoidalArticleFeatureField(
                f"global_article_{ts_prev}m_{ts_next}m_counts_rank",
            ),
            SinusoidalArticleFeatureField(
                f"global_article_{ts_prev}m_{ts_next}m_readtime_sum",
            ),
            SinusoidalArticleFeatureField(
                f"global_article_{ts_prev}m_{ts_next}m_readtime_sum_rank",
            ),
            SinusoidalArticleFeatureField(
                f"global_article_{ts_prev}m_{ts_next}m_readtime_mean",
            ),
        ]

        # 1h
        ts_prev, ts_next = -5, 55
        fields += [
            SinusoidalArticleFeatureField(
                f"global_article_{ts_prev}m_{ts_next}m_normed_counts",
            ),
            SinusoidalArticleFeatureField(
                f"global_article_{ts_prev}m_{ts_next}m_counts_rank",
            ),
            SinusoidalArticleFeatureField(
                f"global_article_{ts_prev}m_{ts_next}m_readtime_sum_rank",
            ),
            SinusoidalArticleFeatureField(
                f"global_article_{ts_prev}m_{ts_next}m_readtime_mean",
            ),
            SinusoidalArticleFeatureField(
                f"global_article_{ts_prev}m_{ts_next}m_scroll_mean",
            ),
            SinusoidalArticleFeatureField(
                f"global_article_{ts_prev}m_{ts_next}m_scroll_zero_ratio",
            ),
        ]

        # 1h
        ts_prev, ts_next = -55, 5
        fields += [
            SinusoidalArticleFeatureField(
                f"global_article_{ts_prev}m_{ts_next}m_normed_counts",
            ),
            SinusoidalArticleFeatureField(
                f"global_article_{ts_prev}m_{ts_next}m_counts_rank",
            ),
            SinusoidalArticleFeatureField(
                f"global_article_{ts_prev}m_{ts_next}m_readtime_sum_rank",
            ),
            SinusoidalArticleFeatureField(
                f"global_article_{ts_prev}m_{ts_next}m_readtime_mean",
            ),
            SinusoidalArticleFeatureField(
                f"global_article_{ts_prev}m_{ts_next}m_scroll_mean",
            ),
            SinusoidalArticleFeatureField(
                f"global_article_{ts_prev}m_{ts_next}m_scroll_zero_ratio",
            ),
        ]

        return fields

    def __call__(
        self,
        articles: ArticleFeature,
        inview_article_indices: np.ndarray,
        inview_elapsed_min: int,
    ) -> dict:
        outputs = {}

        inview_counts = articles.inview_counts[inview_article_indices]
        read_time_sum = articles.read_time_sum[inview_article_indices]
        inview_elapsed_mins = articles.inview_elapsed_mins[inview_article_indices]
        scroll_percentage_sum = articles.scroll_percentage_sum[inview_article_indices]
        scroll_zero_counts = articles.scroll_zero_counts[inview_article_indices]

        # 10m
        ts_prev, ts_next = -5, 5
        prefix = f"global_article_{ts_prev}m_{ts_next}m"
        counts = compute_ts_agg_features(
            min_ts=inview_elapsed_min + ts_prev,
            max_ts=inview_elapsed_min + ts_next,
            src_ts=inview_elapsed_mins,
            ref_ts=inview_counts,
            agg=np.sum,
        )
        readtimes = compute_ts_agg_features(
            min_ts=inview_elapsed_min + ts_prev,
            max_ts=inview_elapsed_min + ts_next,
            src_ts=inview_elapsed_mins,
            ref_ts=read_time_sum,
            agg=np.sum,
        )
        outputs[f"{prefix}_counts"] = counts
        outputs[f"{prefix}_normed_counts"] = counts / counts.max().clip(1)
        outputs[f"{prefix}_counts_rank"] = compute_rank(counts)
        outputs[f"{prefix}_readtime_sum"] = readtimes
        outputs[f"{prefix}_readtime_mean"] = readtimes / counts.clip(1)
        outputs[f"{prefix}_readtime_sum_rank"] = compute_rank(readtimes)

        # Future 1h
        ts_prev, ts_next = -5, 55
        prefix = f"global_article_{ts_prev}m_{ts_next}m"
        counts = compute_ts_agg_features(
            min_ts=inview_elapsed_min + ts_prev,
            max_ts=inview_elapsed_min + ts_next,
            src_ts=inview_elapsed_mins,
            ref_ts=inview_counts,
            agg=np.sum,
        )
        readtimes = compute_ts_agg_features(
            min_ts=inview_elapsed_min + ts_prev,
            max_ts=inview_elapsed_min + ts_next,
            src_ts=inview_elapsed_mins,
            ref_ts=read_time_sum,
            agg=np.sum,
        )
        scroll_percentages = compute_ts_agg_features(
            min_ts=inview_elapsed_min + ts_prev,
            max_ts=inview_elapsed_min + ts_next,
            src_ts=inview_elapsed_mins,
            ref_ts=scroll_percentage_sum,
            agg=np.sum,
        )
        scroll_zeros = compute_ts_agg_features(
            min_ts=inview_elapsed_min + ts_prev,
            max_ts=inview_elapsed_min + ts_next,
            src_ts=inview_elapsed_mins,
            ref_ts=scroll_zero_counts,
            agg=np.sum,
        )
        outputs[f"{prefix}_normed_counts"] = counts / counts.max().clip(1)
        outputs[f"{prefix}_counts_rank"] = compute_rank(counts)
        outputs[f"{prefix}_readtime_sum_rank"] = compute_rank(readtimes)
        outputs[f"{prefix}_readtime_mean"] = readtimes / counts.clip(1)
        outputs[f"{prefix}_scroll_mean"] = scroll_percentages / counts.clip(1)
        outputs[f"{prefix}_scroll_zero_ratio"] = scroll_zeros / counts.clip(1)

        # Past 1h
        ts_prev, ts_next = -55, 5
        prefix = f"global_article_{ts_prev}m_{ts_next}m"
        counts = compute_ts_agg_features(
            min_ts=inview_elapsed_min + ts_prev,
            max_ts=inview_elapsed_min + ts_next,
            src_ts=inview_elapsed_mins,
            ref_ts=inview_counts,
            agg=np.sum,
        )
        readtimes = compute_ts_agg_features(
            min_ts=inview_elapsed_min + ts_prev,
            max_ts=inview_elapsed_min + ts_next,
            src_ts=inview_elapsed_mins,
            ref_ts=read_time_sum,
            agg=np.sum,
        )
        scroll_percentages = compute_ts_agg_features(
            min_ts=inview_elapsed_min + ts_prev,
            max_ts=inview_elapsed_min + ts_next,
            src_ts=inview_elapsed_mins,
            ref_ts=scroll_percentage_sum,
            agg=np.sum,
        )
        scroll_zeros = compute_ts_agg_features(
            min_ts=inview_elapsed_min + ts_prev,
            max_ts=inview_elapsed_min + ts_next,
            src_ts=inview_elapsed_mins,
            ref_ts=scroll_zero_counts,
            agg=np.sum,
        )
        outputs[f"{prefix}_normed_counts"] = counts / counts.max().clip(1)
        outputs[f"{prefix}_counts_rank"] = compute_rank(counts)
        outputs[f"{prefix}_readtime_sum_rank"] = compute_rank(readtimes)
        outputs[f"{prefix}_readtime_mean"] = readtimes / counts.clip(1)
        outputs[f"{prefix}_scroll_mean"] = scroll_percentages / counts.clip(1)
        outputs[f"{prefix}_scroll_zero_ratio"] = scroll_zeros / counts.clip(1)

        return outputs


class FutureStatisticsArticleFeatureExtractor(FeatureExtractor):
    is_available = USE_FUTURE_ARTICLE_STATS

    @property
    def fields(self) -> list[FeatureField]:
        return [
            SinusoidalArticleFeatureField(
                "article_total_inviews",
            ),
            SinusoidalArticleFeatureField(
                "article_total_pageviews",
            ),
            SinusoidalArticleFeatureField(
                "article_total_read_time",
            ),
            SinusoidalArticleFeatureField(
                "article_total_inviews_rank",
            ),
            SinusoidalArticleFeatureField(
                "article_total_pageviews_rank",
            ),
            SinusoidalArticleFeatureField(
                "article_total_read_time_rank",
            ),
        ]

    def __call__(
        self,
        articles: ArticleFeature,
        inview_article_indices: np.ndarray,
    ) -> dict:
        outputs = {}
        outputs["article_total_inviews"] = np.log1p(
            articles.total_inviews[inview_article_indices],
        )
        outputs["article_total_pageviews"] = np.log1p(
            articles.total_pageviews[inview_article_indices]
        )
        outputs["article_total_read_time"] = np.log1p(
            articles.total_read_time[inview_article_indices]
        )
        outputs["article_total_inviews_rank"] = compute_rank(
            articles.total_inviews[inview_article_indices],
        )
        outputs["article_total_pageviews_rank"] = compute_rank(
            articles.total_pageviews[inview_article_indices],
        )
        outputs["article_total_read_time_rank"] = compute_rank(
            articles.total_read_time[inview_article_indices],
        )

        return outputs


# -----------------------------------------------------------------------------
#                        FEATURE EXTRACTION PIPELINE
# -----------------------------------------------------------------------------


class FeatureExtractionPipeline:
    feature_extractors = list(
        filter(
            lambda x: x.is_available,
            [
                MetaFieldFeatureExtractor(),
                LabelFeatureExtractor(),
                ImpressionFeatureExtractor(),
                HistoryImpressionFeatureExtractor(),
                PastImpressionFeatureExtractor(),
                FutureImpressionFeatureExtractor(),
                ArticleFeatureExtractor(),
                HistoryArticleFeatureExtractor(),
                PastArticleFeatureExtractor(),
                GlobalPastArticleFeatureExtractor(),
                FutureArticleFeatureExtractor(),
                GlobalFutureArticleFeatureExtractor(),
                FutureStatisticsArticleFeatureExtractor(),
            ],
        )
    )

    @property
    def fields(self) -> list[FeatureField]:
        return [f for x in self.feature_extractors for f in x.fields]

    @classmethod
    def get_fields(
        cls,
        types: type | tuple[type, ...] | None = None,
    ) -> list[FeatureField]:
        return [
            f
            for x in cls.feature_extractors
            for f in x.fields
            if (types is None or isinstance(f, types))
        ]

    def __str__(self) -> str:
        with pl.Config(set_tbl_rows=-1):
            df = pl.DataFrame(
                [
                    {
                        "name": f.name,
                        "type": f.__class__.__name__,
                    }
                    for x in self.feature_extractors
                    for f in x.fields
                    if isinstance(f, InputFeatureField)
                ]
            )
            df = df.with_row_index()
            return str(df)

    def __call__(self, **outputs) -> dict:
        for f in self.feature_extractors:
            parameters = inspect.signature(f).parameters
            _inputs = {k: outputs[k] for k in parameters.keys()}
            _outputs = f(**_inputs)
            names = set([x.name for x in f.fields if x.name is not None]) - set(
                _outputs.keys()
            )
            if len(names) > 0:
                raise ValueError(f"Missing fields ({f.__class__.__name__}): {names}")
            names = set(outputs.keys()) & set(_outputs.keys())
            if len(names) > 0:
                raise ValueError(f"Duplicated fields ({f.__class__.__name__}): {names}")
            outputs.update(_outputs)
        return outputs


# =============================================================================
#                                  DATASET
# =============================================================================


class DataSample(dict):
    @classmethod
    def from_dict(cls, data: dict) -> DataSample:
        outputs = {}
        for f in FeatureExtractionPipeline.get_fields():
            if f.dtype is not None:
                outputs[f.name] = data[f.name].astype(f.dtype)
            elif f.name is not None:
                outputs[f.name] = data[f.name]
        return DataSample(**outputs)


@REGISTRY.add
class DataCollator:
    def __call__(self, samples: list[dict]) -> dict:
        batch = {}
        for f in FeatureExtractionPipeline.get_fields():
            if f.collate_fn is not None:
                batch[f.name] = f.collate_fn([x[f.name] for x in samples])
        return batch


@REGISTRY.add
class Dataset(torch.utils.data.Dataset):
    def __init__(
        self,
        fold: int,
        split: str,
        small: bool,
        debug: bool,
    ) -> None:
        super().__init__()
        self.fold = fold
        self.split = split
        self.small = small
        self.debug = debug
        self.pipeline = FeatureExtractionPipeline()
        self.articles = create_article_feature(
            fold=fold,
            split=split,
            debug=debug,
        )
        self.users = create_user_feature(
            fold=fold,
            split=split,
            debug=debug,
        )
        self.impressions = create_impression_feature(
            fold=fold,
            split=split,
            debug=debug,
            small=small,
        )
        self.splits = list(self.impressions.keys())
        self.split_indices = np.concatenate(
            [
                np.repeat([i], len(x.impression_index)).astype(np.int8)
                for i, x in enumerate(self.impressions.values())
            ]
        )
        self.impression_indices = np.concatenate(
            [x.impression_index for x in self.impressions.values()]
        )

    def __len__(self):
        return len(self.impression_indices)

    def __getitem__(self, idx) -> dict:
        split = str(self.splits[self.split_indices[idx]])
        impression_index = int(self.impression_indices[idx])
        return self.extract_feature(split=split, impression_index=impression_index)

    def extract_feature(self, split: str, impression_index: int):
        # ---  0. Preparation  ---
        features = self.pipeline(
            articles=self.articles[split],
            users=self.users[split],
            impressions=self.impressions[split],
            impression_index=impression_index,
            split=split,
        )

        return DataSample.from_dict(features)


@REGISTRY.add
class BatchSampler:
    def __init__(
        self,
        dataset: Dataset,
        batch_size: int,
        num_workers: int,
        max_samples: int | None,
        max_sample_per_user: int,
        shuffle: bool,
        drop_last: bool,
    ):
        self.dataset = dataset
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.num_epoch = 0
        self.max_samples = max_samples
        self.max_sample_per_user = max_sample_per_user
        self.shuffle = shuffle
        self.drop_last = drop_last
        self.user_indices = np.concatenate(
            [x.user_index for x in dataset.impressions.values()]
        )
        self.is_pseudo_labels = np.concatenate(
            [
                np.repeat(x.is_pseudo_label, len(x.impression_index))
                for x in dataset.impressions.values()
            ]
        )
        self.num_impressions = np.concatenate(
            [x.num_impressions for x in dataset.impressions.values()]
        )
        self.lengths = {}

    @property
    def indices(self):
        indices = np.arange(len(self.user_indices))
        return indices

    @functools.cached_property
    def length(self):
        _, counts = np.unique(self.user_indices, return_counts=True)
        if self.max_samples is not None:
            max_samples = self.max_samples
        elif self.max_sample_per_user is None:
            max_samples = counts.sum()
        else:
            max_samples = np.where(
                counts >= self.max_sample_per_user,
                self.max_sample_per_user,
                counts,
            ).sum()
        return (
            max_samples // self.batch_size
            if self.drop_last
            else math.ceil(max_samples / self.batch_size)
        )

    def __len__(self):
        return self.length

    def __iter__(self):
        self.num_epoch += 1
        user_index_counts = Counter()
        indices = np.copy(self.indices)

        if self.shuffle:
            np.random.shuffle(indices)

        samples = []
        for i in indices:
            user_index = self.user_indices[i]
            if self.max_sample_per_user <= user_index_counts[user_index]:
                continue
            user_index_counts[user_index] += 1
            samples.append(i)

        if self.shuffle:
            np.random.shuffle(samples)
        batch, num_batches = [], 0
        for i in samples:
            batch.append(i)
            if len(batch) == self.batch_size:
                yield batch
                batch, num_batches = [], num_batches + 1
        if len(self) <= num_batches:
            return
        yield batch


# =============================================================================
#                                  MODEL
# =============================================================================


class Lambda(torch.nn.Module):
    def __init__(self, func: Callable):
        super().__init__()
        self.func = func

    def forward(self, x):
        return self.func(x)


class MaskedBatchNormNd(torch.nn.Module):
    def __init__(
        self,
        n: int,
        **kwargs,
    ):
        super().__init__()
        self.norm = eval(f"torch.nn.BatchNorm{n}d")(**kwargs)

    def forward(self, x: torch.Tensor):
        fill_val = self.norm.running_mean.detach().unsqueeze(0)
        for _ in range(len(x.shape) - 2):
            fill_val = fill_val.unsqueeze(-1)
        x = x.where(~x.isnan(), fill_val)
        return self.norm(x)


def embed_sinusoidal(
    x: torch.Tensor,
    embedding_dim: int,
    M: int = 10000,
    padding_idx: int = 0,
) -> torch.Tensor:
    device = x.device
    half_dim = embedding_dim // 2
    emb = np.log(M) / half_dim
    emb = torch.exp(torch.arange(half_dim, device=device) * (-emb))
    emb = x[..., None] * emb[None, ...]
    emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
    emb = emb.masked_fill((x == padding_idx).unsqueeze(-1), 0)
    return emb


class SinusoidalEmbedding(torch.nn.Module):
    def __init__(
        self,
        embedding_dim: int,
        M: int = 10000,
        padding_idx: int = -1,
    ):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.M = M
        self.padding_idx = padding_idx

    def forward(self, x: torch.Tensor):
        return embed_sinusoidal(
            x=x,
            embedding_dim=self.embedding_dim,
            M=self.M,
            padding_idx=self.padding_idx,
        )


class CategoricalArticleEmbedding(torch.nn.Module):
    def __init__(
        self,
        num_embeddings: int,
        embedding_dim: int,
        padding_idx: int = 0,
    ):
        super().__init__()
        self.embedding = torch.nn.Embedding(
            num_embeddings=num_embeddings,
            embedding_dim=embedding_dim,
            padding_idx=padding_idx,
        )

    def forward(self, x: torch.Tensor):
        h = self.embedding(x)
        if len(x.shape) == 3:
            h = h.sum(dim=-2) / ((x > 0).sum(dim=-1, keepdims=True) + 1e-6)
        return h


class CategoricalArticleHistoryEmbedding(torch.nn.Module):
    def __init__(
        self,
        num_embeddings: int,
        embedding_dim: int,
        padding_idx: int = 0,
    ):
        super().__init__()
        self.embedding = torch.nn.Embedding(
            num_embeddings=num_embeddings,
            embedding_dim=embedding_dim,
            padding_idx=padding_idx,
        )

    def forward(self, x: torch.Tensor):
        h = self.embedding(x)
        h = h.sum(dim=-2) / ((x > 0).sum(dim=-1, keepdims=True) + 1e-6)
        h = h.unsqueeze(1)
        return h


class PreTrainedArticleEmbedding(torch.nn.Module):
    def __init__(
        self,
        df_path: str,
        embedding_col: str,
        hidden_dim: int,
        padding_idx: int = 0,
        freeze: bool = True,
        use_proj: bool = False,
    ):
        super().__init__()
        with timer(f"Load pretrained embedding ({embedding_col}): {df_path}"):
            df = pl.read_parquet(df_path, columns=[embedding_col])
            values = torch.tensor(np.stack(df[embedding_col]))
        self.embedding = torch.nn.Embedding.from_pretrained(
            torch.cat(
                [
                    torch.zeros((1, values.shape[1]), dtype=torch.float32),
                    values,
                ]
            ),
            padding_idx=padding_idx,
            freeze=freeze,
        )
        self.use_proj = use_proj
        if use_proj:
            self.proj = torch.nn.Linear(self.embedding.embedding_dim, hidden_dim)

    def forward(self, x: torch.Tensor):
        h = self.embedding(x + 1)
        if self.use_proj:
            h = self.proj(h)
        h = h.masked_fill((x == -1).unsqueeze(-1), 0)
        return h


class ArticleEmbedding(torch.nn.Module):
    def __init__(self, sinusoidal_dim: int, hidden_dim: int, output_dim: int):
        super().__init__()
        # Sinusoidal embedding
        self.sinusoidal_embeddings = torch.nn.ModuleDict(
            {
                f.name: torch.nn.Sequential(
                    SinusoidalEmbedding(
                        embedding_dim=sinusoidal_dim,
                        padding_idx=f.padding_idx,
                    ),
                )
                for f in FeatureExtractionPipeline.get_fields(
                    SinusoidalArticleFeatureField
                )
            }
        )

        # Numerical embedding
        self.numerical_feature_cols = [
            x.name
            for x in FeatureExtractionPipeline.get_fields(NumericalArticleFeatureField)
        ]
        self.numerical_embedding = torch.nn.Sequential(
            Lambda(lambda x: x.permute(0, 2, 1)),
            torch.nn.BatchNorm1d(len(self.numerical_feature_cols)),
            Lambda(lambda x: x.permute(0, 2, 1)),
            torch.nn.Linear(len(self.numerical_feature_cols), hidden_dim),
        )

        # Categorical embedding
        self.categorical_feature_cols = list(
            FeatureExtractionPipeline.get_fields(CategoricalArticleFeatureField)
        )
        self.num_categorical_features = len(self.categorical_feature_cols)
        self.categorical_embeddings = torch.nn.ModuleDict(
            {
                f.name: CategoricalArticleEmbedding(
                    num_embeddings=f.max_value + 1,
                    embedding_dim=hidden_dim,
                    padding_idx=0,
                )
                for f in FeatureExtractionPipeline.get_fields(
                    (
                        CategoricalArticleFeatureField,
                        CategoricalArticleSimilarityFeatureField,
                    )
                )
            }
        )
        self.categorical_history_embeddings = torch.nn.ModuleDict(
            {
                f.name: CategoricalArticleHistoryEmbedding(
                    num_embeddings=f.max_value + 1,
                    embedding_dim=hidden_dim,
                    padding_idx=0,
                )
                for f in FeatureExtractionPipeline.get_fields(
                    CategoricalArticleHistorySimilarityFeatureField
                )
            }
        )
        self.pretrained_embeddings = torch.nn.ModuleList(
            [
                PreTrainedArticleEmbedding(
                    df_path=f.df_path,
                    embedding_col=f.embedding_col,
                    use_proj=f.use_proj,
                    hidden_dim=hidden_dim,
                    padding_idx=0,
                    freeze=True,
                )
                for f in FeatureExtractionPipeline.get_fields(
                    PreTrainedArticleHistorySimilarityFeatureField
                )
            ]
        )

        # Categorical similarity embedding
        self.categorical_similarity_feature_cols = [
            (f.pair_col, f.name)
            for f in FeatureExtractionPipeline.get_fields(
                CategoricalArticleHistorySimilarityFeatureField
            )
        ]
        self.num_categorical_similarity_features = len(
            self.categorical_similarity_feature_cols
        )
        self.num_pretrained_embeddings = len(self.pretrained_embeddings)
        self.categorical_similarity_embeddings_aggregator = torch.nn.Sequential(
            torch.nn.Linear(self.num_categorical_similarity_features, hidden_dim),
        )
        self.pretrained_similarity_embeddings_aggregator = torch.nn.Sequential(
            torch.nn.Linear(self.num_pretrained_embeddings, hidden_dim),
        )

        # Heterogieneous feature aggregator
        embedding_dim = (
            len(self.sinusoidal_embeddings) * sinusoidal_dim
            # (len(self.sinusoidal_embeddings) > 0) * hidden_dim
            + (len(self.numerical_feature_cols) > 0) * hidden_dim
            + self.num_categorical_features * hidden_dim
            + (self.num_categorical_similarity_features > 0) * hidden_dim
            + len(self.pretrained_embeddings) * hidden_dim
        )
        self.aggregator = torch.nn.Sequential(
            torch.nn.Dropout(0.1),
            torch.nn.Linear(embedding_dim, output_dim),
            Lambda(lambda x: x.permute(0, 2, 1)),
            torch.nn.BatchNorm1d(num_features=output_dim),
            Lambda(lambda x: x.permute(0, 2, 1)),
            torch.nn.ReLU(),
        )

    def forward(self, batch: dict):
        hs = []
        # Embed sinusoidal features
        if len(self.sinusoidal_embeddings) > 0:
            xs = [f(batch[k]) for k, f in self.sinusoidal_embeddings.items()]
            hs.append(torch.cat(xs, dim=-1))

        # Embed numerical features
        if len(self.numerical_feature_cols) > 0:
            xs = [batch[k].unsqueeze(-1) for k in self.numerical_feature_cols]
            hs.append(self.numerical_embedding(torch.concat(xs, dim=-1)))

        # Embed categorical features
        if self.num_categorical_features > 0:
            hs.append(
                torch.concat(
                    [
                        self.categorical_embeddings[col](batch[col])
                        for col in self.categorical_feature_cols
                    ],
                    dim=-1,
                )
            )

        # Embed categorical features
        if self.num_categorical_similarity_features > 0:
            xs = [
                F.cosine_similarity(
                    self.categorical_embeddings[src_col](batch[src_col]),
                    self.categorical_history_embeddings[trg_col](batch[trg_col]),
                    dim=2,
                ).unsqueeze(-1)
                for src_col, trg_col in self.categorical_similarity_feature_cols
            ]
            hs.append(
                self.categorical_similarity_embeddings_aggregator(
                    torch.concat(xs, dim=-1)
                )
            )

        if len(self.pretrained_embeddings) > 0:
            history_mask = batch["history_article_indices"] != -1
            xs = [
                F.cosine_similarity(
                    embedding(batch["inview_article_indices"]),
                    (
                        embedding(batch["history_article_indices"]).sum(dim=1)
                        / history_mask.sum(dim=1, keepdim=True).clip(1e-6)
                    )[:, None],
                    dim=2,
                ).unsqueeze(-1)
                for embedding in self.pretrained_embeddings
            ]
            hs.append(
                self.pretrained_similarity_embeddings_aggregator(
                    torch.concat(xs, dim=-1)
                )
            )

        # Aggregate heterogeneous features
        h = torch.concat(hs, dim=-1)
        h = self.aggregator(h)
        h = h.masked_fill((batch["inview_article_indices"] == -1).unsqueeze(-1), 0)
        return h


class ImpressionEmbedding(torch.nn.Module):
    def __init__(self, sinusoidal_dim: int, hidden_dim: int, output_dim: int):
        super().__init__()
        # Sinusoidal embedding
        self.sinusoidal_embeddings = torch.nn.ModuleDict(
            {
                f.name: torch.nn.Sequential(
                    SinusoidalEmbedding(
                        embedding_dim=sinusoidal_dim,
                        padding_idx=f.padding_idx,
                    ),
                )
                for f in FeatureExtractionPipeline.get_fields(
                    SinusoidalImpressionFeatureField
                )
            }
        )
        sinusoidal_embedding_dim = len(self.sinusoidal_embeddings) * sinusoidal_dim

        # Numerical embedding
        self.numerical_feature_cols = [
            f.name
            for f in FeatureExtractionPipeline.get_fields(
                NumericalImpressionFeatureField
            )
        ]
        self.numerical_embedding = torch.nn.Sequential(
            torch.nn.BatchNorm1d(len(self.numerical_feature_cols)),
            torch.nn.Linear(len(self.numerical_feature_cols), hidden_dim),
        )

        # Categorical embedding
        self.categorical_embeddings = torch.nn.ModuleDict(
            {
                f.name: torch.nn.Embedding(
                    num_embeddings=f.max_value + 1,
                    embedding_dim=hidden_dim,
                    padding_idx=0,
                )
                for f in FeatureExtractionPipeline.get_fields(
                    CategoricalImpressionFeatureField
                )
            }
        )
        embedding_dim = (
            sinusoidal_embedding_dim
            + (len(self.numerical_feature_cols) > 0) * hidden_dim
            + len(self.categorical_embeddings) * hidden_dim
        )
        self.aggregator = torch.nn.Sequential(
            torch.nn.Dropout(0.1),
            torch.nn.Linear(embedding_dim, output_dim),
            torch.nn.BatchNorm1d(num_features=output_dim),
            torch.nn.ReLU(),
        )

    def forward(self, batch):
        hs = []
        # Embed sinusoidal features
        if len(self.sinusoidal_embeddings) > 0:
            xs = [f(batch[k]) for k, f in self.sinusoidal_embeddings.items()]
            hs.append(torch.cat(xs, dim=-1))

        # Embed numerical features
        if len(self.numerical_feature_cols) > 0:
            xs = [batch[k].unsqueeze(-1) for k in self.numerical_feature_cols]
            hs.append(self.numerical_embedding(torch.concat(xs, dim=-1)))

        # Embed categorical features
        if len(self.categorical_embeddings) > 0:
            hs.append(
                torch.concat(
                    [f(batch[k]) for k, f in self.categorical_embeddings.items()],
                    dim=-1,
                )
            )
        h = torch.concat(hs, dim=-1)
        h = self.aggregator(h)
        return h


@REGISTRY.add
class Embedding(torch.nn.Module):
    def __init__(
        self,
        sinusoidal_dim: int,
        hidden_dim: int,
        output_dim: int,
    ):
        super().__init__()
        self.article_embedding = ArticleEmbedding(
            sinusoidal_dim=sinusoidal_dim,
            hidden_dim=hidden_dim,
            output_dim=output_dim,
        )
        self.impression_embedding = ImpressionEmbedding(
            sinusoidal_dim=sinusoidal_dim,
            hidden_dim=hidden_dim,
            output_dim=output_dim,
        )

    def forward(self, batch: dict):
        # Article embedding: (bs, num_articles, dim)
        h_articles = self.article_embedding(batch)

        # Impression embedding: (bs, 1, dim)
        # h_imp = self.impression_embedding(batch).unsqueeze(1)

        # 3. Concat imp + article: (bs, num_articles + 1, dim)
        # h = torch.concat([h_imp, h_articles], dim=1)
        h = h_articles
        attention_mask = torch.cat(
            [
                # torch.full((h.shape[0], 1), True, device=h.device),
                batch["inview_article_indices"] != -1,
            ],
            dim=-1,
        )
        return {
            "h": h,
            "attention_mask": attention_mask,
        }


@REGISTRY.add
class GLU(torch.nn.Module):
    def __init__(
        self,
        activation: str = "gelu",
    ):
        super().__init__()
        self.activation = getattr(torch.nn.functional, activation)

    def forward(self, x: torch.Tensor):
        x, gate = x.chunk(2, dim=-1)
        return x * self.activation(gate)


@REGISTRY.add
class TransformerEncoderLayer(torch.nn.TransformerEncoderLayer):
    def __init__(
        self,
        d_model: int,
        nhead: int,
        dim_feedforward: int = 2048,
        dropout: float = 0.1,
        activation: str | Callable[[torch.Tensor], torch.Tensor] = "relu",
        layer_norm_eps: float = 1e-5,
        b2t_connection: bool = False,
        batch_first: bool = True,
        norm_first: bool = False,
        device=None,
        dtype=None,
    ):
        self.b2t_connection = b2t_connection
        super().__init__(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation=activation,
            layer_norm_eps=layer_norm_eps,
            batch_first=batch_first,
            norm_first=norm_first,
            device=device,
            dtype=dtype,
        )
        if isinstance(activation, GLU):
            self.linear1 = torch.nn.Linear(d_model, dim_feedforward * 2)

    def forward(
        self,
        src: torch.Tensor,
        src_mask: torch.Tensor | None = None,
        src_key_padding_mask: torch.Tensor | None = None,
        is_causal: bool = False,
    ) -> torch.Tensor:
        src_key_padding_mask = F._canonical_mask(
            mask=src_key_padding_mask,
            mask_name="src_key_padding_mask",
            other_type=F._none_or_dtype(src_mask),
            other_name="src_mask",
            target_type=src.dtype,
        )

        src_mask = F._canonical_mask(
            mask=src_mask,
            mask_name="src_mask",
            other_type=None,
            other_name="",
            target_type=src.dtype,
            check_other=False,
        )

        x = src
        if self.norm_first:
            x = x + self._sa_block(
                self.norm1(x),
                src_mask,
                src_key_padding_mask,
                is_causal=is_causal,
            )
            if self.b2t_connection:
                x = x + self._ff_block(self.norm2(x)) + src
            else:
                x = x + self._ff_block(self.norm2(x))

        else:
            x = self.norm1(
                x
                + self._sa_block(
                    x,
                    src_mask,
                    src_key_padding_mask,
                    is_causal=is_causal,
                )
            )
            if self.b2t_connection:
                x = self.norm2(x + self._ff_block(x) + src)
            else:
                x = self.norm2(x + self._ff_block(x))
        return x


@REGISTRY.add
class PredictionHead(torch.nn.Module):
    def __init__(
        self,
        hidden_dim: int,
        num_hidden_layers: int,
        output_dim: int,
    ):
        super().__init__()
        layers = []
        for _ in range(num_hidden_layers):
            layers.append(torch.nn.Linear(hidden_dim, hidden_dim))
            layers.append(Lambda(lambda x: x.permute(0, 2, 1)))
            layers.append(torch.nn.BatchNorm1d(hidden_dim))
            layers.append(Lambda(lambda x: x.permute(0, 2, 1)))
            layers.append(torch.nn.ReLU())
        layers.append(torch.nn.Linear(hidden_dim, output_dim))
        self.layers = torch.nn.Sequential(*layers)

    def forward(self, x):
        return self.layers(x)


@REGISTRY.add
class Model(L.LightningModule):
    def __init__(
        self,
        embedding: torch.nn.Module,
        encoder: torch.nn.Module,
        prediction_head: torch.nn.Module,
        metrics_meter: torch.nn.Module,
        optimizer_factory: Callable,
        scheduler_factory: Callable,
    ):
        super().__init__()
        self.embedding = embedding
        self.encoder = encoder
        self.prediction_head = prediction_head
        self.metrics_meter = metrics_meter
        self.optimizer_factory = optimizer_factory
        self.scheduler_factory = scheduler_factory

    def _format_metrics(self, metrics: dict, split: str):
        return {f"{k}/{split}": float(v) for k, v in metrics.items()}

    def forward(self, batch):
        # Feature embedding
        # x: (bs, inview + history, num_token, dim) -> (bs, inview + history, dim)
        xs = self.embedding(batch)

        # Sequence encoding
        # h: (batch_size, num_inview, dim)
        if isinstance(self.encoder, torch.nn.TransformerEncoder):
            h = self.encoder(
                src=xs["h"],
                src_key_padding_mask=~xs["attention_mask"],
            )
            # h = h[:, 1:]
        else:
            h = self.encoder(
                inputs_embeds=xs["h"],
                attention_mask=xs["attention_mask"],
            )
            h = h.last_hidden_state

        # Make prediction
        # preds: (batch_size, num_inview, 1)
        inview_mask = batch["inview_article_indices"] != -1
        preds = self.prediction_head(h).squeeze(-1)
        preds = preds.masked_fill(~inview_mask, 0)

        return {
            "preds": preds,
        }

    def on_train_epoch_start(self):
        self.metrics_meter.reset()

    def training_step(self, batch, batch_idx):
        batch = {**batch, **self.forward(batch)}
        num_batches = len(self.trainer.train_dataloader)
        batch_size = len(batch["impression_id"])

        # Compute metrics
        metrics = self.metrics_meter.update(batch, n_samples=1)
        logging_metrics = self._format_metrics(metrics=metrics, split="train")
        logging_metrics["epoch"] = self.current_epoch + (batch_idx + 1) / num_batches

        self.log_dict(logging_metrics, prog_bar=False, batch_size=batch_size)

        # Compute accum metrics
        accum_metrics = self.metrics_meter.compute(suffix="_accum")
        logging_metrics = self._format_metrics(metrics=accum_metrics, split="train")
        self.log_dict(logging_metrics, prog_bar=True, batch_size=batch_size)

        # Epoch end
        if (batch_idx + 1) == num_batches:
            logging_metrics = {
                k: round(float(v), 5) for k, v in self.metrics_meter.compute().items()
            }
            self.print(f"Training (epoch={self.current_epoch + 1}): {logging_metrics}")
        return metrics["loss"]

    def on_validation_epoch_start(self):
        self.metrics_meter.reset()

    def validation_step(self, batch, batch_idx):
        batch = {**batch, **self.forward(batch)}
        batch_size = len(batch["impression_id"])

        # Compute metrics
        metrics = self.metrics_meter.update(batch, n_samples=None)
        logging_metrics = self._format_metrics(metrics=metrics, split="valid")
        logging_metrics["epoch"] = self.current_epoch + 1
        self.log_dict(logging_metrics, prog_bar=False, batch_size=batch_size)

        # Compute accum metrics
        accum_metrics = self.metrics_meter.compute(suffix="_accum")
        logging_metrics = self._format_metrics(metrics=accum_metrics, split="valid")
        self.log_dict(
            logging_metrics,
            prog_bar=True,
            on_step=True,
            on_epoch=False,
            batch_size=batch_size,
        )

    def on_validation_epoch_end(self):
        if self.trainer.global_step == 0:
            return
        logging_metrics = {
            k: round(float(v), 5) for k, v in self.metrics_meter.compute().items()
        }
        self.print(f"Validation (epoch={self.current_epoch + 1}): {logging_metrics}")

    def on_test_start(self):
        self.preds_df = None
        self.preds = {"impression_id": [], "pred": [], "rank": []}

    def test_step(self, batch, batch_idx):
        batch = {**batch, **self.forward(batch)}
        self.preds["impression_id"].extend(
            batch["impression_id"].cpu().numpy().astype(np.uint32)
        )
        self.preds["pred"].extend(
            [
                y[m == 1].detach().cpu().numpy()
                for y, m in zip(batch["preds"], batch["is_inview"])
            ]
        )
        self.preds["rank"].extend(
            [
                (
                    np.argsort(np.argsort(y[m == 1].detach().cpu().numpy())[::-1]) + 1
                ).astype(np.uint8)
                for y, m in zip(batch["preds"], batch["is_inview"])
            ]
        )

    def on_test_end(self):
        self.preds_df = pl.DataFrame(self.preds)
        self.preds = {}

    def configure_optimizers(self):
        optimizer = self.optimizer_factory(self.parameters())
        scheduler = self.scheduler_factory(optimizer)
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
            },
        }


# =============================================================================
#                                  METRICS
# =============================================================================


class AUC(torchmetrics.MeanMetric):
    def update(self, preds: torch.Tensor, labels: torch.Tensor):
        value = torch.stack(
            [
                torchmetrics.functional.classification.binary_auroc(
                    preds=_preds[_labels != -1],
                    target=_labels[_labels != -1],
                )
                for _preds, _labels in zip(preds, labels)
            ]
        ).mean()
        super().update(value=value, weight=1)


class NDCG5(torchmetrics.MeanMetric):
    def update(self, preds: torch.Tensor, labels: torch.Tensor):
        value = torch.stack(
            [
                torchmetrics.functional.retrieval.retrieval_normalized_dcg(
                    preds=_preds[_labels != -1],
                    target=_labels[_labels != -1],
                    top_k=5,
                )
                for _preds, _labels in zip(preds, labels)
            ]
        ).mean()
        super().update(value=value, weight=1)


class NDCG10(torchmetrics.MeanMetric):
    def update(self, preds: torch.Tensor, labels: torch.Tensor):
        value = torch.stack(
            [
                torchmetrics.functional.retrieval.retrieval_normalized_dcg(
                    preds=_preds[_labels != -1],
                    target=_labels[_labels != -1],
                    top_k=10,
                )
                for _preds, _labels in zip(preds, labels)
            ]
        ).mean()
        super().update(value=value, weight=1)


def binary_listnet_loss(y_pred, y_true, eps=1e-5, padded_value_indicator=-1):
    y_pred = y_pred.clone()
    y_true = y_true.clone()

    mask = y_true == padded_value_indicator
    y_pred[mask] = float("-inf")
    y_true[mask] = 0.0
    normalizer = torch.unsqueeze(y_true.sum(dim=-1), 1)
    normalizer[normalizer == 0.0] = 1.0
    normalizer = normalizer.expand(-1, y_true.shape[1])
    y_true = torch.div(y_true, normalizer)

    preds_smax = torch.softmax(y_pred, dim=1)

    preds_smax = preds_smax + eps
    preds_log = torch.log(preds_smax)

    return torch.mean(-torch.sum(y_true * preds_log, dim=1))


@REGISTRY.add
class MetricsMeter(torch.nn.Module):
    def __init__(self, loss_weights: dict[str, float]):
        super().__init__()
        self.loss_weights = loss_weights
        self.bce_loss = torch.nn.BCEWithLogitsLoss(
            reduction="none",
        )
        self.dice_loss = pytorch_toolbelt.losses.DiceLoss(
            mode="binary",
            ignore_index=-1,
        )
        self.eval_metrics = torchmetrics.MetricCollection({"auc": AUC()})
        self.reset()

    def reset(self):
        self.eval_metrics.reset()

    def update(self, batch: dict, n_samples: int | None = None):
        metrics = {}
        mask = batch["labels"] != -1
        sample_weights = mask.float()

        # BCE Loss
        if "bce_loss" in self.loss_weights:
            bce_losses = self.bce_loss(batch["preds"], batch["labels"].float())
            bce_losses = bce_losses * sample_weights
            metrics["bce_loss"] = bce_losses.sum() / sample_weights.sum()

        # Loss
        metrics["loss"] = sum(w * metrics[k] for k, w in self.loss_weights.items())

        # Eval metrics
        idx = batch["is_pseudo_label"].argsort()
        _preds = batch["preds"][idx][:n_samples]
        _labels = batch["labels"][idx][:n_samples].int()
        metrics.update(self.eval_metrics(preds=_preds, labels=_labels))

        return metrics

    def compute(self, suffix: str = ""):
        return {f"{k}{suffix}": v for k, v in self.eval_metrics.compute().items()}


# =============================================================================
#                                  FACTORY
# =============================================================================


class Factory:
    def __init__(self, config: ExperimentConfig, registry: Registry):
        self.config = config
        self.registry = registry

    def create_model(self, **kwargs) -> L.LightningModule:
        model = self.registry.get_from_params(
            **asdict(self.config.model),
            **kwargs,
        )
        model = typing.cast(L.LightningModule, model)
        return model

    def create_trainer(self, **kwargs) -> L.Trainer:
        name = (
            f"{self.config.common.exp_id}_fold{self.config.common.fold}"
            f"_{datetime.now().strftime('%Y%m%d-%H%M%S')}"
        )
        loggers = [
            L.loggers.CSVLogger(
                save_dir=self.config.common.output_dir,
                version=f"fold_{self.config.common.fold}",
                name=None,
            ),
        ]
        if self.config.common.wandb:
            loggers.append(
                L.loggers.WandbLogger(
                    project=PROJECT_NAME,
                    save_dir=self.config.common.fold_output_dir,
                    id=name,
                    name=name,
                    group=self.config.common.exp_id,
                    config=asdict(self.config),
                    save_code=True,
                )
            )
        callbacks = [
            L.callbacks.ModelCheckpoint(
                monitor=None,
                dirpath=self.config.common.fold_output_dir,
                save_top_k=1,
                filename="{epoch}",
                mode="min",
            ),
            TQDMProgressBarEx(),
            SlackNotificationCallback(
                exp_id=self.config.common.exp_id,
                enable=self.config.common.slack,
            ),
        ]
        trainer = self.registry.get_from_params(
            **asdict(self.config.trainer),
            callbacks=callbacks,
            logger=loggers,
            **kwargs,
        )
        trainer = typing.cast(L.Trainer, trainer)
        return trainer

    def create_dataloader(self, split: str, **kwargs) -> torch.utils.data.DataLoader:
        dataloader = self.registry.get_from_params(
            **asdict(self.config.dataloaders)[split],
            **kwargs,
        )
        dataloader = typing.cast(torch.utils.data.DataLoader, dataloader)
        return dataloader


# =============================================================================
#                                   MAIN
# =============================================================================


@APP.command()
def dryrun(
    split: str = typer.Option("test", help="Split."),
    check_impression_ids: bool = typer.Option(False, help="Check impression ids."),
    override_dotlists: list[str] = typer.Argument(None, help="Override dotlist."),
):
    config = ExperimentConfig.create(override_dotlists)
    factory = Factory(config=config, registry=REGISTRY)
    L.seed_everything(config.common.seed)

    with timer("Prepare experiment module", level="INFO"):
        dataloader = factory.create_dataloader(split)

    with timer("Dryrun", level="INFO"):
        impression_ids = []
        for batch in tqdm(dataloader):
            if check_impression_ids:
                impression_ids.append(batch["impression_id"].cpu().numpy().copy())
        if check_impression_ids:
            impression_ids = np.sort(np.concatenate(impression_ids))
            impression_ids_ = np.sort(dataloader.dataset.behaviors.impression_id)
            assert len(impression_ids) == len(impression_ids_)
            assert np.all(impression_ids == impression_ids_)


@APP.command()
def train(
    override_dotlists: list[str] = typer.Argument(None, help="Override dotlist."),
):
    config = ExperimentConfig.create(override_dotlists)
    factory = Factory(config=config, registry=REGISTRY)
    L.seed_everything(config.common.seed)

    if config.common.fold_output_dir.exists() and not config.common.resume:
        if config.common.overwrite or typer.confirm(
            f"Delete: {config.common.fold_output_dir}"
        ):
            logger.debug(f"Delete: {config.common.fold_output_dir}")
            shutil.rmtree(config.common.fold_output_dir)

    config.common.fold_output_dir.mkdir(parents=True, exist_ok=True)
    with open(config.common.output_dir / "config.yml", "w") as f:
        yaml.dump(asdict(config), f, indent=4)
    with open(config.common.output_dir / "src.py", "w") as f:
        f.write(Path(__file__).read_text())

    with timer("Prepare experiment module", level="INFO"):
        pipeline = FeatureExtractionPipeline()
        logger.debug(pipeline)
        trainer = factory.create_trainer()
        model = factory.create_model()
        train_dataloader = factory.create_dataloader("train")
        if config.common.fold == 0:
            valid_dataloader = factory.create_dataloader("validation_small")
        else:
            valid_dataloader = None

    with timer("Training", level="INFO"):
        trainer.fit(
            model,
            train_dataloaders=train_dataloader,
            val_dataloaders=valid_dataloader,
        )


@APP.command()
def predict(
    split: str = typer.Option("validation", help="Split."),
    override_dotlists: list[str] = typer.Argument(None, help="Override dotlist."),
):
    config = ExperimentConfig.create(override_dotlists)
    factory = Factory(config=config, registry=REGISTRY)
    L.seed_everything(config.common.seed)

    with timer("Prepare experiment module", level="INFO"):
        ckpt_paths = list(config.common.fold_output_dir.glob("epoch=*.ckpt"))
        assert len(ckpt_paths) == 1
        trainer = factory.create_trainer()
        model = factory.create_model().eval()
        model.load_state_dict(
            torch.load(ckpt_paths[0], map_location="cpu")["state_dict"]
        )
        dataloader = factory.create_dataloader(split)

    with timer("Inference", level="INFO"):
        trainer.test(model, dataloader)
        df = trainer.lightning_module.preds_df
        output_path = config.common.fold_output_dir / "predictions" / f"{split}.parquet"
        output_path.parent.mkdir(parents=True, exist_ok=True)
        df.write_parquet(output_path, use_pyarrow=True)


if __name__ == "__main__":
    APP()
