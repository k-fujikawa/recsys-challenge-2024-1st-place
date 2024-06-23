import json
import multiprocessing
from itertools import compress
from typing import Iterable

import numpy as np
from tqdm.auto import tqdm

from ebrec.evaluation.metrics import (
    accuracy_score,
    f1_score,
    log_loss,
    mean_squared_error,
    mrr_score,
    ndcg_score,
    roc_auc_score,
)
from ebrec.evaluation.protocols import Metric
from ebrec.evaluation.utils import convert_to_binary


class AccuracyScore(Metric):
    def __init__(self, threshold: float = 0.5):
        self.threshold = threshold
        self.name = "accuracy"

    def calculate(self, y_true: list[np.ndarray], y_pred: list[np.ndarray]) -> float:
        res = np.mean(
            [
                accuracy_score(
                    each_labels, convert_to_binary(each_preds, self.threshold)
                )
                for each_labels, each_preds in zip(y_true, y_pred)
            ]
        )
        return float(res)


class F1Score(Metric):
    def __init__(self, threshold: float = 0.5):
        self.threshold = threshold
        self.name = "f1"

    def calculate(self, y_true: list[np.ndarray], y_pred: list[np.ndarray]) -> float:
        res = np.mean(
            [
                f1_score(each_labels, convert_to_binary(each_preds, self.threshold))
                for each_labels, each_preds in zip(y_true, y_pred)
            ]
        )
        return float(res)


class RootMeanSquaredError(Metric):
    def __init__(self):
        self.name = "rmse"

    def calculate(self, y_true: list[np.ndarray], y_pred: list[np.ndarray]) -> float:
        res = np.mean(
            [
                np.sqrt(mean_squared_error(each_labels, each_preds))
                for each_labels, each_preds in zip(y_true, y_pred)
            ]
        )
        return float(res)


class AucScore(Metric):
    def __init__(self):
        self.name = "auc"

    def calculate(self, y_true: list[np.ndarray], y_pred: list[np.ndarray]) -> float:
        res = np.mean(
            [
                roc_auc_score(each_labels, each_preds)
                for each_labels, each_preds in tqdm(zip(y_true, y_pred))
            ]
        )
        return float(res)


class MultiprocessingAucScore(Metric):
    def __init__(self):
        self.name = "auc"

    def get_impression_auc(self, tmp_target_pred_ar):
        target_list = tmp_target_pred_ar[0]
        pred_list = tmp_target_pred_ar[1]
        return roc_auc_score(target_list, pred_list)

    def calculate(self, y_true: list[np.ndarray], y_pred: list[np.ndarray]) -> float:
        target_pred_ar = list(zip(y_true, y_pred))
        with multiprocessing.Pool(multiprocessing.cpu_count()) as pool:
            auc_list = list(
                tqdm(
                    pool.imap_unordered(self.get_impression_auc, target_pred_ar),
                    total=len(target_pred_ar),
                )
            )
        return np.mean(auc_list)


class LogLossScore(Metric):
    def __init__(self):
        self.name = "logloss"

    def calculate(self, y_true: list[np.ndarray], y_pred: list[np.ndarray]) -> float:
        res = np.mean(
            [
                log_loss(
                    each_labels,
                    [max(min(p, 1.0 - 10e-12), 10e-12) for p in each_preds],
                )
                for each_labels, each_preds in zip(y_true, y_pred)
            ]
        )
        return float(res)


class MrrScore(Metric):
    def __init__(self) -> Metric:
        self.name = "mrr"

    def calculate(self, y_true: list[np.ndarray], y_pred: list[np.ndarray]) -> float:
        mean_mrr = np.mean(
            [
                mrr_score(each_labels, each_preds)
                for each_labels, each_preds in tqdm(zip(y_true, y_pred))
            ]
        )
        return float(mean_mrr)


class MultiprocessingMrrScore(Metric):
    def __init__(self):
        self.name = "mrr"

    def get_impression_mrr(self, tmp_target_pred_ar):
        target_list = tmp_target_pred_ar[0]
        pred_list = tmp_target_pred_ar[1]
        return mrr_score(target_list, pred_list)

    def calculate(self, y_true: list[np.ndarray], y_pred: list[np.ndarray]) -> float:
        target_pred_ar = list(zip(y_true, y_pred))
        with multiprocessing.Pool(multiprocessing.cpu_count()) as pool:
            auc_list = list(
                tqdm(
                    pool.imap_unordered(self.get_impression_mrr, target_pred_ar),
                    total=len(target_pred_ar),
                )
            )
        return np.mean(auc_list)


class NdcgScore(Metric):
    def __init__(self, k: int):
        self.k = k
        self.name = f"ndcg@{k}"

    def calculate(self, y_true: list[np.ndarray], y_pred: list[np.ndarray]) -> float:
        res = np.mean(
            [
                ndcg_score(each_labels, each_preds, self.k)
                for each_labels, each_preds in tqdm(zip(y_true, y_pred))
            ]
        )
        return float(res)


class MultiprocessingNdcgScore(Metric):
    def __init__(self, k: int):
        self.k = k
        self.name = f"ndcg@{k}"

    def get_impression_ndcg(self, tmp_target_pred_ar):
        target_list = tmp_target_pred_ar[0]
        pred_list = tmp_target_pred_ar[1]
        return ndcg_score(target_list, pred_list, self.k)

    def calculate(self, y_true: list[np.ndarray], y_pred: list[np.ndarray]) -> float:
        target_pred_ar = list(zip(y_true, y_pred))
        with multiprocessing.Pool(multiprocessing.cpu_count()) as pool:
            auc_list = list(
                tqdm(
                    pool.imap_unordered(self.get_impression_ndcg, target_pred_ar),
                    total=len(target_pred_ar),
                )
            )
        return np.mean(auc_list)


class MetricEvaluator:
    """
    >>> y_true = [[1, 0, 0], [1, 1, 0], [1, 0, 0, 0]]
    >>> y_pred = [[0.2, 0.3, 0.5], [0.18, 0.7, 0.1], [0.18, 0.2, 0.1, 0.1]]

    >>> met_eval = MetricEvaluator(
            labels=y_true,
            predictions=y_pred,
            metric_functions=[
                AucScore(),
                MrrScore(),
                NdcgScore(k=5),
                NdcgScore(k=10),
                LogLossScore(),
                RootMeanSquaredError(),
                AccuracyScore(threshold=0.5),
                F1Score(threshold=0.5),
            ],
        )
    >>> met_eval.evaluate()
    {
        "auc": 0.5555555555555556,
        "mrr": 0.5277777777777778,
        "ndcg@5": 0.7103099178571526,
        "ndcg@10": 0.7103099178571526,
        "logloss": 0.716399020295845,
        "rmse": 0.5022870658128165
        "accuracy": 0.5833333333333334,
        "f1": 0.2222222222222222
    }
    """

    def __init__(
        self,
        labels: list[np.ndarray],
        predictions: list[np.ndarray],
        metric_functions: list[Metric],
    ):
        self.labels = labels
        self.predictions = predictions
        self.metric_functions = metric_functions
        self.evaluations = dict()

    def evaluate(self) -> dict:
        self.evaluations = {
            metric_function.name: metric_function(self.labels, self.predictions)
            for metric_function in self.metric_functions
        }
        return self

    @property
    def metric_functions(self):
        return self.__metric_functions

    @metric_functions.setter
    def metric_functions(self, values):
        invalid_callables = self.__invalid_callables(values)
        if not any(invalid_callables) and invalid_callables:
            self.__metric_functions = values
        else:
            invalid_objects = list(compress(values, invalid_callables))
            invalid_types = [type(item) for item in invalid_objects]
            raise TypeError(f"Following object(s) are not callable: {invalid_types}")

    @staticmethod
    def __invalid_callables(iter: Iterable):
        return [not callable(item) for item in iter]

    def __str__(self):
        if self.evaluations:
            evaluations_json = json.dumps(self.evaluations, indent=4)
            return f"<MetricEvaluator class>: \n {evaluations_json}"
        else:
            return f"<MetricEvaluator class>: {self.evaluations}"

    def __repr__(self):
        return str(self)
