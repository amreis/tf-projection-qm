from abc import ABC, abstractmethod
from typing import Iterable, TypeVar

import tensorflow as tf

# typing.Self only available in Python>=3.11.
TLocalizableMetric = TypeVar("TLocalizableMetric", bound="LocalizableMetric")
TMetric = TypeVar("TMetric", bound="Metric")


class Metric(ABC):
    name: str

    def __init__(self) -> None:
        super().__init__()

    @property
    @abstractmethod
    def config(self) -> dict: ...

    @abstractmethod
    def measure(self, *args, **kwargs): ...

    @abstractmethod
    def measure_from_dict(self, args: dict): ...

    def set_if_missing(self: TMetric, params) -> TMetric:
        new = type(self)(**self.config)
        for k in new.config:
            if k in params and getattr(self, k) is None:
                setattr(new, k, params[k])
        return new


class LocalizableMetric(Metric):
    def __init__(self) -> None:
        super().__init__()
        self._with_local = False

    def with_local(self: TLocalizableMetric) -> TLocalizableMetric:
        new = type(self)(**self.config)
        new._with_local = True
        return new


class MetricSet:
    def __init__(self, metrics: Iterable[Metric], defaults: dict = {}) -> None:
        self.metrics = list(metrics)
        self.defaults: dict = {} | defaults

    def set_default(self, **kwargs):
        self.defaults |= kwargs

    @tf.function
    def _measure(self, X, X_2d, y=None):
        return {
            m.name: m.set_if_missing(self.defaults).measure_from_dict(
                {"X": X, "X_2d": X_2d, "y": y}
            )
            for m in self.metrics
        }

    def measure_from_dict(self, data_dict: dict):
        # We split the dict and call the underlying measure_from_dict
        # so that we can annotate _measure with @tf.function without
        # causing frequent retracing.
        return self._measure(data_dict["X"], data_dict["X_2d"], data_dict.get("y"))
