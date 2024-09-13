from abc import ABC, abstractmethod
from typing import Iterable, TypeVar

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
        self.defaults: dict = defaults

    def set_default(self, **kwargs):
        self.defaults |= kwargs

    def measure_from_dict(self, data_dict):
        return {
            m.name: m.set_if_missing(self.defaults).measure_from_dict(data_dict)
            for m in self.metrics
        }
