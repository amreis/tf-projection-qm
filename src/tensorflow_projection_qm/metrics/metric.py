from abc import ABC, abstractmethod
from typing import Any, Iterable, Optional, TypeVar

import tensorflow as tf

# typing.Self only available in Python>=3.11.
TLocalizableMetric = TypeVar("TLocalizableMetric", bound="LocalizableMetric")
TMetric = TypeVar("TMetric", bound="Metric")


class Metric(ABC):
    name: str

    def __init__(self) -> None:
        super().__init__()

    def __setattr__(self, name: str, value: Any) -> None:
        if name.startswith("_"):
            super().__setattr__(name, value)

        if getattr(self, name, None) is None:
            self.__dict__[name] = tf.Variable(value) if value is not None else None
        elif isinstance(_var := getattr(self, name), tf.Variable):
            _var.assign(value)
        else:
            self.__dict__[name] = tf.Variable(value)

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
        self.defaults: dict[str, Optional[tf.Variable]] = {} | defaults

    def set_default(self, **kwargs):
        for k, v in kwargs.items():
            if isinstance(_var := self.defaults.get(k), tf.Variable):
                _var.assign(v)
            else:
                self.defaults[k] = tf.Variable(v)

    def _unique_name_for(self, m: Metric) -> str:
        same_metric = [m_i for m_i in self.metrics if type(m) is type(m_i)]
        if len(same_metric) == 1:
            return m.name

        params_vals = [{(k, v) for k, v in m_i.config.items()} for m_i in same_metric]
        redundant_params = params_vals[0]
        # params that are the same across all instances of the same metric
        # do not need to make it into the identifier.
        redundant_params.intersection_update(*params_vals[1:])
        return f'{m.name}_{"_".join(f"{k}={v}" for k, v in sorted(m.config.items() - redundant_params))}'

    @tf.function
    def _measure(self, X, X_2d, y=None):
        return {
            self._unique_name_for(m): m.set_if_missing(self.defaults).measure_from_dict(
                {"X": X, "X_2d": X_2d, "y": y}
            )
            for m in self.metrics
        }

    def measure_from_dict(self, data_dict: dict):
        # We split the dict and call the underlying measure_from_dict
        # so that we can annotate _measure with @tf.function without
        # causing frequent retracing.
        return self._measure(data_dict["X"], data_dict["X_2d"], data_dict.get("y"))
