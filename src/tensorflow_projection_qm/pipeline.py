from dataclasses import dataclass
import abc
from copy import deepcopy
from typing import Optional, ClassVar, Union


class Context(dict):
    def __init__(self) -> None:
        super().__init__()

    def augment(self, resource_id: str, value) -> "Context":
        new = deepcopy(self)
        new[resource_id] = value
        return new


@dataclass
class Step:
    step_name: ClassVar[str]
    provides: list[str]
    requires: list[str]
    name: Optional[str]

    @abc.abstractmethod
    def compute(self, ctx: Context):
        pass

    def run(self, ctx: Context) -> Context:
        if any(k not in ctx for k in self.requires):
            raise ValueError()
        output = self.compute(ctx)
        for rid, out in zip(self.provides, output):
            ctx = ctx.augment(rid, out)
        return ctx


class Dataset(Step):
    step_name = "dataset"

    def __init__(self, data, name=None) -> None:
        super().__init__(requires=[], provides=["data"], name=name)

        self._data = deepcopy(data)

    def compute(self, ctx: Context):
        return [self._data]


class Projection(Step):
    step_name = "projection"

    def __init__(self, proj_builder: type, name=None) -> None:
        super().__init__(requires=["data"], provides=["projection"], name=name)

        self._proj_builder = proj_builder

    def compute(self, ctx: Context):
        data = ctx.get("data")
        proj = self._proj_builder().fit_transform(data)

        return [proj]


class Metric(Step):
    step_name = "metric"

    def __init__(self, metric_fn, name=None) -> None:
        super().__init__(requires=["data", "projection"], provides=["metric_value"], name=name)

        self.requires = ["data", "projection"]
        self.provides = ["metric_value"]
        self.metric_fn = metric_fn
        self.name = name

    def compute(self, ctx: Context):
        data = ctx.get("data")
        proj = ctx.get("projection")

        val = self.metric_fn(data, proj)
        return [val]


class Pipeline:
    def __init__(self, steps: list[list[Step]]):
        self.steps = steps

    def run(self):
        from itertools import product

        results = []
        for path in product(*self.steps):
            ctx = Context()
            identifier = {}
            for step in path:
                identifier[step.step_name] = step.name
                ctx = step.run(ctx)
            results.append(identifier | {"output": {k: ctx.get(k) for k in step.provides}})
        return results
