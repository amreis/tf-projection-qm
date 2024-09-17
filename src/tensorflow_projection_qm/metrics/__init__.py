from . import metric
from .avg_local_error import AverageLocalError
from .continuity import ClassAwareContinuity, Continuity
from .distance_consistency import DistanceConsistency
from .jaccard import Jaccard
from .mean_rel_rank_error import MRREData, MRREProj
from .neighborhood_hit import NeighborhoodHit
from .neighbors import FalseNeighbors, MissingNeighbors, TrueNeighbors
from .pearson_correlation import PearsonCorrelation
from .procrustes import Procrustes
from .shepard_goodness import ShepardGoodness
from .stress import NormalizedStress, ScaleNormalizedStress
from .trustworthiness import ClassAwareTrustworthiness, Trustworthiness

_ALL_LOCALIZABLE_METRICS: tuple[metric.LocalizableMetric, ...] = (
    AverageLocalError(),
    ClassAwareContinuity(),
    Continuity(),
    Jaccard(),
    MRREData(),
    MRREProj(),
    NeighborhoodHit(),
    FalseNeighbors(),
    MissingNeighbors(),
    TrueNeighbors(),
    Procrustes(),
    ClassAwareTrustworthiness(),
    Trustworthiness(),
)
_ALL_METRICS: tuple[metric.Metric, ...] = _ALL_LOCALIZABLE_METRICS + (
    DistanceConsistency(),
    PearsonCorrelation(),
    ShepardGoodness(),
    NormalizedStress(),
    ScaleNormalizedStress(),
)
_ALL_METRICS_METRICSET = metric.MetricSet(list(_ALL_METRICS))


def get_all_metrics_runner(defaults: dict) -> metric.MetricSet:
    ms = metric.MetricSet(list(_ALL_METRICS))
    ms.set_default(**defaults)

    return ms


def run_all_metrics(X, X_2d, y, k, n_classes=None, *, as_numpy=False):
    _ALL_METRICS_METRICSET.set_default(k=k, n_classes=n_classes)
    measures = _ALL_METRICS_METRICSET.measure_from_dict({"X": X, "X_2d": X_2d, "y": y})

    if as_numpy:
        measures = {k: v.numpy() for k, v in measures.items()}
    return measures
