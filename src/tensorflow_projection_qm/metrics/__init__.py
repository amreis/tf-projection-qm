from . import (
    continuity,
    distance_consistency,
    jaccard,
    mean_rel_rank_error,
    metric,
    neighborhood_hit,
    neighbors,
    stress,
    trustworthiness,
)

_ALL_LOCALIZABLE_METRICS: tuple[metric.LocalizableMetric, ...] = (
    continuity.Continuity(),
    jaccard.Jaccard(),
    mean_rel_rank_error.MRREData(),
    mean_rel_rank_error.MRREProj(),
    neighborhood_hit.NeighborhoodHit(),
    neighbors.FalseNeighbors(),
    neighbors.MissingNeighbors(),
    neighbors.TrueNeighbors(),
    trustworthiness.Trustworthiness(),
)
_ALL_METRICS: tuple[metric.Metric, ...] = _ALL_LOCALIZABLE_METRICS + (
    distance_consistency.DistanceConsistency(),
    stress.NormalizedStress(),
    stress.ScaleNormalizedStress(),
)


def run_all_metrics(X, X_2d, y, k, *, as_numpy=False):
    ms = metric.MetricSet(list(_ALL_METRICS))
    ms.set_default(k=k)
    measures = ms.measure_from_dict({"X": X, "X_2d": X_2d, "y": y})

    if as_numpy:
        measures = {k: v.numpy() for k, v in measures.items()}
    return measures
