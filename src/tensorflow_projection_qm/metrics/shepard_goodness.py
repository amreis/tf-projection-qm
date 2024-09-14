import scipy.stats
import tensorflow as tf

from tensorflow_projection_qm.metrics.metric import Metric
from tensorflow_projection_qm.util import distance


def shepard_goodness(X, X_2d):
    flat_d_high = distance.flat_psqdist(X)
    flat_d_low = distance.flat_psqdist(X_2d)
    return tf.convert_to_tensor(scipy.stats.spearmanr(flat_d_high.numpy(), flat_d_low.numpy())[0])


class ShepardGoodness(Metric):
    name = "shepard_goodness"

    def __init__(self) -> None:
        super().__init__()

    @property
    def config(self):
        return {}

    def measure(self, X, X_2d):
        return shepard_goodness(X, X_2d)

    def measure_from_dict(self, args: dict):
        return self.measure(args["X"], args["X_2d"])
