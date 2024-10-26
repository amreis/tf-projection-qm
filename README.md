[![Python package](https://github.com/amreis/tf-projection-qm/actions/workflows/python-package.yml/badge.svg)](https://github.com/amreis/tf-projection-qm/actions/workflows/python-package.yml)
# Accelerated Projection Quality Metrics

When evaluating Dimensionality Reduction (AKA Projection) techniques, a number of quality metrics
are usually employed.

These quality metrics are numeric ways of evaluating a projection, and might be useful to determine
whether a sane projection has been produced by an algorithm (e.g, t-SNE, or UMAP).

In this repository, I aim to provide a comprehensive set of implementations of projection
quality metrics that are fast and use idiomatic TensorFlow in their implementation.

## Quality Metrics

A quality metric is a function $\mathcal{M}_\eta$ with two arguments: a dataset $\mathbf{X} \in \mathbb{R}^{n\times D}$ of $D$-dimensional data points, and a corresponding projection $\mathbf{Y} = \mathcal{P}(\mathbb{X}) \in \mathbb{R}^{n\times d}$ where $d$ is usually 2 or 3. We represent by $\eta$ the hyperparameters associated with $\mathcal{M}$ -- for instance, the size $k$ of the neighborhood in neighborhood-based metrics such as trustworthiness, continuity, and neighborhood hit.

Projection algorithms can generate $\mathbf{Y}$ in many ways. Of course, not all such projections are equally useful and/or truthful to the data they are based on. While some techniques might be better at representing global aspects of the original dataset $\mathbf{X}$, others might instead favor local neighborhood preservation.

Each $\mathcal{M}_\eta(\mathbf{X}, \mathbf{Y})$ returns a single score representing the quality of $\mathbf{Y}$ as a projection for $\mathbb{X}$. Different quality metrics aim to evaluate different aspects of _data pattern preservation_. For example, Trustworthiness is a metric that aims to evaluate the amount of false neighbors introduced in a projection -- that is to say, points that were not close in $D$-dimensional space and have been _wrongfully_ brought together by $\mathcal{P}$. Stress is another metric, aimed at measuring discrepancies in pairwise distances in $\mathbf{X}$ when compared to pairwise distances in $\mathbf{Y}$.

## Installation

Installation is possible using `pip` directly:

```bash
pip install tensorflow-projection-qm
```

If you have CUDA available, explicitly enable the CUDA-capable TensorFlow dependency by installing with

```bash
pip install tensorflow-projection-qm[and-cuda]
```

## Using

The functions that calculate the quality metrics all sit in the `tensorflow_projection_qm.metrics` package.

```python
from tensorflow_projection_qm.metrics import continuity, trustworthiness

# Set up some fake data
import numpy as np
X = np.random.randn(100, 5)  # 100 data points with 5 dimensions.

# Project to 2-D with TSNE
from sklearn.manifold import TSNE
X_proj = TSNE(n_components=2).fit_transform(X).astype(X.dtype)

# Evaluate the projection:
C = continuity.continuity(X, X_proj, k=21).numpy()
T = trustworthiness.trustworthiness(X, X_proj, k=21).numpy()
print(f"Continuity: {C}")
print(f"Trustworthiness: {T}")

# Compute per-point value of a metric (not all metrics support this)
C_i = continuity.continuity_with_local(X, X_proj, k=21)[1].numpy()
T_i = trustworthiness.trustworthiness_with_local(X, X_proj, k=21)[1].numpy()

print(f"Per-point Continuity: {C_i}")
print(f"Per-point Trustworthiness: {T_i})
```

## Implemented metrics

* Average Local Error [\[7\]](#ale-and-neighbors)
* Class-Aware Continuity [\[3\]](#class-aware-tnc)
* Class-Aware Trustworthiness [\[3\]](#class-aware-tnc)
* Continuity [\[1\]](#continuity-trustworthiness)
* Distance Consistency [\[5\]](#dsc)
* False Neighbors [\[7\]](#ale-and-neighbors)
* Jaccard Dissimilarity of Neighbor Sets
* Mean Relative Rank Errors [\[2\]](#mrre)
* Missing Neighbors [\[7\]](#ale-and-neighbors)
* Neighborhood Hit [\[9\]](#nh)
* Normalized Stress [\[8\]](#stress)
* Pearson Correlation of Distances [\[11\]](#pearson-r)
* Procrustes Statistic [\[4\]](#procrustes)
* Scale-Normalized Stress [\[10\]](#scale-norm-stress)
* Shepard Goodness [\[6\]](#shep-good)
* True Neighbors [\[7\]](#ale-and-neighbors)
* Trustworthiness [\[1\]](#continuity-trustworthiness)

[\[1\]](https://doi.org/10.1016/j.neunet.2006.05.014) <a name="continuity-trustworthiness"></a>Venna and Kaski. Local Multidimensional Scaling. 2006.

[\[2\]](https://doi.org/10.1016/j.neucom.2008.12.017) <a name="mrre"></a>Lee and Verleysen. Quality assessment of dimensionality reduction: Rank-based criteria. 2008.

[\[3\]](https://proceedings.neurips.cc/paper/2020/hash/99607461cdb9c26e2bd5f31b12dcf27a-Abstract.html) <a name="class-aware-tnc"></a>Colange et al. Steering Distortions to Preserve Classes and Neighbors in Supervised Dimensionality Reduction. 2020.

[\[4\]](https://doi.org/10.1007/s10994-009-5107-9) <a name="procrustes"></a>Goldberg and Ritov. Local procrustes for manifold embedding: a measure of embedding quality and embedding algorithms. 2009.

[\[5\]](https://doi.org/10.1111/j.1467-8659.2009.01467.x) <a name="dsc"></a> Sips et al. Selecting good views of high-dimensional data using class consistency. 2009.

[\[6\]](https://journals.lww.com/jonmd/abstract/1957/07000/nonparametric_statistics_for_the_behavioral.32.aspx) <a name="shep-good"></a>Sidney. Nonparametric statistic for the behavioral sciences. 1957.

[\[7\]](https://doi.org/10.1016/j.cag.2014.01.006) <a name="ale-and-neighbors"></a>Martins et al. Visual analysis of dimensionality reduction quality for parameterized projections. 2014.

[\[8\]](https://doi.org/10.1109/TVCG.2011.220) <a name="stress"></a> Joia et al. Local Affine Multidimensional Projection. 2011.

[\[9\]](https://doi.org/10.1109/TVCG.2007.70443) <a name="nh"></a>Paulovich et al. Least Square Projection: A Fast High-Precision Multidimensional Projection Technique and Its Application to Document Mapping. 2007.

[\[10\]](https://doi.org/10.48550/arXiv.2408.07724) <a name="scale-norm-stress"></a>Smelser et al. "Normalized Stress" is Not Normalized: How to Interpret Stress Correctly. Preprint, 2024.

[\[11\]](https://doi.org/10.1109/TSMCB.2005.850151) <a name="pearson-r"></a>Geng et al. Supervised nonlinear dimensionality reduction for visualization and classification. 2005.

## Why this package?

I have a recurring need in my research (see [About Me](#about-me) below) to evaluate different projection algorithms with respect to different quality metrics. While there are some libraries for this, and I am grateful for their authors' work in gathering and implementing different quality metrics (see, for example, [ZADU](https://github.com/hj-n/zadu)), I have found some implementations to not be as performant as I need them to be (keep in mind I evaluate thousands of projections at a time), and sometimes buggy.

At some point I noticed I had been re-implementing the same quality metrics over and over again, sometimes introducing bugs myself due to mistakes when copying and adapting code from a public source, such as Espadoto's comprehensive [survey](https://github.com/mespadoto/dlmp).

Instead, I have chosen to start this package with the goals of:

1. Having easy access to standard implementations of projection quality metrics;
2. Implementing quality metrics in _vectorized_ manners as often as possible, taking advantage of parallel execution for speeding up calculations;
3. Sharing this code openly as my first package to be published on PyPi.org;
4. Using an easily-available framework (TensorFlow) to back up my implementations and seamlessly take advantage of GPUs when available.

## About

This package is under active development, and is **very much** in its early stages. Please feel free to report bugs, but also be mindful that this is a best-effort attempt to generalize/speed up my own implementations of quality metrics.

## About Me

My name is Alister Machado, I am a PhD Candidate researching Data Visualization (more specifically focused in dimensionality reduction and explainable AI). I am the person behind [ShaRP](https://github.com/amreis/sharp) and the [Differentiable DBMs](https://github.com/amreis/differentiable-dbm). You can check out my research [here](https://scholar.google.com.br/citations?user=WVXX6mYAAAAJ&hl=en). I am currently in the 4th year of my PhD (out of 5 total), and am expected to graduate in 2026. Feel free to reach out!
