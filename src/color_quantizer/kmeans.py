"""K-means clustering engine (pure NumPy).

Points are rows of an (N, D) array (for images: RGB pixels, D = 3). Distances
are squared Euclidean and always computed with vectorized NumPy operations.
"""

from dataclasses import dataclass
from typing import Literal

import numpy as np

SeedLike = int | np.random.Generator | None


@dataclass
class KMeansResult:
    """Output of :func:`kmeans`.

    Attributes:
        centroids: (k, D) float64 cluster centers.
        labels: (N,) index of the nearest centroid for each point.
        n_iter: Number of update iterations performed.
        inertia: Sum of squared distances from each point to its centroid.
    """

    centroids: np.ndarray
    labels: np.ndarray
    n_iter: int
    inertia: float


def squared_distances(points: np.ndarray, centroids: np.ndarray) -> np.ndarray:
    """Squared Euclidean distance between every point and every centroid.

    Uses ||x||^2 - 2 x.c + ||c||^2 so memory is O(N * k), not O(N * k * D).

    Args:
        points: (N, D) array.
        centroids: (k, D) array.

    Returns:
        (N, k) float64 array; entry [i, j] is ||points[i] - centroids[j]||^2.
    """
    x = np.asarray(points, dtype=np.float64)
    c = np.asarray(centroids, dtype=np.float64)
    d2 = (
        np.einsum("ij,ij->i", x, x)[:, None]
        - 2.0 * (x @ c.T)
        + np.einsum("ij,ij->i", c, c)[None, :]
    )
    # Rounding can push exact zeros slightly negative.
    return np.maximum(d2, 0.0)


def assign(points: np.ndarray, centroids: np.ndarray) -> np.ndarray:
    """Index of the nearest centroid for each point, shape (N,)."""
    return np.argmin(squared_distances(points, centroids), axis=1)


def init_random(points: np.ndarray, k: int, rng: np.random.Generator) -> np.ndarray:
    """Pick k distinct points (by index) uniformly at random as initial centroids."""
    idx = rng.choice(len(points), size=k, replace=False)
    return np.asarray(points, dtype=np.float64)[idx].copy()


def init_kmeans_plus_plus(
    points: np.ndarray, k: int, rng: np.random.Generator
) -> np.ndarray:
    """k-means++ initialization.

    The first centroid is a uniform random point; each subsequent one is drawn
    with probability proportional to its squared distance to the nearest
    centroid chosen so far. If every remaining distance is zero (e.g. all points
    identical), falls back to a uniform draw.
    """
    x = np.asarray(points, dtype=np.float64)
    n = len(x)
    centroids = np.empty((k, x.shape[1]), dtype=np.float64)
    centroids[0] = x[rng.integers(n)]
    closest = squared_distances(x, centroids[:1])[:, 0]
    for j in range(1, k):
        total = closest.sum()
        if total > 0:
            idx = rng.choice(n, p=closest / total)
        else:
            idx = rng.integers(n)
        centroids[j] = x[idx]
        closest = np.minimum(closest, squared_distances(x, centroids[j : j + 1])[:, 0])
    return centroids


def update_centroids(points: np.ndarray, labels: np.ndarray, k: int) -> np.ndarray:
    """Recompute centroids as the mean of their assigned points.

    An empty cluster is re-seeded at the point farthest from its nearest
    (non-empty) centroid. That point's distance is then zeroed so several
    empty clusters never receive the same point.

    Args:
        points: (N, D) array.
        labels: (N,) cluster index per point, values in [0, k).
        k: Number of clusters.

    Returns:
        (k, D) float64 array of new centroids.
    """
    x = np.asarray(points, dtype=np.float64)
    counts = np.bincount(labels, minlength=k)
    sums = np.zeros((k, x.shape[1]), dtype=np.float64)
    np.add.at(sums, labels, x)

    nonempty = counts > 0
    centroids = np.zeros_like(sums)
    centroids[nonempty] = sums[nonempty] / counts[nonempty, None]

    empty = np.flatnonzero(~nonempty)
    if empty.size:
        if nonempty.any():
            closest = squared_distances(x, centroids[nonempty]).min(axis=1)
        else:
            closest = np.zeros(len(x))
        for j in empty:
            far = int(np.argmax(closest))
            centroids[j] = x[far]
            closest[far] = 0.0
    return centroids


def kmeans(
    points: np.ndarray,
    k: int,
    *,
    max_iter: int = 100,
    tol: float = 1e-4,
    init: Literal["k-means++", "random"] = "k-means++",
    seed: SeedLike = None,
) -> KMeansResult:
    """Cluster points into k groups with Lloyd's algorithm.

    Stops when no assignment changes, when the largest centroid shift is below
    ``tol``, or after ``max_iter`` iterations.

    Args:
        points: (N, D) array of points.
        k: Number of clusters, 1 <= k <= N.
        max_iter: Maximum number of iterations (>= 1).
        tol: Centroid-shift tolerance, in the same units as the points.
        init: "k-means++" (default) or "random".
        seed: Seed or Generator for ``numpy.random.default_rng``.

    Returns:
        A :class:`KMeansResult`.

    Raises:
        ValueError: On invalid shapes or parameters.
    """
    x = np.asarray(points, dtype=np.float64)
    if x.ndim != 2 or len(x) == 0:
        raise ValueError(f"Expected a non-empty (N, D) array, got shape {x.shape}")
    if not 1 <= k <= len(x):
        raise ValueError(f"k must be between 1 and {len(x)}, got {k}")
    if max_iter < 1:
        raise ValueError(f"max_iter must be >= 1, got {max_iter}")

    rng = np.random.default_rng(seed)
    if init == "k-means++":
        centroids = init_kmeans_plus_plus(x, k, rng)
    elif init == "random":
        centroids = init_random(x, k, rng)
    else:
        raise ValueError(f"Unknown init method: {init!r}")

    labels = assign(x, centroids)
    n_iter = 0
    for n_iter in range(1, max_iter + 1):
        new_centroids = update_centroids(x, labels, k)
        shift = np.max(np.sum((new_centroids - centroids) ** 2, axis=1))
        centroids = new_centroids
        new_labels = assign(x, centroids)
        changed = np.any(new_labels != labels)
        labels = new_labels
        if not changed or shift <= tol**2:
            break

    inertia = float(np.sum((x - centroids[labels]) ** 2))
    return KMeansResult(centroids, labels, n_iter, inertia)
