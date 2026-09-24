"""Tests for color_quantizer.kmeans on tiny known datasets."""

import numpy as np
import pytest

from color_quantizer.kmeans import (
    assign,
    init_kmeans_plus_plus,
    init_random,
    kmeans,
    squared_distances,
    update_centroids,
)


def blobs(centers, n_per=20, spread=1.0, seed=0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return np.vstack(
        [c + rng.normal(0, spread, size=(n_per, 3)) for c in np.asarray(centers, float)]
    )


def test_squared_distances_matches_brute_force():
    rng = np.random.default_rng(1)
    x, c = rng.random((30, 3)) * 255, rng.random((4, 3)) * 255
    expected = ((x[:, None, :] - c[None, :, :]) ** 2).sum(axis=2)
    np.testing.assert_allclose(squared_distances(x, c), expected, rtol=1e-9, atol=1e-6)


def test_squared_distances_never_negative():
    x = np.full((5, 3), 123.456)
    assert (squared_distances(x, x[:1]) >= 0).all()


def test_assign_picks_nearest():
    c = np.array([[0, 0, 0], [100, 100, 100]], float)
    x = np.array([[1, 2, 3], [90, 99, 101], [49, 49, 49]], float)
    np.testing.assert_array_equal(assign(x, c), [0, 1, 0])


@pytest.mark.parametrize("init", ["random", "k-means++"])
def test_recovers_separated_blobs(init):
    centers = [[20, 20, 20], [128, 30, 200], [230, 230, 10]]
    x = blobs(centers)
    res = kmeans(x, 3, init=init, seed=0)
    found = sorted(map(tuple, np.round(res.centroids, -1)))
    assert found == sorted(map(tuple, np.round(np.array(centers, float), -1)))
    # Each blob ends up in exactly one cluster.
    for b in range(3):
        assert len(set(res.labels[b * 20 : (b + 1) * 20])) == 1


def test_k1_is_mean():
    x = blobs([[10, 10, 10], [50, 60, 70]])
    res = kmeans(x, 1, seed=0)
    np.testing.assert_allclose(res.centroids[0], x.mean(axis=0))


def test_k_equals_n_gives_points():
    x = np.array([[0, 0, 0], [10, 0, 0], [0, 10, 0], [0, 0, 10]], float)
    res = kmeans(x, 4, seed=3)
    assert res.inertia == pytest.approx(0.0)
    assert sorted(map(tuple, res.centroids)) == sorted(map(tuple, x))


def test_reproducible_with_seed():
    x = np.random.default_rng(5).random((200, 3)) * 255
    a, b = kmeans(x, 5, seed=42), kmeans(x, 5, seed=42)
    np.testing.assert_array_equal(a.centroids, b.centroids)
    np.testing.assert_array_equal(a.labels, b.labels)


def test_inertia_non_increasing_with_more_iterations():
    x = np.random.default_rng(7).random((300, 3)) * 255
    inertias = [kmeans(x, 6, max_iter=m, seed=1, tol=0).inertia for m in range(1, 8)]
    assert all(b <= a + 1e-9 for a, b in zip(inertias, inertias[1:]))


def test_stops_before_max_iter_when_converged():
    x = blobs([[0, 0, 0], [200, 200, 200]])
    assert kmeans(x, 2, max_iter=100, seed=0).n_iter < 100


def test_init_random_distinct_points():
    x = np.arange(30, dtype=float).reshape(10, 3)
    c = init_random(x, 10, np.random.default_rng(0))
    assert len({tuple(r) for r in c}) == 10


def test_kmeans_plus_plus_distinct_centers():
    x = blobs([[0, 0, 0], [100, 0, 0], [0, 100, 0], [0, 0, 100]], n_per=5)
    c = init_kmeans_plus_plus(x, 4, np.random.default_rng(0))
    assert len({tuple(r) for r in c}) == 4
    # Spread-out init: one center per blob.
    assert len(set(assign(c, np.array([[0, 0, 0], [100, 0, 0], [0, 100, 0], [0, 0, 100]], float)))) == 4


def test_kmeans_plus_plus_handles_identical_points():
    x = np.full((10, 3), 7.0)
    c = init_kmeans_plus_plus(x, 3, np.random.default_rng(0))
    np.testing.assert_array_equal(c, np.full((3, 3), 7.0))
    res = kmeans(x, 3, seed=0)
    assert np.isfinite(res.centroids).all()
    assert res.inertia == 0.0


def test_empty_cluster_reseeded_at_farthest_point():
    x = np.array([[v, v, v] for v in (0, 1, 4, 10)], float)
    labels = np.array([0, 0, 0, 1])  # cluster 2 is empty
    c = update_centroids(x, labels, 3)
    np.testing.assert_allclose(c[0], [5 / 3] * 3)
    np.testing.assert_allclose(c[1], [10] * 3)
    np.testing.assert_allclose(c[2], [4] * 3)  # farthest from its nearest centroid


def test_multiple_empty_clusters_get_different_points():
    x = np.array([[v, v, v] for v in (0, 1, 4, 10)], float)
    labels = np.zeros(4, dtype=int)  # clusters 1 and 2 empty
    c = update_centroids(x, labels, 3)
    assert len({tuple(r) for r in c}) == 3


@pytest.mark.parametrize("k", [0, -1, 5])
def test_invalid_k_raises(k):
    with pytest.raises(ValueError):
        kmeans(np.zeros((4, 3)), k)


def test_invalid_params_raise():
    x = np.zeros((4, 3))
    with pytest.raises(ValueError):
        kmeans(x, 2, max_iter=0)
    with pytest.raises(ValueError):
        kmeans(x, 2, init="bogus")
    with pytest.raises(ValueError):
        kmeans(np.zeros((0, 3)), 1)
