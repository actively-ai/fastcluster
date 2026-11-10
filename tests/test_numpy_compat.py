"""Clean-wheel regressions for NumPy 2 and the fork's weighted linkage."""

import unittest

import numpy as np
from numpy.testing import assert_allclose, assert_array_equal
from scipy.cluster.hierarchy import linkage as scipy_linkage

from fastcluster import linkage, linkage_vector


class NumpyCompatibilityTest(unittest.TestCase):
    def test_weighted_members_allow_lists_and_dtype_conversion(self):
        for members in [[2, 1, 1], np.array([2, 1, 1], dtype=np.int64)]:
            with self.subTest(members=type(members).__name__):
                original = np.array(members, copy=True)
                result = linkage([1.0, 3.0, 2.0], method="average", members=members)
                assert_allclose(result[:, 2], [1.0, 8.0 / 3.0])
                assert_array_equal(members, original)

    def test_condensed_distances_preserve_input(self):
        for method in [
            "single",
            "complete",
            "average",
            "weighted",
            "ward",
            "centroid",
            "median",
        ]:
            for dtype in [np.float32, np.float64]:
                with self.subTest(method=method, dtype=dtype):
                    distances = np.array([1, 3, 2], dtype=dtype)
                    original = distances.copy()
                    assert_allclose(
                        linkage(distances, method=method),
                        scipy_linkage(distances, method=method),
                    )
                    assert_array_equal(distances, original)

    def test_vector_inputs_allow_copies_when_required(self):
        for method in ["single", "ward", "centroid", "median"]:
            for dtype in [np.float32, np.float64]:
                with self.subTest(method=method, dtype=dtype):
                    points = np.array([[0, 0], [1, 0], [3, 0]], dtype=dtype)
                    expected = scipy_linkage(points, method=method)
                    assert_allclose(linkage(points.tolist(), method=method), expected)
                    assert_allclose(
                        linkage_vector(points.tolist(), method=method), expected
                    )
                    assert_allclose(linkage_vector(points, method=method), expected)

    def test_non_preserving_condensed_input_allows_conversion(self):
        for distances in [[1, 3, 2], np.array([1, 3, 2], dtype=np.float32)]:
            assert_allclose(
                linkage(distances, method="average", preserve_input=False)[:, 2],
                [1, 2.5],
            )
