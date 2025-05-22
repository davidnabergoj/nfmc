import numpy as np
from nfmc.algorithms.mh.dual_averaging import DualAveraging


def test_step():
    initial_step_size = 1.0
    da = DualAveraging(initial_step_size=initial_step_size)
    da.step(0.2)

    assert np.isfinite(da.value)
    assert da.value > 0
    assert da.value != initial_step_size


def test_history():
    initial_step_size = 1.0
    da = DualAveraging(
        initial_step_size=initial_step_size,
        store_step_sizes=True,
        store_errors=True
    )

    for _ in range(10):
        da.step(0.001)

    assert len(da.step_size_history) == 10
    assert len(da.error_history) == 10

    for i in range(10):
        assert np.isfinite(da.step_size_history[i])
        assert da.step_size_history[i] > 0
        assert da.error_history[i] == 0.001
