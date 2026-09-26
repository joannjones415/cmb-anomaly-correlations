"""Tests of the p-value bookkeeping on synthetic ensembles with known answers."""
import numpy as np
import pytest

import analyze


def test_independent_statistics_have_correlation_factor_one():
    rng = np.random.default_rng(0)
    sims = rng.uniform(size=(400_000, 5))
    # anomalous means below 0.1 for the first three and above 0.9 for S_QO
    data = np.array([0.1, 0.1, 0.1, 0.9, 0.5])
    t = analyze.pairwise_table(sims, data)
    assert np.allclose(np.diag(t), 0.1, atol=3e-3)
    assert np.allclose(t[np.tril_indices(4, -1)], 0.01, atol=1e-3)
    assert np.allclose(t[np.triu_indices(4, 1)], 1, atol=0.1)
    j = analyze.joint_summary(sims, data)
    assert j['p'] == pytest.approx(1e-4, abs=5e-5)


def test_identical_statistics_have_joint_equal_to_individual():
    x = np.random.default_rng(1).uniform(size=100_000)
    sims = np.stack([x, x, x, 1 - x, x], axis=1)
    flags = analyze.anomalous(sims, [0.05, 0.05, 0.05, 0.95, 0])
    p = analyze.pvalue(flags, [0])
    assert analyze.pvalue(flags, [0, 1, 2, 3]) == p
    assert analyze.correlation_factor(flags, [0, 1]) == pytest.approx(1 / p)


def test_direction_and_strict_inequality():
    sims = np.array([[1, 1, 1, 1, 0], [2, 2, 2, 2, 0], [3, 3, 3, 3, 0.]])
    flags = analyze.anomalous(sims, [2, 2, 2, 2])
    assert flags[:, :3].tolist() == [[True]*3, [False]*3, [False]*3]
    assert flags[:, 3].tolist() == [False, False, True]


def test_missing_sqo_is_nan_in_table():
    sims = np.random.default_rng(2).uniform(size=(1000, 5))
    sims[:, 3:] = np.nan
    t = analyze.pairwise_table(sims, [0.5]*5)
    assert np.all(np.isnan(t[3])) and np.all(np.isnan(t[:, 3]))
    assert not np.any(np.isnan(t[:3, :3]))
    assert analyze.joint_summary(sims, [0.5]*5)['stats'] == ['S12', 'RTT', 'sigma16']


def test_missing_data_value_is_skipped():
    sims = np.random.default_rng(3).uniform(size=(1000, 5))
    data = [0.5, 0.5, 0.5, np.nan, np.nan]
    assert np.all(np.isnan(analyze.pairwise_table(sims, data)[3]))
    assert analyze.joint_summary(sims, data)['stats'] == ['S12', 'RTT', 'sigma16']


def test_load_sims_stacks_files(tmp_path):
    for k in range(3):
        np.save(tmp_path / f'sims_{k:04d}.npy', np.full((4, 5), k))
    sims = analyze.load_sims(str(tmp_path / '*.npy'))
    assert sims.shape == (12, 5)
    with pytest.raises(FileNotFoundError):
        analyze.load_sims(str(tmp_path / 'none_*.npy'))
