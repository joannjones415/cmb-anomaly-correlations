"""Tests of the realization loop on a toy mask."""
import numpy as np
import healpy as hp
import pytest

import simulate
from test_maps import band_mask


@pytest.fixture(scope='module')
def toy_setup():
    return simulate.setup((band_mask(64), band_mask(16)))


def test_theory_cl_indexes_by_ell(tmp_path):
    cl = simulate.theory_cl()
    assert cl[:2].tolist() == [0, 0]
    # first row of the planck file is l = 2, D_l = 1016.73 muK^2
    assert cl[2] == pytest.approx(1016.73 * 2*np.pi / 6)
    # a copy with zero rows added for l = 0, 1 gives the same spectrum
    rows = open(simulate.THEORY_FILE).read().splitlines()
    padded = tmp_path / 'padded.txt'
    padded.write_text('\n'.join([rows[0], '0 0 0 0 0 0', '1 0 0 0 0 0'] + rows[1:]))
    assert np.array_equal(simulate.theory_cl(padded), cl)


def test_synthesized_spectra_average_to_theory():
    cl = simulate.theory_cl()
    np.random.seed(0)
    n = 400
    mean = np.mean([hp.alm2cl(hp.synalm(cl, lmax=60)) for _ in range(n)], axis=0)
    ell = np.arange(2, 61)
    # cosmic variance of the mean is sqrt(2/((2l+1) n)) of C_l
    assert np.all(np.abs(mean[ell] / cl[ell] - 1) < 4 * np.sqrt(2 / ((2*ell + 1) * n)))


def test_run_is_reproducible(toy_setup):
    a = simulate.run(3, seed=5, s=toy_setup)
    b = simulate.run(3, seed=5, s=toy_setup)
    assert a.shape == (3, 5)
    assert np.array_equal(a, b, equal_nan=True)
    assert not np.array_equal(a, simulate.run(3, seed=6, s=toy_setup), equal_nan=True)


@pytest.mark.slow
def test_realization_statistics_are_sensible(toy_setup):
    out = simulate.run(300, seed=7, s=toy_setup)
    assert np.all(out[:, :3] > 0)
    # parity is not preferred in lcdm
    assert np.median(out[:, 1]) == pytest.approx(1, abs=0.05)
    if simulate.stats.mpd is not None:
        assert np.all((out[:, 3] >= 0) & (out[:, 3] <= 1))
