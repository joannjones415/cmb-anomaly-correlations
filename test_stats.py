"""Tests of the statistics on inputs with known answers."""
import numpy as np
import pytest
from scipy.integrate import quad
from scipy.special import eval_legendre
from scipy.spatial.transform import Rotation

import stats


def test_imatrix_matches_mathematica():
    # reference values from ImnMatrix.ipynb, checked there against mathematica
    I = stats.imatrix(300)
    assert np.allclose(I, I.T, atol=1e-14)
    assert I[20, 20] == pytest.approx(0.3481420231334969, abs=1e-10)
    assert I[299, 299] == pytest.approx(5.059627837950241, abs=1e-10)
    assert I[29, 287] == pytest.approx(0.001159280339306247, abs=1e-10)
    assert I[292, 21] == pytest.approx(-0.003058238391121596, abs=1e-10)


@pytest.mark.parametrize('l1, l2', [(0, 0), (0, 7), (2, 2), (3, 17), (40, 40), (12, 39)])
def test_imatrix_matches_quadrature(l1, l2):
    direct = quad(lambda x: eval_legendre(l1, x) * eval_legendre(l2, x), -1, 0.5, limit=200)[0]
    direct *= (2*l1 + 1) * (2*l2 + 1) / (4*np.pi)**2
    assert stats.imatrix(40)[l1, l2] == pytest.approx(direct, abs=1e-12)


def test_s12_is_integral_of_squared_correlation_function():
    rng = np.random.default_rng(0)
    ell = np.arange(2, stats.S12_LMAX + 1)
    cl = rng.uniform(0.5, 2, ell.size) * 1e3 / ell**2
    def ctheta(x):
        return np.sum((2*ell + 1) / (4*np.pi) * cl * eval_legendre(ell, x))
    direct = quad(lambda x: ctheta(x)**2, -1, 0.5, limit=400)[0]
    assert stats.calc_s12(cl) == pytest.approx(direct, rel=1e-8)


def test_s12_lmax_truncates_the_sum():
    cl = np.ones(60)
    assert stats.calc_s12(cl, lmax=20) == pytest.approx(cl[:19] @ stats.imatrix(20)[2:, 2:] @ cl[:19])


def test_rtt_is_one_for_flat_dl():
    ell = np.arange(60)
    cl = np.zeros(60)
    cl[1:] = 2*np.pi / (ell[1:]*(ell[1:] + 1))
    assert stats.calc_rtt(cl) == pytest.approx(1)


@pytest.mark.parametrize('lmax', [21, 27, 33])
def test_rtt_matches_dplus_over_dminus_for_odd_lmax(lmax):
    rng = np.random.default_rng(1)
    cl = rng.uniform(1, 2, (3, 40))
    ell = np.arange(2, lmax + 1)
    dl = ell*(ell + 1)/(2*np.pi) * cl[:, 2:lmax+1]
    dplus = 2/(lmax - 1) * np.sum(dl * (1 + (-1)**ell)/2, axis=1)
    dminus = 2/(lmax - 1) * np.sum(dl * (1 - (-1)**ell)/2, axis=1)
    assert np.allclose(stats.calc_rtt(cl, lmax), dplus / dminus)


def test_rtt_even_lmax_compares_means():
    rng = np.random.default_rng(4)
    cl = rng.uniform(1, 2, 40)
    ell = np.arange(2, 25)
    dl = ell*(ell + 1)/(2*np.pi) * cl[2:25]
    assert stats.calc_rtt(cl, 24) == pytest.approx(dl[ell % 2 == 0].mean() / dl[ell % 2 == 1].mean())


def test_rtt_ignores_monopole_and_dipole():
    cl = np.ones(40)
    moved = cl.copy()
    moved[:2] = 1e6
    assert stats.calc_rtt(moved) == stats.calc_rtt(cl)


def test_north_pixels_and_variance():
    nside = 16
    npix = 12 * nside**2
    north = stats.north_pixels(np.ones(npix))
    assert north.sum() == npix//2 + 1
    mp = np.where(north, 3.0, 0.0)
    mp[:10] = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    assert stats.calc_sigma16(mp, north) == pytest.approx(np.var(mp[north], ddof=1))
    assert stats.calc_sigma16(np.full(npix, 7.0), north) == 0


def test_sqo_known_configuration():
    x, y = np.eye(3)[:2]
    v2 = np.array([x, y])
    v3 = np.array([x, y, (x + y)/np.sqrt(2)])
    s, t = stats.calc_sqo_tqo(v2, v3)
    dp = np.array([1, 1/np.sqrt(2), 1/np.sqrt(2)])
    assert s == pytest.approx(dp.sum()/3)
    assert t == pytest.approx(1 - np.sum((1 - dp)**2)/3)


def test_sqo_invariant_under_rotation_and_sign():
    rng = np.random.default_rng(2)
    v2 = rng.normal(size=(2, 3))
    v3 = rng.normal(size=(3, 3))
    v2 /= np.linalg.norm(v2, axis=1)[:, None]
    v3 /= np.linalg.norm(v3, axis=1)[:, None]
    R = Rotation.random(random_state=3).as_matrix()
    flip = np.array([1, -1, 1])[:, None]
    assert np.allclose(stats.calc_sqo_tqo(v2, v3), stats.calc_sqo_tqo(v2 @ R.T, flip * v3 @ R.T))


@pytest.mark.skipif(stats.mpd is None, reason='mpd_decomp.py is not on the path')
def test_multipole_vectors_of_an_axisymmetric_quadrupole():
    import healpy as hp
    lmax = 10
    alm = np.zeros(hp.Alm.getsize(lmax), dtype=complex)
    alm[hp.Alm.getidx(lmax, 2, 0)] = 1
    v = stats.initialize_mpv_calculator(2, lmax)(alm)
    # a pure Y_20 has both multipole vectors along z
    assert np.allclose(np.abs(v[:, 2]), 1)
