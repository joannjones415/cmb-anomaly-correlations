"""Tests of rotating, downgrading and masking, on toy maps with known answers."""
import numpy as np
import healpy as hp
import pytest

import maps


def band_mask(nside, deg=20):
    """1 away from the equator, 0 within deg of it."""
    theta, _ = hp.pix2ang(nside, np.arange(hp.nside2npix(nside)))
    return (np.abs(np.pi/2 - theta) > np.radians(deg)).astype(float)


def random_alm(lmax, seed=0):
    np.random.seed(seed)
    return hp.synalm(1 / (np.arange(lmax + 1) + 1.0)**2, lmax=lmax)


def test_downgrade_map_swaps_beam_and_pixel_window():
    # a map made at nside 64 and downgraded should match the same alm made directly at nside 16
    alm = random_alm(40)
    map64 = hp.alm2map(alm, 64, pixwin=True, fwhm=maps.BEAM_FWHM[64])
    direct = hp.alm2map(alm, 16, pixwin=True, fwhm=maps.BEAM_FWHM[16])
    assert np.allclose(maps.downgrade_map(map64, 16), direct, atol=1e-3 * direct.std())


def test_downgrade_mask_is_binary_at_new_nside():
    mask16 = maps.downgrade_mask(band_mask(64), 16)
    assert hp.npix2nside(len(mask16)) == 16
    assert set(np.unique(mask16)) == {0, 1}
    # the 0.9 cutoff only ever shrinks the unmasked region
    assert 0.5 < mask16.mean() < band_mask(16).mean()


def test_rotation_round_trip():
    # rotation works at lmax = 200, which needs nside >= 128, the planck maps are at 2048
    alm = random_alm(30, seed=1)
    mp = hp.alm2map(alm, 128)
    back = maps.change_coord_map(maps.change_coord_map(mp, ['G', 'E']), ['E', 'G'])
    assert np.allclose(back, mp, atol=1e-6 * mp.std())


def test_rotated_mask_is_binary():
    mask = maps.change_coord_mask(band_mask(128), ['G', 'E'])
    assert set(np.unique(mask)) == {0, 1}
    assert mask.mean() == pytest.approx(band_mask(128).mean(), abs=0.01)


def test_masked_cl_on_full_sky_matches_anafast():
    mask = np.ones(hp.nside2npix(64))
    bins, workspace = maps.s12_workspace(mask)
    mp = hp.alm2map(random_alm(maps.SIM_LMAX, seed=2), 64)
    cl = maps.masked_cl(mp, mask, bins, workspace)
    assert np.allclose(cl[:40], hp.anafast(mp, lmax=maps.SIM_LMAX)[2:42], rtol=1e-2)

