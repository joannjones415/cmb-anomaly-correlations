"""Rotating and downgrading Planck maps and masks, and the anomaly statistics of the data.

Maps and masks are rotated from galactic to ecliptic coordinates, then
downgraded to nside 64 (S_1/2 and R^TT) and nside 16 (sigma^2_16) with the
beams and pixel windows of Muir et al. 2018.
"""
import os
import numpy as np
import healpy as hp
import pymaster as nmt

import stats

HERE = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.environ.get('CMB_DATA', os.path.join(HERE, 'data'))

MASK_FILE = 'COM_Mask_CMB-common-Mask-Int_2048_R3.00.fits'
MAP_FILES = {
    'commander': 'COM_CMB_IQU-commander_2048_R3.00_full.fits',
    'nilc':      'COM_CMB_IQU-nilc_2048_R3.00_full.fits',
    'sevem':     'COM_CMB_IQU-sevem_2048_R3.01_full.fits',
    'smica':     'COM_CMB_IQU-smica_2048_R3.00_full.fits',
}

S12_NSIDE = 64
SIG16_NSIDE = 16
SIM_LMAX = 200

# fwhm in radians: 5 arcmin for the planck maps, 160 and 640 arcmin after downgrading
BEAM_FWHM = {2048: 0.00145444,
             64: 0.0465421,
             16: 0.186168}


def change_coord_map(m, coord):
    """Rotate a healpix map, coord = ['G', 'E'] goes from galactic to ecliptic."""
    rot = hp.Rotator(coord=coord)
    return rot.rotate_map_alms(m, lmax=SIM_LMAX)


def change_coord_mask(inmask, coord):
    """Rotate a mask and reset pixels to 0 or 1 at the midpoint of its range."""
    rot = hp.Rotator(coord=coord)
    newmask = rot.rotate_map_alms(inmask, lmax=SIM_LMAX)
    ind = newmask >= (newmask.max() + newmask.min()) / 2
    newmask[ind] = 1
    newmask[~ind] = 0
    return newmask


def downgrade_map(inmap, new_nside):
    """Swap the input beam and pixel window for those of new_nside."""
    new_pixwin = hp.pixwin(new_nside)
    new_lmax = len(new_pixwin) - 1
    new_beam = hp.gauss_beam(BEAM_FWHM[new_nside], lmax=new_lmax)

    orig_nside = hp.npix2nside(len(inmap))
    orig_pixwin = hp.pixwin(orig_nside, lmax=new_lmax)
    orig_beam = hp.gauss_beam(BEAM_FWHM[orig_nside], lmax=new_lmax)

    orig_alm = hp.map2alm(inmap, lmax=new_lmax)
    new_alm = hp.almxfl(orig_alm, new_pixwin * new_beam / (orig_pixwin * orig_beam))
    return hp.alm2map(new_alm, new_nside)


def downgrade_mask(inmask, new_nside):
    """Downgrade a mask and reset pixels to 0 or 1 at a cutoff of 0.9."""
    new_mask = downgrade_map(inmask, new_nside)
    inds = new_mask >= 0.9
    new_mask[inds] = 1
    new_mask[~inds] = 0
    return new_mask


def prepare_mask(path=None):
    """The common mask in ecliptic coordinates at nside 64 and nside 16."""
    mask = hp.read_map(path or os.path.join(DATA_DIR, MASK_FILE))
    mask = change_coord_mask(mask, ['G', 'E'])
    return downgrade_mask(mask, S12_NSIDE), downgrade_mask(mask, SIG16_NSIDE)


def s12_workspace(mask64):
    """Namaster bins and mode-coupling workspace for the masked S_1/2 spectra."""
    bins = nmt.NmtBin.from_lmax_linear(SIM_LMAX, 1)
    field = nmt.NmtField(mask64, [mask64], lmax=SIM_LMAX, lmax_mask=SIM_LMAX)
    return bins, nmt.NmtWorkspace.from_fields(field, field, bins)


def masked_cl(map64, mask64, bins, workspace):
    """Decoupled pseudo-C_l of a masked map, starting at l = 2."""
    field = nmt.NmtField(mask64, [map64], lmax=SIM_LMAX, lmax_mask=SIM_LMAX)
    return nmt.compute_full_master(field, field, bins, workspace=workspace)[0]


def data_statistics(name, mask64, mask16, bins, workspace, path=None,
                    s12_lmax=stats.S12_LMAX, rtt_lmax=stats.RTT_LMAX):
    """[S_1/2, R^TT, sigma^2_16, S_QO, T_QO] of one Planck map, in muK units.

    S_QO and T_QO of the data come from Copi's multipole vector code and are
    left nan here, to be filled in before comparing with realizations.
    """
    mp = hp.read_map(path or os.path.join(DATA_DIR, MAP_FILES[name]))
    mp = change_coord_map(mp, ['G', 'E'])
    # planck maps are in K
    map64 = downgrade_map(mp, S12_NSIDE) * 1e6
    map16 = downgrade_map(map64, SIG16_NSIDE)

    cl_cut = masked_cl(map64, mask64, bins, workspace)
    s12 = stats.calc_s12(cl_cut, s12_lmax)
    rtt = stats.calc_rtt(hp.anafast(map64), rtt_lmax)
    sig16 = stats.calc_sigma16(map16, stats.north_pixels(mask16))
    return np.array([s12, rtt, sig16, np.nan, np.nan])


if __name__ == '__main__':
    import argparse
    p = argparse.ArgumentParser(description='Anomaly statistics of the Planck maps found in the data directory.')
    p.add_argument('--out', default=os.path.join(HERE, 'results', 'data_statistics.npz'))
    p.add_argument('--s12-lmax', type=int, default=stats.S12_LMAX)
    p.add_argument('--rtt-lmax', type=int, default=stats.RTT_LMAX)
    args = p.parse_args()

    mask64, mask16 = prepare_mask()
    bins, workspace = s12_workspace(mask64)
    found = [n for n in MAP_FILES if os.path.exists(os.path.join(DATA_DIR, MAP_FILES[n]))]
    if not found:
        raise SystemExit(f'no planck maps in {DATA_DIR}, see data/README.md')
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    np.savez(args.out, **{n: data_statistics(n, mask64, mask16, bins, workspace, s12_lmax=args.s12_lmax,
                                                  rtt_lmax=args.rtt_lmax) for n in found})
