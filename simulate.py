"""LCDM realizations of the anomaly statistics.

Each realization draws alm from the Planck 2018 best-fit spectrum up to
l = 200, makes nside 64 and nside 16 maps in ecliptic coordinates with the
matching beam and pixel window, and records the same five numbers as
maps.data_statistics. With no mpd_decomp.py on the path, the S_QO and T_QO
columns are nan.

    python simulate.py --nsim 1000 --seed 0 --out results/sims/sims_0000.npy
"""
import argparse
import os
import numpy as np
import healpy as hp

import maps
import stats

THEORY_FILE = os.path.join(maps.DATA_DIR,
                           'COM_PowerSpect_CMB-base-plikHM-TTTEEE-lowl-lowE-lensing-minimum-theory_R3.01.txt')
COLUMNS = ['S12', 'RTT', 'sigma16', 'SQO', 'TQO']


def theory_cl(path=THEORY_FILE, lmax=maps.SIM_LMAX):
    """C_TT in muK^2 indexed by l, zero for l < 2."""
    ell, dl = np.loadtxt(path, usecols=(0, 1), unpack=True)
    ell = ell.astype(int)
    keep = (ell >= 2) & (ell <= lmax)
    cl = np.zeros(lmax + 1)
    cl[ell[keep]] = dl[keep] * 2*np.pi / (ell[keep]*(ell[keep] + 1))
    return cl


def setup(masks=None, s12_lmax=stats.S12_LMAX, rtt_lmax=stats.RTT_LMAX):
    """Everything for the loop that does not change between realizations.

    masks is (mask64, mask16), and defaults to the Planck common mask.
    """
    mask64, mask16 = masks or maps.prepare_mask()
    bins, workspace = maps.s12_workspace(mask64)
    return dict(cl=theory_cl(), mask64=mask64, bins=bins, workspace=workspace,
                north=stats.north_pixels(mask16), s12_lmax=s12_lmax, rtt_lmax=rtt_lmax)


def realization_stats(alms, s):
    """The five statistics of one realization."""
    out = np.full(5, np.nan)
    mp = hp.alm2map(alms, nside=maps.S12_NSIDE, pixwin=True, fwhm=maps.BEAM_FWHM[maps.S12_NSIDE])
    out[1] = stats.calc_rtt(hp.anafast(mp), s['rtt_lmax'])
    cl = maps.masked_cl(mp, s['mask64'], s['bins'], s['workspace'])
    out[0] = stats.calc_s12(cl, s['s12_lmax'])
    mp = hp.alm2map(alms, nside=maps.SIG16_NSIDE, pixwin=True, fwhm=maps.BEAM_FWHM[maps.SIG16_NSIDE])
    out[2] = stats.calc_sigma16(mp, s['north'])
    if 'mpv2' in s:
        out[3:] = stats.calc_sqo_tqo(s['mpv2'](alms), s['mpv3'](alms))
    return out


def run(nsim, seed, s=None):
    """An (nsim, 5) array of statistics with columns COLUMNS."""
    s = s or setup()
    if stats.mpd is not None:
        s['mpv2'] = stats.initialize_mpv_calculator(2, maps.SIM_LMAX)
        s['mpv3'] = stats.initialize_mpv_calculator(3, maps.SIM_LMAX)
    # healpy and mpd_decomp both draw from the global numpy generator
    np.random.seed(seed)
    out = np.empty((nsim, 5))
    for i in range(nsim):
        out[i] = realization_stats(hp.synalm(s['cl'], lmax=maps.SIM_LMAX), s)
    return out


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--nsim', type=int, default=1000, help='number of realizations (default 1000)')
    p.add_argument('--seed', type=int, default=0, help='random seed, use a different one for every file')
    p.add_argument('--out', default=None, help='output .npy (default results/sims/sims_<seed>.npy)')
    p.add_argument('--s12-lmax', type=int, default=stats.S12_LMAX)
    p.add_argument('--rtt-lmax', type=int, default=stats.RTT_LMAX)
    args = p.parse_args()

    out = args.out or os.path.join(maps.HERE, 'results', 'sims', f'sims_{args.seed:04d}.npy')
    os.makedirs(os.path.dirname(out), exist_ok=True)
    np.save(out, run(args.nsim, args.seed, setup(s12_lmax=args.s12_lmax, rtt_lmax=args.rtt_lmax)))
