"""The four large-angle anomaly statistics: S_1/2, R^TT, sigma^2_16 and S_QO.

S_QO and T_QO need Copi's multipole vector code, mpd_decomp.py, on the python
path. Without it the other three statistics still work.
"""
from functools import lru_cache

import numpy as np
import healpy as hp
from scipy.special import eval_legendre

try:
    import mpd_decomp as mpd
except ImportError:
    mpd = None

# defaults, every function that uses them takes lmax as an argument
S12_LMAX = 40
RTT_LMAX = 27


def imatrix(lmax, x=0.5):
    """The S_1/2 kernel I_ll' for 0 <= l, l' <= lmax.

    The (2l+1)(2l'+1)/(4pi)^2 factor is absorbed, so S_1/2 = C I C. The
    recursions are the ones worked out in ImnMatrix.ipynb.
    """
    n = np.arange(lmax + 3)
    P = eval_legendre(np.arange(lmax + 4), x)
    I = np.zeros((lmax + 3, lmax + 3))
    I[0, :] = I[:, 0] = (P[n + 1] - P[n - 1]) / (2*n + 1)

    # off-diagonal closed form, the diagonal it produces is overwritten below
    m, k = n[np.newaxis, 1:], n[1:, np.newaxis]
    with np.errstate(divide='ignore', invalid='ignore'):
        I[1:, 1:] = (m*P[k]*(P[m-1] - x*P[m]) - k*P[m]*(P[k-1] - x*P[k])) / (k*(k+1) - m*(m+1))

    I[0, 0] = 1 + x
    I[1, 1] = (x**3 + 1) / 3
    for l in range(2, lmax + 1):
        I[l, l] = ((P[l+1] - P[l-1])*(P[l] - P[l-2]) - (2*l - 1)*I[l+1, l-1]
                   + (2*l + 1)*I[l, l-2] + (2*l - 1)*I[l-1, l-1]) / (2*l + 1)

    l = np.arange(lmax + 1)
    return I[:lmax+1, :lmax+1] * np.outer(2*l + 1, 2*l + 1) / (4*np.pi)**2


@lru_cache
def s12_kernel(lmax):
    """The kernel for 2 <= l <= lmax, matching namaster bandpowers that start at l = 2."""
    return imatrix(lmax)[2:, 2:]


def calc_s12(cl, lmax=S12_LMAX):
    """S_1/2 = C I C summed over 2 <= l <= lmax, where cl starts at l = 2."""
    cl = cl[..., :lmax - 1]
    return cl @ s12_kernel(lmax) @ cl


def calc_rtt(cl, lmax=RTT_LMAX):
    """R^TT, the mean even D_l over the mean odd D_l for 2 <= l <= lmax.

    For odd lmax this is D_+/D_- of Kim and Naselsky 2010, and the means keep
    even lmax normalized too. cl starts at l = 0 and can be a stack of spectra
    along the first axis.
    """
    ell = np.arange(lmax + 1)
    dl = ell*(ell + 1)/(2*np.pi) * cl[..., :lmax+1]
    return np.mean(dl[..., 2::2], axis=-1) / np.mean(dl[..., 3::2], axis=-1)


def north_pixels(mask16):
    """Unmasked pixels of the northern ecliptic hemisphere in a ring-ordered mask."""
    npix = len(mask16)
    return mask16.astype(bool) & (np.arange(npix) <= npix//2)


def calc_sigma16(map16, north):
    """The pixel variance of an nside 16 map over the given pixels."""
    return np.var(map16[north], ddof=1)


def initialize_mpv_calculator(l, lmax):
    """Returns a function that takes healpy alm and gives the l multipole vectors."""
    if mpd is None:
        raise ImportError('S_QO needs mpd_decomp.py from Copi, see the README')
    almvec = np.zeros(2*l + 1)
    idxm0 = hp.Alm.getidx(lmax, l, 0)
    idxm = hp.Alm.getidx(lmax, l, np.arange(1, l+1))
    def calculate_multipole_vectors(alm):
        almvec[0] = alm[idxm0].real
        almvec[1:2*l:2] = alm[idxm].real
        almvec[2:2*l+1:2] = alm[idxm].imag
        (vec, A) = mpd.mpd_decomp_full_fit(almvec)
        return vec
    return calculate_multipole_vectors


def cross2(v):
    return np.cross(v[0], v[1])


def cross3(v):
    return np.array([
        np.cross(v[0], v[1]),
        np.cross(v[1], v[2]),
        np.cross(v[2], v[0])
    ])


def calc_sqo_tqo(v2, v3):
    """S_QO and T_QO from the quadrupole and octopole multipole vectors."""
    dp = np.abs(np.sum(cross2(v2) * cross3(v3), axis=-1))
    return np.sum(dp) / 3, 1 - np.sum((1 - dp)**2) / 3
