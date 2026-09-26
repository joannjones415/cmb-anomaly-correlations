"""Individual and joint p-values of the anomaly statistics in an ensemble of realizations.

A realization counts as anomalous in a statistic when it is at least as
extreme as the data in the same direction: lower S_1/2, R^TT and sigma^2_16,
higher S_QO.

    python analyze.py --sims 'results/sims/*.npy' --data results/data_statistics.npz
"""
import argparse
import glob
import os
import numpy as np

STATS = ['S12', 'RTT', 'sigma16', 'SQO']
LABELS = [r'$S_{1/2}$', r'$R^{TT}$', r'$\sigma^2_{16}$', r'$S_{QO}$']
LOW_IS_ANOMALOUS = np.array([True, True, True, False])


def load_sims(pattern):
    """Stack every (n, 5) realization file matching a glob pattern."""
    files = sorted(glob.glob(pattern))
    if not files:
        raise FileNotFoundError(f'no realization files match {pattern}')
    return np.concatenate([np.load(f) for f in files])


def anomalous(sims, data):
    """(n, 4) boolean array, true where a realization is as anomalous as the data."""
    x, d = sims[:, :4], np.asarray(data)[:4]
    return np.where(LOW_IS_ANOMALOUS, x < d, x > d)


def pvalue(flags, which):
    """Fraction of realizations anomalous in every statistic listed in which."""
    return np.mean(np.all(flags[:, list(which)], axis=1))


def correlation_factor(flags, which):
    """Joint p-value over the product of the individual p-values."""
    with np.errstate(divide='ignore', invalid='ignore'):
        return pvalue(flags, which) / np.prod([pvalue(flags, [i]) for i in which])


def _available(sims, data):
    """Statistics present in both the realizations and the data."""
    return ~np.all(np.isnan(sims[:, :4]), axis=0) & ~np.isnan(np.asarray(data, float)[:4])


def pairwise_table(sims, data):
    """4x4 table of p-values.

    The diagonal holds individual p-values, the lower triangle joint pairwise
    p-values and the upper triangle their correlation factors. Rows and columns
    of statistics missing from the realizations or the data are nan.
    """
    flags = anomalous(sims, data)
    have = _available(sims, data)
    t = np.full((4, 4), np.nan)
    for i in np.flatnonzero(have):
        t[i, i] = pvalue(flags, [i])
        for j in np.flatnonzero(have):
            if j < i:
                t[i, j] = pvalue(flags, [i, j])
                t[j, i] = correlation_factor(flags, [i, j])
    return t


def joint_summary(sims, data):
    """p-value and correlation factor of all statistics present in the realizations and the data."""
    have = np.flatnonzero(_available(sims, data))
    flags = anomalous(sims, data)
    return dict(stats=[STATS[i] for i in have], p=pvalue(flags, have),
                factor=correlation_factor(flags, have), n=len(sims))


def format_table(t):
    rows = [f"{'':>8}" + ''.join(f'{s:>10}' for s in STATS)]
    for i, s in enumerate(STATS):
        rows.append(f'{s:>8}' + ''.join(f'{v:>10.2g}' for v in t[i]))
    return '\n'.join(rows)


if __name__ == '__main__':
    here = os.path.dirname(os.path.abspath(__file__))
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--sims', default=os.path.join(here, 'results', 'sims', '*.npy'))
    p.add_argument('--data', default=os.path.join(here, 'results', 'data_statistics.npz'))
    args = p.parse_args()

    sims = load_sims(args.sims)
    data = np.load(args.data)
    for name in data.files:
        j = joint_summary(sims, data[name])
        print(f"{name}: {j['n']} realizations, joint p of {', '.join(j['stats'])} = {j['p']:.2g}, "
              f"correlation factor {j['factor']:.2g}")
        print(format_table(pairwise_table(sims, data[name])), end='\n\n')
