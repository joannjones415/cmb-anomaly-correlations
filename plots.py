"""Figures comparing the realizations to the Planck maps."""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from matplotlib.lines import Line2D

import paperstyle as ps
from analyze import LABELS

# S_1/2 spans decades, so it is shown in log
LOG = [True, False, False, False]


def _values(x, i):
    return np.log10(x) if LOG[i] else x


def _label(i):
    return rf'$\log_{{10}}$ {LABELS[i]}' if LOG[i] else LABELS[i]


def _present(sims):
    return [i for i in range(4) if not np.all(np.isnan(sims[:, i]))]


def histograms(sims, data, stem=None):
    """One panel per statistic: the realizations with a line for each map."""
    ps.apply()
    idx = _present(sims)
    fig, axes = plt.subplots(1, len(idx), figsize=(ps.WIDTH * len(idx) / 2, ps.WIDTH / 2 * 1.1),
                             constrained_layout=True)
    for ax, i in zip(np.atleast_1d(axes), idx):
        ax.hist(_values(sims[:, i], i), bins=60, density=True, color=ps.GREY_FILL)
        for name, d in data.items():
            ax.axvline(_values(d[i], i), color=ps.MAP_COLORS[name], lw=1.6)
        ax.set_xlabel(_label(i))
        ax.set_yticklabels([])
    handles = [Line2D([], [], color=ps.MAP_COLORS[n], lw=1.6, label=ps.MAP_LABELS[n]) for n in data]
    fig.legend(handles=handles, loc='lower center', ncol=len(handles), bbox_to_anchor=(0.5, 1.0))
    if stem:
        ps.save(fig, stem)
    return fig


def pairs(sims, data, stem=None):
    """Lower-triangle grid of the realizations in each pair of statistics, with the maps marked."""
    ps.apply()
    idx = _present(sims)
    n = len(idx)
    fig, axes = plt.subplots(n, n, figsize=(ps.WIDTH * 1.3, ps.WIDTH * 1.3), sharex='col',
                             constrained_layout=True)
    for r, i in enumerate(idx):
        for c, j in enumerate(idx):
            ax = axes[r, c]
            if c > r:
                ax.axis('off')
                continue
            if r == c:
                ax.hist(_values(sims[:, i], i), bins=50, density=True, color=ps.GREY_FILL)
                ax.set_yticklabels([])
                for name, d in data.items():
                    ax.axvline(_values(d[i], i), color=ps.MAP_COLORS[name], lw=1.2)
            else:
                ax.hist2d(_values(sims[:, j], j), _values(sims[:, i], i), bins=50,
                          cmap='Greys', norm=LogNorm(), rasterized=True)
                for name, d in data.items():
                    ax.plot(_values(d[j], j), _values(d[i], i), ps.MAP_MARKERS[name],
                            color=ps.MAP_COLORS[name], ms=6, mec='white', mew=0.6)
            if r == n - 1:
                ax.set_xlabel(_label(j))
            else:
                ax.tick_params(labelbottom=False)
            if c == 0 and r > 0:
                ax.set_ylabel(_label(i))
            elif c > 0:
                ax.set_yticklabels([])
    handles = [Line2D([], [], color=ps.MAP_COLORS[m], marker=ps.MAP_MARKERS[m], ls='', ms=7,
                      label=ps.MAP_LABELS[m]) for m in data]
    axes[0, n - 1].legend(handles=handles, loc='upper right')
    if stem:
        ps.save(fig, stem)
    return fig
