"""
The colors are Paul Tol's muted scheme plus one mid blue. Each Planck map
keeps the same color in every figure, and greys are for context.

    import paperstyle as ps
    ps.apply()
"""
import matplotlib.pyplot as plt

INDIGO = '#332288'
GREEN = '#117733'
WINE = '#882255'
SKY = '#5877CC'

MAP_COLORS = {'commander': WINE, 'nilc': SKY, 'sevem': GREEN, 'smica': INDIGO}
MAP_LABELS = {'commander': 'Commander', 'nilc': 'NILC', 'sevem': 'SEVEM', 'smica': 'SMICA'}
MAP_MARKERS = {'commander': 'o', 'nilc': 's', 'sevem': 'D', 'smica': '^'}

GREY_FILL = '#BBBBBB'
GREY_LINE = '#555555'

# figures print at about 0.8 of a revtex column, so fonts are large for the shrink
WIDTH = 6.4
FS_LABEL, FS_TITLE, FS_TICK, FS_LEG = 15, 15, 13, 13


def apply(**overrides):
    """House rcParams, with nothing bold and mathtext so no TeX install is needed."""
    plt.rcParams.update({
        'figure.dpi': 120, 'savefig.dpi': 200,
        'font.size': FS_TICK, 'font.family': 'serif',
        'mathtext.fontset': 'cm', 'text.usetex': False,
        'axes.grid': True, 'grid.alpha': 0.18, 'grid.linewidth': 0.6,
        'axes.labelweight': 'normal', 'axes.titleweight': 'normal',
        'figure.titleweight': 'normal',
        'axes.labelsize': FS_LABEL, 'axes.titlesize': FS_TITLE,
        'axes.linewidth': 0.9, 'axes.edgecolor': '#444444',
        'xtick.direction': 'in', 'ytick.direction': 'in',
        'xtick.top': True, 'ytick.right': True,
        'xtick.minor.visible': True, 'ytick.minor.visible': True,
        'xtick.major.size': 6, 'ytick.major.size': 6,
        'xtick.minor.size': 3, 'ytick.minor.size': 3,
        'xtick.labelsize': FS_TICK, 'ytick.labelsize': FS_TICK,
        'legend.frameon': False, 'legend.fontsize': FS_LEG,
    })
    plt.rcParams.update(overrides)


def save(fig, stem):
    for ext in ('pdf', 'png'):
        fig.savefig(f'{stem}.{ext}', dpi=200, bbox_inches='tight')
