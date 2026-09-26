# cmb anomaly correlations code!

this repository contains instructions and example code to compute four large-angle anomaly statistics on the Planck 2018 temperature maps. it measures how often LCDM reproduces them, one at a time and jointly. this code was created for the analysis presented in https://ui.adsabs.harvard.edu/abs/2023arXiv231012859J

 the statistics are:
- $S_{1/2}$, the lack of large-angle correlations;
- $R^{TT}$, the parity asymmetry;
- $\sigma^2_{16}$, the low variance of the northern ecliptic sky;
- $S_{QO}$, the alignment of the quadrupole and octopole.

the statistics are measured on the Commander, NILC, SEVEM and SMICA maps with the common mask, in ecliptic coordinates. the same statistics are computed on realizations of the Planck best-fit LCDM spectrum. 

## whats here

```
stats.py          the four statistics and the S_1/2 kernel matrix
maps.py           rotating, downgrading and masking, and the statistics of the Planck maps
simulate.py       LCDM realizations of the statistics
analyze.py        individual and joint p-values and correlation factors
plots.py          figures
cluster/          slurm script 
notebooks/        walkthrough.ipynb
data/             where the Planck files go, see data/README.md
test_*.py         tests
```

## setup

```
conda env create -f environment.yml
conda activate cmb-anomalies
```

Download the maps and the mask as described in `data/README.md`. The theory spectrum is already there.

## multipole vectors

$S_{QO}$ needs the multipole vector code, `mpd_decomp.py`, which is not included (yet). With a copy in the repo root or on the python path, `simulate.py` also records $S_{QO}$ and $T_{QO}$. Without it, those columns are nan and everything else runs. `maps.py` leaves $S_{QO}$ and $T_{QO}$ of the data as nan. Under numpy 2, version 1.20 of `mpd_decomp.py` needs `_Cp1(L,0)` changed to `_Cp1(L,0)[0]` on line 70.

## running it

The notebook `notebooks/walkthrough.ipynb` goes through every step. It computes the data statistics and runs a set of realizations. It then prints the $p$-value tables and makes the figures.

$S_{1/2}$ is summed up to $\ell_{\max} = 40$ and $R^{TT}$ up to $\ell_{\max} = 27$ by default. Both `maps.py` and `simulate.py` take `--s12-lmax` and `--rtt-lmax`. Use the same values for the data and the realizations.

Each realization takes about 15 ms. `analyze.py` stacks every file in `results/sims/`, so more runs with new seeds add to the ensemble. `cluster/run_sims.sbatch` makes $10^8$ realizations in 100 slurm array tasks of $10^6$ each:

```
mkdir -p cluster/logs && sbatch cluster/run_sims.sbatch
```

## tests

```
pytest -m "not slow"     #not slow... a few seconds. 
pytest
```

The tests check each statistic against its definition on inputs with known answers. They check that downgrading and rotating maps preserve what they should, and that the realizations are reproducible and average to the theory spectrum. The multipole vector test runs only when `mpd_decomp.py` is present. None of the tests need the Planck maps.


This code was written entirely by hand originally,  then cleaned and compacted for Github by Claude Opus 5.5. If you find any errors or want the original code, please contact joannjones@uchicago.edu

:)


