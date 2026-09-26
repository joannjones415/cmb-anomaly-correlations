# Data

The code reads everything from this directory. To keep the data somewhere else, set the environment variable `CMB_DATA` to that path.

The Planck 2018 best-fit theory spectrum is included because it is small. The maps and the mask come from the Planck Legacy Archive. They total about 7 GB, so they arent here. The files are:

| File | Size | Download |
|---|---|---|
| `COM_Mask_CMB-common-Mask-Int_2048_R3.00.fits` | 0.2 GB | [IRSA](https://irsa.ipac.caltech.edu/data/Planck/release_3/ancillary-data/masks/COM_Mask_CMB-common-Mask-Int_2048_R3.00.fits) |
| `COM_CMB_IQU-commander_2048_R3.00_full.fits` | 1.6 GB | [IRSA](https://irsa.ipac.caltech.edu/data/Planck/release_3/all-sky-maps/maps/component-maps/cmb/COM_CMB_IQU-commander_2048_R3.00_full.fits) |
| `COM_CMB_IQU-nilc_2048_R3.00_full.fits` | 1.6 GB | [IRSA](https://irsa.ipac.caltech.edu/data/Planck/release_3/all-sky-maps/maps/component-maps/cmb/COM_CMB_IQU-nilc_2048_R3.00_full.fits) |
| `COM_CMB_IQU-sevem_2048_R3.01_full.fits` | 1.6 GB | [PLA](http://pla.esac.esa.int/pla/aio/product-action?MAP.MAP_ID=COM_CMB_IQU-sevem_2048_R3.01_full.fits) |
| `COM_CMB_IQU-smica_2048_R3.00_full.fits` | 2.0 GB | [IRSA](https://irsa.ipac.caltech.edu/data/Planck/release_3/all-sky-maps/maps/component-maps/cmb/COM_CMB_IQU-smica_2048_R3.00_full.fits) |
| `COM_PowerSpect_CMB-base-plikHM-TTTEEE-lowl-lowE-lensing-minimum-theory_R3.01.txt` | included | [IRSA](https://irsa.ipac.caltech.edu/data/Planck/release_3/ancillary-data/cosmoparams/COM_PowerSpect_CMB-base-plikHM-TTTEEE-lowl-lowE-lensing-minimum-theory_R3.01.txt) |

IRSA mirrors only SEVEM R3.00, so the SEVEM R3.01 map comes from the archive at ESA. You can fetch everything from this directory with:

```
I=https://irsa.ipac.caltech.edu/data/Planck/release_3
curl -LO $I/ancillary-data/masks/COM_Mask_CMB-common-Mask-Int_2048_R3.00.fits
for m in commander nilc smica; do
    curl -LO $I/all-sky-maps/maps/component-maps/cmb/COM_CMB_IQU-${m}_2048_R3.00_full.fits
done
curl -L -o COM_CMB_IQU-sevem_2048_R3.01_full.fits \
    "http://pla.esac.esa.int/pla/aio/product-action?MAP.MAP_ID=COM_CMB_IQU-sevem_2048_R3.01_full.fits"
```

Only the temperature field of each map is used. You don't need every map. `maps.py` computes the statistics of whichever maps are here.
