# alfpy
__alfpy__ is a Python translation of the Absorption Line Fitter [__alf__](https://github.com/cconroy20/alf/tree/master/src). The original Fortran code from [__alf__](https://github.com/cconroy20/alf/tree/master/src) has been almost directly translated into Python, with some improvements.

## Overview
I started this project in 2021 with the goal of getting a comprehensive understanding of how the __alf__ code works. I’m happy to say that I have achieved that goal, and now the Python version, __alfpy__, not only mirrors the function but also offers faster performance and flexibility (e.g., changing number of parameters instead of simply shrinking their priors) in many cases.

I thank Charlie Conroy for his guidance on alf since the very beginning and sharing his epertise on its every detail.  I thank Josh Speagle for invaluable discussions on parameter convergence, and for introducing me to dynesty and optimizers like differential evolution.  I thank Aliza Beverage for helpful suggestions, implementing convergence tests and an automatic check of the acceptance fraction.  The prior helper classes in `scripts/priors.py` are adapted from Prospector, and I thank the Prospector team.

## Key Features and Differences from the Original Fortran Version
- Samplers: __alfpy__ supports both emcee and dynesty samplers.
- Performance: The code has been partially accelerated using numba, and parallelized with multiprocessing.
- Dependencies: __alfpy__ requires all the models from the original __alf__ project, located under `alf/infiles/`.

## Installation and Requirements
To run alfpy, you’ll need the following Python packages (I list the version I use)
- numpy (1.26.4)
- numba (0.60.0)
- pickle
- emcee (3.1.6)
- dynesty (2.1.4)
- multiprocessing

## Usage Instructions
1. Define environment variables for the model/data resources and the Python run directory. `ALF_HOME` should point to a directory containing the original ALF `infiles/` model grids and filter files, while `ALFPY_HOME` should point to this `alfpy` repository/run directory. 
2. Edit `tofit_parameters.py` to specify the parameters you want to fit and the default values for those not being fitted
3. With `<filename>.dat` placed in `alf/indata/`, run the following command to build the model and fit it:
`python alf.py <filename> <tag>`

Each `<filename>.dat` has 5 whitespace columns — wavelength [Å], flux, error, weight (`0` masks the pixel), instrumental σ [km/s] — preceded by `#`-header line(s) giving the fit interval(s) in μm, e.g. `# 0.40 0.47`.

## Outputs
Results are written under `$ALFPY_HOME/`:
- **emcee** (`results_emcee/`): `res_emcee_<file>_<tag>.p` (chain, array `(nstep, nwalker, npar)`), `prob_emcee_*.p` (log-probability), `bestspec_*.dat` (best-fit model spectrum).
- **dynesty** (`results_dynesty/`): `res_dynesty_<file>_<tag>.p` (raw, importance-weighted results object). With `nested_post_process=True`, also `res_dynesty_*.hdf5` holding equal-weight `samples_eq`, posterior `mean`/`cov`, and `m2l` (mass-to-light).

```python
import pickle, h5py
chain = pickle.load(open("results_emcee/res_emcee_<file>_<tag>.p", "rb"))   # emcee: (nstep, nwalker, npar)
with h5py.File("results_dynesty/res_dynesty_<file>_<tag>.hdf5") as f:        # dynesty (post-processed)
    samples, m2l = f["samples_eq"][:], f["m2l"][:]
```

`post_process.alfres` wraps loading + percentiles (full-parameter fits).

## Citation
If __alfpy__ is helpful in your work, please cite this repository together with the original [__alf__](https://github.com/cconroy20/alf) papers and ASCL entry.
