# ==================================== #
# - edit this file to specify parameters 
#   included in the fitting
# ==================================== #
# ======== parameters to fit ========= #
# - (included or not, default value) - #
tofit_params = {
    # -------------------------------- #
    'velz':  (True, 0.0), #0
    'sigma': (True, 11.0), 
    'logage': (True, 1.0), 
    'zh': (True, 0.0), 
    # -------------------------------- #
    'feh': (False, 0.0), #4
    'ah': (False, 0.0), 
    'ch': (False, 0.0), 
    'nh': (False, 0.0), 
    'nah': (False, 0.0), 
    'mgh': (False, 0.0), 
    'sih': (False, 0.0),
    'kh': (False, 0.0),
    'cah': (False, 0.0),
    'tih': (False, 0.0), 
    # -------------------------------- #
    'vh': (False, 0.0), #14
    'crh': (False, 0.0),
    'mnh': (False, 0.0),
    'coh': (False, 0.0),
    'nih': (False, 0.0),
    'cuh': (False, 0.0),
    'srh': (False, 0.0),
    'bah': (False, 0.0),
    'euh': (False, 0.0),
    # -------------------------------- #
    'teff': (False, 0.0), #23, # killed off in alf.f90
    'imf1': (False, 2.3), 
    'imf2': (False, 2.3), 
    'logfy': (False, -5.9), 
    'sigma2': (False, 10.1),
    'velz2': (False, 0.0),
    'logm7g': (False, -6.+1e-5), # killed off in alf.f90
    'hotteff': (False, 8.0+1e-5),
    'loghot': (False, -6.0+1e-5),
    'fy_logage': (False, 0.3),
    # -------------------------------- #
    'logemline_h': (False, -6.0+1e-5), #33
    'logemline_oii': (False, -6.0+1e-5),
    'logemline_oiii': (False, -6.0+1e-5),
    'logemline_sii': (False, -6.0+1e-5),
    'logemline_ni': (False, -6.0+1e-5),
    'logemline_nii': (False, -6.0+1e-5), 
    # -------------------------------- #
    'logtrans': (False, -5.9), #39 
    'jitter': (True, 1.0), 
    'logsky': (False, -5.9), 
    'imf3': (False, 0.08+1e-5), 
    'imf4': (False, 0.0), 
    'h3': (False, 0.0),
    'h4': (False, 0.0),
}
