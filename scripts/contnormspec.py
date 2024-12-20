import numpy as np
from linterp import locate
from numba import jit, njit

__all__ = ['contnormspec',]
# ------------------------------------------------------------------------- 
@jit(nopython=True, fastmath=True)
def tmp_cal(lam, il1, il2, npow=None, npolymax = 10):
    # ---- !divide by a power-law of degree npow. one degree per poly_dlam.
    # ---- !don't let things get out of hand (force Npow<=npolymax)
    poly_dlam = 100.0
    buff = 0.0
    n1 = lam.size
    if npow is None:
        # ---- use -1 to be consistent with alf ---- # 
        # ---- -> returned coeff should have the same length ---- #
        npow = max(1, min((il2-il1)//poly_dlam, npolymax) - 1)  
    i1 = min(max(locate(lam, il1-buff),0), n1-2)
    i2 = min(max(locate(lam, il2+buff),1), n1-1)   
    ml = (il1+il2)/2.0

    return i1, i2, ml, npow


# ------------------------------------------------------------------------- #
@jit(nopython=True, fastmath=True)
def polyfit_vandermonde(x, y, yerr, degree):
    """
    Fit a polynomial of the given degree to the data points (x, y)
    using a weighted Vandermonde matrix and least squares solution.
    Returns:
    - coefficients: 1D array of polynomial coefficients (from lowest to highest degree)
    """
    V = np.vander(x, degree + 1, increasing=True)
    
    weights = 1 / yerr**2
    W = np.diag(weights)
    # (V^T * W * V) * c = (V^T * W * y)
    A = V.T @ W @ V
    b = V.T @ W @ y
    coefficients = np.linalg.solve(A, b)
    
    return coefficients

# ------------------------------------------------------------------------- #
@jit(nopython=True, fastmath=True)
def evaluate_poly(x, coefficients):
    V = np.vander(x, len(coefficients), increasing=True)
    return V @ coefficients

# ------------------------------------------------------------------------- #
def contnormspec(lam, flx, err, il1, il2, coeff=False, return_poly=False, 
                 npolymax = 14, npow = None):
    """
    !routine to continuum normalize a spectrum by a high-order
    !polynomial.  The order of the polynomial is determined by
    !n=(lam_max-lam_min)/100.  Only normalized over the input
    !min/max wavelength range
    #, lam,flx,err,il1,il2,flxout,coeff=None
    """
    i1, i2, ml, npow = tmp_cal(lam, il1, il2, npow, npolymax)
    lam_sub = lam[i1:i2]
    flx_sub = flx[i1:i2]
    err_sub = err[i1:i2]

    #!simple linear least squares polynomial fit
    valid = np.isfinite(flx_sub) & np.isfinite(err_sub)
    lam_valid = lam_sub[valid] - ml  # Centering around ml
    flx_valid = flx_sub[valid]
    err_valid = err_sub[valid]
    #res = np.polyfit(x = lam[i1:i2][ind]-ml, 
    #                 y = flx[i1:i2][ind], 
    #                 deg = npow, full = True, 
    #                 w = 1./(err[i1:i2][ind]**2), 
    #                 cov = True)
    #covar = res[2]
    #chi2sqr = res[1]
    #tcoeff = res[0]
    #p = np.poly1d(tcoeff)
    #poly = p(lam-ml)

    tcoeff = polyfit_vandermonde(lam_valid, flx_valid, err_valid, int(npow)+1)
    poly = evaluate_poly(lam - ml, tcoeff)
    
    if coeff == False and return_poly==False:
        return npow
    
    elif coeff == True and return_poly==False:
        return npow, tcoeff    
    
    elif coeff == True and return_poly==True:
        return npow, tcoeff, poly

