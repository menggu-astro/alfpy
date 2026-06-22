import numpy as np
from alf_constants import clight, huge_number

def maskemlines(alfvar, zred, sigma):
    """
    !routine to mask emission lines
    """
    
    neml = alfvar.neml
    wave0, wavel ,waveh = np.empty((3, neml))
    m=2

    #!central wavelengths of (potentially) strong emission lines
    #!convert to observed frame
    wave0 = alfvar.emlines * (1 + zred/clight*1e5)
    
    #!mask within +/-m sigma of the central wavelength
    wavel = wave0 - m*wave0*sigma/clight*1e5
    waveh = wave0 + m*wave0*sigma/clight*1e5
    
    #for i in range(alfvar.datmax):
    #    if alfvar.data.lam[i] >= wavel[j] and data.lam[i] <= waveh[j]:
    for j in range(alfvar.neml):
        index = np.logical_and(alfvar.data.lam >= wavel[j], 
                               alfvar.data.lam <= waveh[j])
        alfvar.data.wgt[index] = huge_number
        alfvar.data.err[index] = huge_number
    
    return alfvar
                
