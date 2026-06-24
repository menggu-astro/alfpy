from alf_vars import ALFVAR
from linterp import locate
from getmodel import getmodel
from setup import setup
import os, numpy as np
from alf_constants import *
from str2arr import alfobj


__all__ = ['write_a_model']

def write_a_model():
    """
    !write a model to file
    """
    ALFPY_HOME = os.environ['ALFPY_HOME']
    alfvar = ALFVAR()
    alfvar.imf_type = 1
    alfvar.fit_type = 0

    alfvar = setup(alfvar, onlybasic = True)
    nl = alfvar.nl
    pos = alfobj()

    #instrumental resolution (<10 -> no broadening)
    ires     = 1.0 #!100.

    #!initialize the random number generator
    #np.random.seed()

    lmin = 3700.0
    lmax = 11000.0
    alfvar.l1 = [3700, 6700]
    alfvar.l2 = [6700, 11000]
    datmax = int(lmax-lmin)
    lam = alfvar.sspgrid.lam


    # ---- !define the log wavelength grid used in velbroad.f90
    alfvar.nl_fit = min(max(locate(lam,lmax+500.0),0),alfvar.nl-1)
    alfvar.dlstep = (np.log(alfvar.sspgrid.lam[alfvar.nl_fit])-np.log(alfvar.sspgrid.lam[0]))/alfvar.nl_fit
    alfvar.lnlam = np.arange(alfvar.nl_fit)*alfvar.dlstep + np.log(alfvar.sspgrid.lam[0])
    alfvar.l1[0] = lmin
    alfvar.l2[1] = lmax


    # ---- !loop to generate multiple mock datasets
    for j in range(1):
        # ---- !string for indexing of filenames
        file = 'alfpy_example.dat'
        pos.sigma  = 200.
        s2n  = 100.
        pos.logage = np.log10(10.)
        pos.zh     = 0.0
        pos.feh     = 0.0
        pos.ah     = 0.0
        pos.mgh     = 0.0
        pos.kh   = 0.0
        emnorm     = -5.5

        pos.imf1   = 2.30
        pos.imf2   = 2.30
        pos.imf3   = 0.08

        pos.logemline_h    = emnorm
        pos.logemline_oii  = emnorm
        pos.logemline_oiii = emnorm
        pos.logemline_sii  = emnorm
        pos.logemline_ni   = emnorm
        pos.logemline_nii  = emnorm
        pos.velz2 = 0.
        pos.sigm2 = 0.
        pos.loghot = -8.0
        pos.hotteff = 8.0
        pos.logfy = -4.0
        pos.fy_logage = 0.3000
        pos.logtrans = -4.0
        pos.logm7g = -4.0
        pos.logsky = -4.00
        print(pos.__dict__)

        # ---- !get a model spectrum
        mspec = getmodel(pos, alfvar = alfvar)
        s2np_arr = np.ones(lam.shape)*s2n*np.sqrt(0.9)
        s2np_arr[lam>=7500] = s2n*np.sqrt(2.5)
        err = mspec/s2np_arr
        gspec = np.random.normal(mspec, err, nl)

        #!write model spectrum to file
        header = '0.400 0.470\n0.470 0.560'
        idx = np.logical_and(lam >= lmin, lam<= lmax)
        print("saving to")
        np.savetxt("{0}models/{1}".format(ALF_HOME, file),
                   np.transpose([lam[idx], mspec[idx], err[idx],
                                 np.ones(lam.shape)[idx], np.ones(idx.sum())*ires]),
                   delimiter="     ", fmt='   %12.4f   %12.4E   %12.4E   %12.4f   %12.4f' ,
                   header = header)

    return pos

if __name__ == "__main__":
    write_a_model()
