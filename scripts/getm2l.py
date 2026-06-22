import os, numpy as np
from linterp import tsum
from getmass import getmass
from alf_constants import ALF_HOME,mypi,clight,lsun,pc2cm
from sedpy import observate
from numba import jit
__all__ = ['getm2l']

ALF_HOME = os.environ['ALF_HOME']
#f15 = np.array(pd.read_csv(f'{ALF_HOME}infiles/filters.dat', 
#                           delim_whitespace=True, header=None, comment='#'))
f15 = np.loadtxt(f'{ALF_HOME}infiles/filters.dat', comments='#')

#@jit(nopython=True)
def getm2l(lam, spec, pos, 
           #alfvar, 
           nstart=100, nend=10566, nfil=3, imf_type=3, 
           mw = 0, other_filter = None):
    """
    !compute mass-to-light ratio in several filters (AB mags)
    -- INPUTS: lam, spec, pos
    -- OUTPUTS: m2l
    """

    nl = nend - nstart + 1
    
    msto_t0 = 0.33250847; msto_t1 = -0.29560944
    msto_z0 = 0.95402521; msto_z1 = 0.21944863; msto_z2 = 0.070565820
    krpa_imf1, krpa_imf2, krpa_imf3 = 1.3, 2.3, 2.3
    imflo, imfhi = 0.08, 100.0
    magsun = (4.64, 4.52, 5.14)
    
    #f15 = np.loadtxt('{0}infiles/filters.dat'.format(ALF_HOME))
    nfil_f15 = f15.shape[1]-1
    if other_filter is not None:
        filterlist = observate.load_filters(other_filter)
        nfil = nfil_f15 + len(other_filter)

    filters = np.zeros((nl, nfil))
    for fi in range(nfil_f15):
        filters[:,fi] = np.copy(f15[nstart-1:nend, fi+1])
        
    m2l = np.zeros(nfil)
    mag = np.zeros(nfil)

    #!---------------------------------------------------------------!
    #!---------------------------------------------------------------!
    #!convert to the proper units
    aspec = spec*lsun/1e6*lam**2/clight/1e8/4/mypi/pc2cm**2 
    msto = 10**(msto_t0 + msto_t1*pos.logage)*(msto_z0 + msto_z1*pos.zh + msto_z2*pos.zh**2 )
    
    if mw == 1:
        mass, imfnorm = getmass(imflo, msto, 
                                krpa_imf1, krpa_imf2, krpa_imf3)
    else:
        if imf_type ==0:
            #!single power-law IMF with a fixed lower-mass cutoff
            mass, imfnorm = getmass(imflo, msto, 
                                    pos.imf1, pos.imf1, krpa_imf3)
            
        elif imf_type ==1: 
            #!double power-law IMF with a fixed lower-mass cutoff
            mass, imfnorm = getmass(imflo, msto, 
                                    pos.imf1, pos.imf2, krpa_imf3)
            
        elif imf_type ==2:
            #!single powerlaw index with variable low-mass cutoff
            mass, imfnorm = getmass(pos.imf3, msto, 
                                    pos.imf1, pos.imf1, krpa_imf3)
            
        elif imf_type == 3:
            #!double powerlaw index with variable low-mass cutoff
            mass, imfnorm = getmass(pos.imf3, msto, pos.imf1, pos.imf2, krpa_imf3)
            
            
        elif imf_type == 4:
            #non-parametric IMF for 0.08-1.0 Msun; Salpeter slope at >1 Msun
            mass, imfnorm = getmass(imflo, msto, pos.imf1, pos.imf2, krpa_imf3, 
                                    pos.imf3, pos.imf4)
            

    if other_filter is not None:
        solar_mag_l = np.zeros(len(filterlist))
        for ji in range(len(filterlist)):
            solar_mag_l[ji] = filterlist[ji].solar_ab_mag
        # ---- need to convert from fv(which is aspec) to flambda ---- #
        fv = spec*lsun/1e6/4/mypi/pc2cm**2 
        jmags = observate.getSED(lam, fv, filterlist=filterlist)
    
    for j in range(nfil):
        if j >= nfil_f15:
            jmag = jmags[j - nfil_f15]
            solar_mag = solar_mag_l[j-nfil]
            m2l[j] = mass/10**(2./5*(solar_mag-jmag))
        else:
            jmag = tsum(lam, aspec * filters[:,j]/lam)
            if jmag <= 0:
                m2l[j] = 0.
            else:
                jmag = -2.5*np.log10(jmag)-48.60
                m2l[j] = mass/10**(2./5*(magsun[j]-jmag))
                if m2l[j] > 100:
                    m2l[j] = 0.
        mag[j] = jmag

    return m2l

