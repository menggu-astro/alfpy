import os, math, numpy as np, pandas as pd
import multiprocessing
from functools import partial
from multiprocessing import Pool
from schwimmbad import MultiPool
from dynesty import utils as dyfunc
from getm2l import *
from getmodel import *
from str2arr import *
import time
from str2arr import key_list


__all__ = ['calm2l_dynesty']
# ---------------------------------------------------------------- #
def worker_m2l(alfvar, use_keys, inarr):
    tem_posarr = fill_param(inarr, usekeys = use_keys)
    tem_pos = str2arr(2, inarr = tem_posarr)
    
    tem_mspec = getmodel(tem_pos, alfvar=alfvar)
    tem_mspec_mw = getmodel(tem_pos, alfvar=alfvar, mw=1)
    # ---- turn off various parameters for computing M/L
    tem_pos.logemline_h    = -8.0
    tem_pos.logemline_oii  = -8.0
    tem_pos.logemline_oiii = -8.0
    tem_pos.logemline_nii  = -8.0
    tem_pos.logemline_sii  = -8.0
    tem_pos.logemline_ni   = -8.0
    tem_pos.logtrans       = -8.0
    
    m2l = getm2l(alfvar.sspgrid.lam, tem_mspec, tem_pos, alfvar=alfvar)
    m2lmw = getm2l(alfvar.sspgrid.lam, tem_mspec_mw, tem_pos, mw=1, alfvar=alfvar)
    return np.append(m2l, m2lmw)


# ---------------------------------------------------------------- #
def calm2l_dynesty(in_res, alfvar, use_keys, outname):
    ALFPY_HOME = os.environ['ALFPY_HOME']
    
    samples, weights = in_res.samples, np.exp(in_res.logwt - in_res.logz[-1])
    mean, cov = dyfunc.mean_and_cov(samples, weights)
    samples = dyfunc.resample_equal(in_res.samples, weights)
    tstart = time.time()
    
    with MultiPool() as pool:
        pwork = partial(worker_m2l, alfvar, use_keys)
        print('post_process.py, dynesty, using {} processes'.format(pool.size))
        ml_res = pool.map(pwork, [samples[i] for i in range(100)])
        
    ndur = time.time() - tstart
    print('\npost processing dynesty results: {:.2f}minutes'.format(ndur/60.))
    np.savez('{0}results/{1}_dynestym2l.npz'.format(ALFPY_HOME, outname), m2l=ml_res)
    return np.array(ml_res)
    
    
# ---------------------------------------------------------------- #
def calm2l_emcee(in_res, alfvar, use_keys, ncpu):
    return None




# ---------------------------------------------------------------- #
def calm2l_mcmc(infile, alfvar, ncpu, outname):
    """
    - for test purpose
    """
    ALFPY_HOME = os.environ['ALFPY_HOME']
    
    samples = np.array(pd.read_csv(infile, delim_whitespace=True, 
                                header=None, comment='#'))[:,1:47]
    tstart = time.time()
    
    with MultiPool() as pool:
        pwork = partial(worker_m2l, alfvar, key_list)
        ml_res = pool.map(pwork, samples)
    ndur = time.time() - tstart
    pool.close()
    print('\ncalculating m2l in .mcmc file: {:.2f}minutes'.format(ndur/60.))
    np.savez('{0}results/{1}_mcmcm2l_b.npz'.format(ALFPY_HOME, outname), m2l=ml_res)
    return np.array(ml_res)