import os, math, numpy as np
from functools import partial
from dynesty import utils as dyfunc
from getm2l import getm2l
from getmodel import getmodel
from str2arr import str2arr, fill_param
from tofit_parameters import tofit_params
import time
import h5py
import math, numpy as np
from scipy import interpolate
__all__ = ['calm2l_dynesty']

ALFPY_HOME = os.environ['ALFPY_HOME']
key_list = [k for k, (v1, v2) in tofit_params.items() if v1 == True]

# ---------------------------------------------------------------- #
def worker_m2l(alfvar, use_keys, inarr):
    tem_posarr = fill_param(inarr, usekeys = use_keys)
    tem_pos = str2arr(2, inarr = tem_posarr)
    # ---- turn off various parameters for computing M/L
    tem_pos.logemline_h    = -8.0
    tem_pos.logemline_oii  = -8.0
    tem_pos.logemline_oiii = -8.0
    tem_pos.logemline_nii  = -8.0
    tem_pos.logemline_sii  = -8.0
    tem_pos.logemline_ni   = -8.0
    tem_pos.logtrans       = -8.0
    tem_pos.sigma          = 4.0
    
    tem_mspec = getmodel(tem_pos, alfvar=alfvar)
    tem_mspec_mw = getmodel(tem_pos, alfvar=alfvar, mw=1)
    m2l = getm2l(alfvar.sspgrid.lam, tem_mspec, tem_pos, 
                 alfvar.nstart, alfvar.nend, alfvar.nfil, alfvar.imf_type, )
    m2lmw = getm2l(alfvar.sspgrid.lam, tem_mspec_mw, tem_pos,
                   alfvar.nstart, alfvar.nend, alfvar.nfil, alfvar.imf_type, mw=1)
    return np.append(m2l,m2lmw)


# ---------------------------------------------------------------- #
def calm2l_dynesty(in_res, alfvar, use_keys, outname, pool=None):
    print('creating results file:\n {0}results_dynesty/res_dynesty_{1}.hdf5'.format(ALFPY_HOME, outname))
    f1 = h5py.File("{0}results_dynesty/res_dynesty_{1}.hdf5".format(ALFPY_HOME, outname), "w")
    for ikey in ['samples', 'logwt', 'logl', 'logvol', 'logz', 'logzerr', 'information']:
        dset = f1.create_dataset(ikey, dtype=np.float32, data=np.array(getattr(in_res, ikey), dtype=np.float32))

    samples, weights = in_res.samples, np.exp(in_res.logwt - in_res.logz[-1])
    mean, cov = dyfunc.mean_and_cov(samples, weights)
    samples = dyfunc.resample_equal(in_res.samples, weights)

    dset = f1.create_dataset('samples_eq', dtype=np.float32, data= np.array(samples, dtype=np.float32))
    dset = f1.create_dataset('mean', dtype=np.float32, data= np.array(mean, dtype=np.float32))
    dset = f1.create_dataset('cov', dtype=np.float32, data= np.array(cov, dtype=np.float32))
    #dset = f1.create_dataset('use_keys', dtype=dt, data=use_keys)

    nspec = samples.shape[0]
    select_ind = np.random.choice(np.arange(nspec), size=1000)
    samples = np.copy(samples[select_ind,:])

    tstart = time.time()
                    
    pwork = partial(worker_m2l, alfvar, use_keys)
    ml_res = list(map(pwork, samples)) if pool is None else pool.map(pwork, samples)

    ndur = time.time() - tstart
    print('\npost processing dynesty results: {:.2f}minutes'.format(ndur/60.))

    dset = f1.create_dataset('m2l', dtype=np.float32, data= np.array(ml_res, dtype=np.float32))
    f1.close()


    
# ---------------------------------------------------------------- #    
class alfres(object):
    """
    Examples:
    
    from tofit_parameters import tofit_params
    use_keys = [k for k, (v1, v2) in tofit_params.items() if v1 == True]
    [emcee results]:
        res = pickle.load(open('{0}results_emcee/res_emcee_filename_tag.p'.format(ALFPY_HOME), 
        "rb" ))
        prob = pickle.load(open('{0}results_emcee/prob_emcee_filename_tag.p'.format(ALFPY_HOME), 
        "rb"  ))
        pos2D = res.reshape(res.shape[0]*res.shape[1], res.shape[2])
        prob1D = prob.flatten()
                        
        out = alfres(pos2D, use_keys, prob1D)
        out.libcorrect()
    [dynesty results]:
        res = pickle.load(open('../results_dynesty/res_dynesty_filename_tag.p', 
                        "rb" ))
        out = alfres(res.samples, use_keys, res.logl)
        out.libcorrect()
    """
    def __init__(self, posarr, usekeys, prob):
        
        self.labels = list(usekeys)
        self.posarr = np.array(posarr)
        self.prob = np.array(prob)
        
        #reshape_ = posarr.reshape(posarr.shape[0]*posarr.shape[1], 
        #                          posarr.shape[2])
        assert posarr.ndim == 2, "Please reshape posarr"
        assert prob.ndim == 1, "Please flatten prob"
        """
        0:   Mean of the posterior
        1:   Parameter at chi^2 minimum
        2:   1 sigma error
        3-7: 2.5%, 16%, 50%, 84%, 97.5% CLs
        8-9: lower and upper priors
        """

        self.mean = dict(zip(usekeys, np.nanmean(posarr, axis=0)))
        self.minchi2 = dict(zip(self.labels, 
                                posarr[np.where(prob==prob.max())[0], :][0,:]))
        self.onesigma = dict(zip(usekeys, np.nanstd(posarr, axis=0)))
        self.cl25 = dict(zip(usekeys, np.nanpercentile(posarr, 2.5, axis=0)))
        self.cl16 = dict(zip(usekeys, np.nanpercentile(posarr, 16, axis=0)))
        self.cl50 = dict(zip(usekeys, np.nanpercentile(posarr, 50, axis=0)))
        self.cl84 = dict(zip(usekeys, np.nanpercentile(posarr, 84, axis=0)))
        self.cl98 = dict(zip(usekeys, np.nanpercentile(posarr, 97.5, axis=0)))
        self.mcmc = dict(zip(usekeys, posarr.T))
        
        
    def libcorrect(self):
        """
        Need to correct the raw abundance values given by ALF.
        Use the metallicity-dependent correction factors from the literature.
        To-Do:
            Only correcting the mean of the posterior values for now.
            Correct other parameters later.
        """

        #;Schiavon 2007
        #libmgfe = [0.4,0.4,0.4,0.4,0.29,0.20,0.13,0.08,0.05,0.04]
        #libcafe = [0.32,0.3,0.28,0.26,0.20,0.12,0.06,0.02,0.0,0.0]
        # library correction factors from Schiavon 2007, Table 6;
        libfeh  = [-1.6,-1.4,-1.2,-1.0,-0.8,-0.6,-0.4,-0.2,0.0,0.2]
        libofe  = [0.6,0.5,0.5,0.4,0.3,0.2,0.2,0.1,0.0,0.0]
        #libmgfe = [0.4,0.4,0.4,0.4,0.29,0.20,0.13,0.08,0.0,0.0]
        #libcafe = [0.32,0.3,0.28,0.26,0.20,0.12,0.06,0.02,0.0,0.0]
        # Bensby et al. 2014
        #;fitted to Milone et al. 2011 HR MILES stars
        libmgfe = [0.4,0.4,0.4,0.4,0.34,0.22,0.14,0.11,0.05,0.04]
        #;from B14
        libcafe = [0.32,0.3,0.28,0.26,0.26,0.17,0.12,0.06,0.0,0.0]
        #libmgfe = [0.4,0.4,0.4,0.38,0.37,0.27,0.21,0.12,0.05,0.0]
        #libcafe = [0.32,0.3,0.28,0.26,0.26,0.17,0.12,0.06,0.0,0.0]


        # In ALF the oxygen abundance is used a proxy for alpha abundance
        # why? --> NOTE: Forcing factors to be 0 for [Fe/H]=0.0,0.2
        spl_delafe = interpolate.interp1d(libfeh, libofe,kind='linear',bounds_error=False,fill_value='extrapolate')
        spl_delmgfe = interpolate.interp1d(libfeh, libmgfe, kind='linear',bounds_error=False,fill_value='extrapolate')
        spl_delcafe = interpolate.interp1d(libfeh, libcafe, kind='linear',bounds_error=False,fill_value='extrapolate')
        # .mcmc file first
        al_mcmccorr = spl_delafe(np.copy(self.mcmc['zh']))
        mg_mcmccorr = spl_delmgfe(np.copy(self.mcmc['zh']))
        ca_mcmccorr = spl_delcafe(np.copy(self.mcmc['zh']))
        
        #al_mcmccorr = np.interp(np.copy(self.mcmc['zh']), libfeh, libofe)
        #mg_mcmccorr = np.interp(np.copy(self.mcmc['zh']), libfeh, libmgfe)
        #ca_mcmccorr = np.interp(np.copy(self.mcmc['zh']), libfeh, libcafe)
        

        udlabel = ['aFe', 'MgFe', 'SiFe', 'CaFe', 'TiFe', 'CFe',
                   'NFe', 'NaFe', 'KFe', 'VFe', 'CrFe', 'MnFe',
                   'CoFe', 'NiFe', 'CuFe', 'SrFe', 'BaFe', 'EuFe']
        udlabel = [i.lower() for i in  udlabel]
        for ilabel in udlabel:
            xFe_mcmc = self.mcmc[ilabel[:-2]+'h'] - self.mcmc['feh']
            self.mcmc[ilabel] = np.copy(xFe_mcmc)
        self.mcmc['afe'] =  np.copy(self.mcmc['afe']) + al_mcmccorr
        self.mcmc['mgfe'] =  np.copy(self.mcmc['mgfe']) + mg_mcmccorr
        self.mcmc['cafe'] =  np.copy(self.mcmc['cafe']) + ca_mcmccorr
        self.mcmc['sife'] =  np.copy(self.mcmc['sife']) + ca_mcmccorr
        self.mcmc['tife'] =  np.copy(self.mcmc['tife']) + ca_mcmccorr
        self.mcmc['feh'] = self.mcmc['feh'] + self.mcmc['zh']  # mcmc.Fe is updated

        # update mean, cl50, cl16, cl84, minchi2, onesigma for all elements
        old_minchi2FeH = float(self.minchi2['feh'])
        old_onesigmaFeH = float(self.onesigma['feh'])
        
        udlabel = ['aFe', 'MgFe', 'SiFe', 'CaFe', 'TiFe', 'CFe',
                   'NFe', 'NaFe', 'KFe', 'VFe', 'CrFe', 'MnFe',
                   'CoFe', 'NiFe', 'CuFe', 'SrFe', 'BaFe', 'EuFe']
        udlabel = [i.lower() for i in  udlabel]
        for ilabel in udlabel:
            self.mean.update({ilabel: float( np.nanmean(self.mcmc[ilabel]) )})  #lib corrected
            self.cl50.update({ilabel: float( np.percentile(self.mcmc[ilabel], 50))})  #lib corrected
            self.cl16.update({ilabel: float( np.percentile(self.mcmc[ilabel], 16))})  #lib corrected
            self.cl84.update({ilabel: float( np.percentile(self.mcmc[ilabel], 84))})  #lib corrected
            self.minchi2.update({ilabel: float( self.minchi2[ilabel[:-2]+'h'] ) - old_minchi2FeH}) # NOT corrected
            self.onesigma.update({ilabel: math.sqrt(float( self.onesigma[ilabel[:-2]+'h'] )**2. + old_onesigmaFeH**2.)})

        # update aFe, MgFe, CaFe, SiFe and TiFe with lib correction, only for .minchi2
        self.minchi2['afe'] =  self.minchi2['afe'] + spl_delafe(float(self.minchi2['zh']))
        self.minchi2['mgfe'] =  self.minchi2['mgfe'] + spl_delmgfe(float(self.minchi2['zh']))
        self.minchi2['cafe'] =  self.minchi2['cafe'] + spl_delcafe(float(self.minchi2['zh']))
        self.minchi2['sife'] =  self.minchi2['sife'] + spl_delcafe(float(self.minchi2['zh']))
        self.minchi2['tife'] =  self.minchi2['tife'] + spl_delcafe(float(self.minchi2['zh']))

        # ------ #
        # update FeH at last
        self.mean['feh'] = float(np.nanmean(self.mcmc['feh']))
        self.minchi2['feh'] = float(self.minchi2['feh']) + float(self.minchi2['zh'])
        self.cl50['feh'] = float( np.percentile(self.mcmc['feh'], 50) )
        self.cl16['feh'] = float( np.percentile(self.mcmc['feh'], 16) )
        self.cl84['feh'] = float( np.percentile(self.mcmc['feh'], 84) )
        self.onesigma['feh'] = math.sqrt(float(self.onesigma['feh'])**2. + float(self.onesigma['zh'])**2.)

        # update velocity dispersion
        self.mcmc['sigma'] = np.sqrt(np.copy(self.mcmc['sigma'])**2. + 100.**2.)
        self.mean['sigma'] = np.nanmean( self.mcmc['sigma'] )
        self.minchi2['sigma'] = np.sqrt(np.copy(self.minchi2['sigma'])**2. + 100.**2.)
        self.cl50['sigma'] = np.percentile( self.mcmc['sigma'], 50 )
        self.cl16['sigma'] = np.percentile( self.mcmc['sigma'], 16 )
        self.cl84['sigma'] = np.percentile( self.mcmc['sigma'], 84 )
        self.onesigma['sigma'] = np.nanstd( self.mcmc['sigma'] )
