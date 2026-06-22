import numpy as np
import numba
__all__ = ['OUTDIR', 'ALFVAR', 'ALFSSP', 
           'ALFPARAM', 'ALFTDATA', 'ALFIDATA']

OUTDIR = 'results/'
ssp_type = 'vcj'
 
    
# ---------------------------------------------------------------- #    
paramdict = {'velz':0.0, 'sigma':0.0, 'logage':1.0, 'zh':0.0, 'feh':0.0, 'ah':0.0,
             'ch':0.0,'nh':0.0,'nah':0.0,'mgh':0.0,'sih':0.0,'kh':0.0,
             'cah':0.0,'tih':0.0,'vh':0.0,'crh':0.0,'mnh':0.0,'coh':0.0,'nih':0.0,
             'cuh':0.0,'srh':0.0,'bah':0.0,'euh':0.0,'teff':0.0,'imf1':1.3,
             'imf2':2.3,'logfy':-4.0,'sigma2':0.0,'velz2':0.0,'logm7g':-4.0,'hotteff':20.0,
             'loghot':-4.0,'fy_logage':0.3,'logtrans':-4.0,'logemline_h':-4.0,
             'logemline_oiii':-4.0,'logemline_sii':-4.0,'logemline_ni':-4.0,
             'logemline_nii':-4.0,'logemline_oii':-4.0,'jitter':1.0,'imf3':0.08,
             'logsky':-4.0,'imf4':0.0,'h3':0.0,'h4':0.0, 'chi2':1e33}
#alfparam_type = [(i, numba.float64) for i in paramdict.keys()]

#@jitclass(alfparam_type)
class ALFPARAM(object):  
    def __init__(self):
        self.velz = 0.0; self.sigma = 0.0; self.logage = 1.0;
        self.zh = 0.0; self.feh = 0.0; self.ah = 0.0; self.cah = 0.0;
        self.ch = 0.0; self.nh = 0.0; self.nah = 0.0;
        self.mgh = 0.0; self.sih = 0.0; self.kh = 0.0;
        self.cuh = 0.0; self.srh = 0.0; self.bah = 0.0;
        self.euh = 0.0; self.teff = 0.0; self.imf1 = 1.3;
        self.imf2 = 2.3;self.logfy=-4.0; self.sigma2 = 0.0;
        self.velz2 = 0.0; self.logm7g = -4.0; self.hotteff = 20.0;
        self.loghot = -4.0;self.fy_logage=0.3; self.logtrans=-4.0;
        self.logemline_h = -4.0; self.logemline_oiii=-4.0;
        self.logemline_sii = -4.0; self.logemline_ni=-4.0;
        self.logemline_nii = -4.0; self.logemline_oii=-4.0;
        self.jitter=1.0; self.imf3 = 0.08;
        self.logsky = -4.0; self.imf4 = 4.0;
        self.h3 = 0.0; self.h4=0.0;
        #self.chi2 = 1e33 

        
        
# ---------------------------------------------------------------- # 
"""
alfssp_type_3darr = ['solar','nap','nam','cap','cam','fep','fem',
                    'cp','cm','ap','np','nm','tip','tim','mgp','mgm','sip',
                    'sim','crp','mnp','bap','bam','nip','cup','cop','eup',
                    'srp','kp','vp','teffp','teffm','nap6','nap9', 
                     'hotspec'] 
alfssp_type_1darr = ['logagegrid_rfcn', 'logagegrid', 'logzgrid', 
                     'logzgrid2', 'imfx1', 'imfx2', 'imfx3', 
                     'atm_trans_h2o', 'atm_trans_o2', 'lam', 'm7g', 
                     'teffarrhot']
alfssp_type_4darr = ['sspnp']
alfssp_type_5darr = ['logssp']
alfssp_type_6darr = ['logsspm']
alfssp_type = [(i,numba.float64[:]) for i in alfssp_type_1darr] +\
              [(i,numba.float64[:,:,:]) for i in alfssp_type_3darr] +\
              [(i,numba.float64[:,:,:,:]) for i in alfssp_type_4darr] +\
              [(i,numba.float64[:,:,:,:,:]) for i in alfssp_type_5darr] +\
              [(i,numba.float64[:,:,:,:,:,:]) for i in alfssp_type_6darr] 
"""
#@jitclass(alfssp_type)
class ALFSSP(object):    
    def __init__(self, nl, nage_rfcn, nzmet, nage, nzmet3,
                 nimfnp, nimf, nmcut, nhot, allocate_logsspm=False):
        
        self.lam = np.zeros(nl)
        self.m7g = np.zeros(nl) 
        
        self.solar = np.zeros(shape=(nl, nage_rfcn, nzmet))
        self.nap   = np.zeros(shape=(nl, nage_rfcn, nzmet))
        self.nam   = np.zeros(shape=(nl, nage_rfcn, nzmet))
        self.cap   = np.zeros(shape=(nl, nage_rfcn, nzmet))
        self.cam   = np.zeros(shape=(nl, nage_rfcn, nzmet))
        self.fep   = np.zeros(shape=(nl, nage_rfcn, nzmet))
        self.fem   = np.zeros(shape=(nl, nage_rfcn, nzmet))
        self.cp   = np.zeros(shape=(nl, nage_rfcn, nzmet))
        self.cm   = np.zeros(shape=(nl, nage_rfcn, nzmet))
        self.ap   = np.zeros(shape=(nl, nage_rfcn, nzmet))
        self.np   = np.zeros(shape=(nl, nage_rfcn, nzmet))
        self.nm   = np.zeros(shape=(nl, nage_rfcn, nzmet))
        self.tip   = np.zeros(shape=(nl, nage_rfcn, nzmet))
        self.tim   = np.zeros(shape=(nl, nage_rfcn, nzmet))
        self.mgp   = np.zeros(shape=(nl, nage_rfcn, nzmet))
        self.mgm   = np.zeros(shape=(nl, nage_rfcn, nzmet))
        self.sip   = np.zeros(shape=(nl, nage_rfcn, nzmet))
        self.sim   = np.zeros(shape=(nl, nage_rfcn, nzmet))
        self.crp   = np.zeros(shape=(nl, nage_rfcn, nzmet))
        self.mnp   = np.zeros(shape=(nl, nage_rfcn, nzmet))
        self.bap   = np.zeros(shape=(nl, nage_rfcn, nzmet))
        self.bam   = np.zeros(shape=(nl, nage_rfcn, nzmet))
        self.nip   = np.zeros(shape=(nl, nage_rfcn, nzmet))
        self.cup   = np.zeros(shape=(nl, nage_rfcn, nzmet))
        self.cop   = np.zeros(shape=(nl, nage_rfcn, nzmet))
        self.eup   = np.zeros(shape=(nl, nage_rfcn, nzmet))
        self.srp   = np.zeros(shape=(nl, nage_rfcn, nzmet))
        self.kp   = np.zeros(shape=(nl, nage_rfcn, nzmet))
        self.vp   = np.zeros(shape=(nl, nage_rfcn, nzmet))
        self.teffp   = np.zeros(shape=(nl, nage_rfcn, nzmet)) 
        self.teffm   = np.zeros(shape=(nl, nage_rfcn, nzmet)) 
        self.nap6   = np.zeros(shape=(nl, nage_rfcn, nzmet)) 
        self.nap9   = np.zeros(shape=(nl, nage_rfcn, nzmet)) 

                             
        self.logagegrid_rfcn = np.zeros(nage_rfcn) 
        self.logagegrid = np.zeros(nage)
        self.logzgrid = np.zeros(nzmet)
        self.logzgrid2 = np.zeros(nzmet3) 
        self.logssp = np.zeros((nl,nimf,nimf,nage,nzmet))
        if allocate_logsspm:
            self.logsspm = np.zeros((nl,nimf,nimf,nage,nmcut,nzmet3))
        else:
            self.logsspm = np.zeros((1,1,1,1,1,1))
        self.sspnp = np.zeros((nl,nimfnp,nage,nzmet))
        self.imfx1 =  np.zeros(nimf) 
        self.imfx2 = np.zeros(nimf) 
        self.imfx3 = np.zeros(nmcut) 
        self.hotspec = np.zeros((nl, nhot, nzmet))
        self.atm_trans_h2o  = np.zeros(nl) 
        self.atm_trans_o2 = np.zeros(nl) 
        self.teffarrhot = np.zeros(nhot) 
        
        
        

# ---------------------------------------------------------------- # 
alftdata_param = np.array(['lam', 'flx', 'err', 'wgt', 'ires', 'lam0', 'sky'])
alftdata_type = np.array([(i, numba.float64[:]) for i in alftdata_param])
#@jitclass(alftdata_type)
class ALFTDATA(object):
    def __init__(self, ndat):
        self.lam = np.ones(ndat)*1e6
        self.flx = np.zeros(ndat)
        self.err = np.zeros(ndat)
        self.wgt = np.zeros(ndat)
        self.ires = np.zeros(ndat)
        self.lam0 = np.ones(ndat)*1e6
        self.sky = np.zeros(ndat)
        
        
# ---------------------------------------------------------------- # 
alfidata_param = np.array(['indx', 'err'])
alfidata_type = np.array([(i, numba.float64[:]) for i in alfidata_param])
#@jitclass(alfidata_type)
class ALFIDATA(object):
    def __init__(self, nindx):
        self.indx = np.zeros(nindx)
        self.err = np.ones(nindx)*99.
        
        
# ---------------------------------------------------------------- # 
"""
alfvar_type_float = ['params', 'sspgrid', 'data_indx', 'data']
alfvar_type_string = ['ssp_type', 'atlas_imf', 'filename', 'tag' ]
alfvar_type_float = ['fix_age_dep_resp_fcns', 'fix_z_dep_resp_fcns', 
                     'smooth_trans', 'poly_dlam', 
                     'imflo', 'imfhi', 'krpa_imf1', 'krpa_imf2', 'krpa_imf3','imf5', 
                     'msto_t0', 'msto_t1', 'msto_z0', 'msto_z1', 'msto_z2', 'dlstep', ]
alfvar_type_int = ['fit_type', 'use_age_dep_resp_fcns', 'use_z_dep_resp_fcns', 
                   'fit_trans', 'mwimf', 'imf_type', 'observed_frame', 'powell_fitting', 
                   'nonpimf_alpha', 'fit_two_ages', 'nonpimf_regularize', 
                   'fit_indices', 'fit_hermite', 'velbroad_simple', 'extmlpr', 
                   'fit_poly', 'maskem', 'apply_temperrfcn', 'fake_response', 
                   'blueimf_off','nzmet3', 'nstart', 'nend', 'nl', 'nl_fit', 
                   'nlint_max', 'nlint', 'neml', 'npar', 'nage',  'nzmet', 'npowell', 
                   'nage_rfcn', 'nimf_full', 'nimf', 'npolymax', 'ndat', 'nparsimp', 
                   'nmcut', 'nimfoff', 'nimfnp', 'nindx', 'nfil', 'nhot', 'nimfnp5', 
                  'datmax', 'lam7', 'imfr1', 'imfr2', 'imfr3','nmlprtabmax', 'nskylines']
alfvar_type_1darr = ['magsun', 'mbin_nimf9', 'corr_bin_weight', 'l1', 'l2', 'prloarr', 'prhiarr', 
                     'temperrfcn', 'emlines',  'lnlam', 'lsky', 'fsky', 'indx2fit',
                    'npi_alphav', 'npi_renorm', ]
alfvar_type_2darr = ['filters', 'mlprtab', 'indxdef', 'indxcat']   
alfvar_type = [(i,numba.types.unicode_type) for i in alfvar_type_string]+ \
              [(i,numba.int64) for i in alfvar_type_int]+ \
              [(i,numba.float64) for i in alfvar_type_float]+ \
              [(i,numba.float64[:]) for i in alfvar_type_1darr] +\
              [(i,numba.float64[:,:]) for i in alfvar_type_2darr] +\
              [('params', ALFPARAM.class_type.instance_type), 
               ('sspgrid', ALFSSP.class_type.instance_type), 
               ('data_indx', ALFIDATA.class_type.instance_type), 
               ('data',ALFTDATA.class_type.instance_type)]
    
"""    
#@jitclass(alfvar_type)
class ALFVAR(object):
    """
     ! module to set up most arrays and variables
    """
    def __init__(self):
        self.ssp_type = ssp_type
            
        #-------------------set various parameters---------------------!
        # -- 0: fit the full model (IMF, all abundances, nuisance params, etc)
        # -- 1: only fit velz, sigma, SSP age, Z, Fe,C,N,O,Mg,Si,Ca,Ti,Na
        # -- 2: only fit velz, sigma, SSP age, Z
        self.fit_type = 0

        # -- turn on the use of age-dependent response functions
        self.use_age_dep_resp_fcns = 1
        # -- if above is set to 0, fix the response functions to this age (Gyr)
        self.fix_age_dep_resp_fcns = 10.0
        
        # -- turn on the use of Z-dependent response functions
        self.use_z_dep_resp_fcns = 1
        # -- if above is set to 0, fix the response functions to this [Z/H]
        self.fix_z_dep_resp_fcns = 0.0
        # -- flag to include transmission spectrum in fitting
        # -- even if flag is set, only included in full model
        self.fit_trans =1

        # -- force the IMF to be a MW IMF if set
        # -- this is automatically assumed if fit_type=1,2
        self.mwimf = 0

        # -- flag to fit either a double-power law IMF or power-law + cutoff
        # -- 0 = single power-law
        # -- 1 = double power-law
        # -- 2 = power-law + cutoff
        # -- 3 = double power-law + cutoff
        self.imf_type = 1

        # -- are the data in the original observed frame?
        self.observed_frame=1

        # -- IMF used to compute the element response functions
        self.atlas_imf='krpa'  #'salp'

        # -- extra smoothing (km/s) of the transmission spectrum
        # -- if the input spectrum has been smoothed by an amount more than
        # -- the instrumental resolution, set the parameter below to that value
        self.smooth_trans=0.0

        # -- flag used to tell the code if we are fitting in powell mode or not
        # -- this is set internally in the code
        self.powell_fitting = 0

        # -- IMF power-law slope within each bin for non-paramtric IMF
        # -- 0 = flat, 1 = Kroupa, 2 = Salpeter
        self.nonpimf_alpha = 2

        # -- fit two-component SFH. Also requires fit_type=0
        self.fit_two_ages = 1

        # -- regularize the non-parametric IMF
        self.nonpimf_regularize = 1

        #fit indices
        self.fit_indices = 0

        #fit the h3 and h4 parameters for the LOSVD
        self.fit_hermite = 0

        # -- if set, compute velocity broadening via a simple method
        # -- rather than the proper convolution in log_lambda space
        # -- don't turn this on - the "correct" version is just as fast
        # -- unless you are fitting for the Hermite terms!
        self.velbroad_simple=0

        # -- flag to turn on the use of an external, tabulated M/L prior
        self.extmlpr = 0

        #--------------------------------------------------------------!
        #  the options below have not been tested/used in a long time  !
        #  and so are effectively deprecated                           !
        #--------------------------------------------------------------!

        # -- fit a polynomial to the ratio of model and data
        # -- if zero, then both data and model are continuum divided
        self.fit_poly = 1
        # -- mask emission lines? (if 0, then the em lines are incl in the fit)
        self.maskem = 0
        # -- apply template error function? (only works for SDSS stacks)
        self.apply_temperrfcn = 0
        # -- flag to implement fake element response functions
        self.fake_response = 0
        # -- Turn off the IMF sensitivity at <7000A if this parameter is =1
        self.blueimf_off = 0

        #--------------------------------------------------------------!
        #    the parameters below should not be modified unless you    !
        #    really know what you are doing!                           !
        #--------------------------------------------------------------!
        # -- VCJ models
        if ssp_type == 'vcj':
            self.nzmet3 = 3
        elif ssp_type == 'cvd':
            self.nzmet3 = 1


        # -- nstart and nend allow us to use only a subset of
        # -- the full wavelength array
        self.nstart = 100 # 100   ! 0.36 um
        self.nend   = 10566  # 5830(all) ! 5830 (1.10u)
        #self.nend   = 10566 
        
        self.nl = self.nend - self.nstart + 1    # number of spectral elements in SSPs
        self.nl_fit = self.nl    # number actually used over the range to be fit
        self.nlint_max = 10    # (max) number of wavelength intervals
        self.nlint = 0    #actual number of wavelength intervals, determined at run time
        self.neml = 19    #total number of emission lines
        self.npar = 46    #number of parameters
        self.nage = 7    #number of ages in the empirical SSP grid
        self.nzmet = 5    #number of metallicities in the empirical SSP grid
        
        # -- number of parameters used when fitting in Powell model
        # -- or in the super-simple mode (fit_type=2)
        self.npowell = 4
        self.nage_rfcn = 5    #number of ages in the response functions
        
        # -- number of IMF values in the SSP grid
        self.nimf_full=16; self.nmcut=8; self.nimfoff=2; self.nimfnp=9
        self.nimf = self.nimf_full - self.nimfoff
            
        # -- max degree of polynomial used for continuum fitting
        self.npolymax = 14
        # -- wavelength interval used to determine polynomial degree
        self.poly_dlam = 100.0
        # -- max number of data wavelength points
        self.ndat = 30000
        # -- total number of parameters in the simple model
        self.nparsimp = 14
        self.nindx=25  #number of indices defined in allindices.dat
        self.nfil = 3  #number of filters
        self.nhot = 12   #number of hot stars
        # -- mag of sun in r,I,K filters (AB mag)
        self.magsun = np.array([4.64, 4.52, 5.14])
        #lower and upper limits for the IMF,
        #except when the IMF cutoff parameterization is used
        self.imflo, self.imfhi = 0.08, 100.0
        # -- power-law slopes for a Kroupa IMF
        self.krpa_imf1, self.krpa_imf2, self.krpa_imf3 = 1.3, 2.3, 2.3
        self.imf5 = 0.0  #log(imf5) for non-parametric IMF
        # -- linear fit to log(age) vs. log(MS TO mass)
        self.msto_t0 = 0.33250847; self.msto_t1 = -0.29560944
        self.msto_z0 = 0.95402521; self.msto_z1= 0.21944863; self.msto_z2=0.070565820
        self.nimfnp5 = 5
        # -- mass boundaries for non-para IMF (starting at imflo, and ending at imfhi)
        # -- self.mbin_nimf = [0.2,0.4,0.6,0.8,1.0]
        self.mbin_nimf9 = np.array([0.08,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0])
        self.corr_bin_weight = np.zeros(self.nimfnp)

            
        #----------Setup a common block of arrays and vars-------------!
        self.datmax = 0  # length of input data  ??? 
        self.lam7 = 1  # index in lam array at 7000A
        # -- indices for the fiducial IMF (set in setup.f90)
        self.imfr1=1; self.imfr2=1; self.imfr3=1
            
        self.filters = np.zeros((self.nl, self.nfil))  # common array for filters
        self.l1 = np.zeros(self.nlint_max)  # common array for wavelength intervals
        self.l2 = np.zeros(self.nlint_max)
        self.prloarr = np.zeros(self.npar)  # arrays containing the upper and lower prior limits
        self.prhiarr = np.zeros(self.npar)
        self.mlprtab = np.zeros((self.ndat,2))  #array storing the tabulated M/L prior
        self.nmlprtabmax = 1
        self.temperrfcn = np.ones(self.nl)  #array for the template error function
        
        # -- define central wavelengths of emission lines (in vacuum)
        # -- these wavelengths come from NIST
        #  -- [hd, hy, hb, [OIII], [OIII], [NI], [NII], Ha, [NII], 
        #  -- [SII], [SII], [OII], [OII], Balmer, Balmer, Balmer, Balmer, Balmer, Balmer]
        self.emlines = np.array([4102.89, 4341.69, 4862.71, 4960.30, 5008.24, 
                                 5203.05, 6549.86, 6564.61, 6585.27, 6718.29, 
                                 6732.67, 3727.10, 3729.86, 3751.22, 3771.70, 
                                 3798.99, 3836.49, 3890.17, 3971.20])
   
        self.dlstep = 0.  #variables used in velbroad.f90 routine
        self.lnlam = np.zeros(self.nl)

        
        self.nskylines = 39324    #arrays holding the sky emission lines
        self.lsky = np.empty(shape = self.nskylines)
        self.fsky = np.empty(shape = self.nskylines)

        # -- array of index definitions
        self.indxdef = np.zeros((7, self.nindx))
        self.indx2fit = np.zeros(self.nindx)

        # -- index definition for CaT
        # -- CaT index   
        self.indxcat = np.empty(shape=(6,3))
    
        # -- IMF slopes within each bin
        self.npi_alphav = np.zeros(self.nimfnp)
        self.npi_renorm = np.ones(self.nimfnp)

        #---------------------Physical Constants-----------------------!
        #---------------in cgs units where applicable------------------!
        # --- in alf_constants.py
        
        
        #-------------------update TYPE structures---------------------!
        #structure for the set of parameters necessary to generate a model
        # params, sspgrid
        self.params = ALFPARAM()
        self.sspgrid = ALFSSP(self.nl, self.nage_rfcn, self.nzmet, self.nage, \
                              self.nzmet3, self.nimfnp, self.nimf, self.nmcut, \
                              self.nhot, allocate_logsspm=(self.imf_type in [2,3]))
        self.data_indx = ALFIDATA(self.nindx)
        self.data = ALFTDATA(self.ndat)


# ---------------------------------------------------------------- #  
"""
class ALFPARAM(object):  
    def __init__(self):
        paramdict = {'velz':0.0, 'sigma':0.0, 'logage':1.0, 'zh':0.0, 'feh':0.0, 'ah':0.0,
                     'ch':0.0,'nh':0.0,'nah':0.0,'mgh':0.0,'sih':0.0,'kh':0.0,
                     'cah':0.0,'tih':0.0,'vh':0.0,'crh':0.0,'mnh':0.0,'coh':0.0,'nih':0.0,
                     'cuh':0.0,'srh':0.0,'bah':0.0,'euh':0.0,'teff':0.0,'imf1':1.3,
                     'imf2':2.3,'logfy':-4.0,'sigma2':0.0,'velz2':0.0,'logm7g':-4.0,'hotteff':20.0,
                     'loghot':-4.0,'fy_logage':0.3,'logtrans':-4.0,'logemline_h':-4.0,
                     'logemline_oiii':-4.0,'logemline_sii':-4.0,'logemline_ni':-4.0,
                     'logemline_nii':-4.0,'logemline_oii':-4.0,'jitter':1.0,'imf3':0.08,
                     'logsky':-4.0,'imf4':0.0,'h3':0.0,'h4':0.0, 'chi2':huge_number}
        
        paramname = list(paramdict.keys())
        for iname in paramname:
            self.__setattr__(iname, paramdict[iname])
            
"""            
        

        
