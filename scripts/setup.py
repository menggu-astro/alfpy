import numpy as np
from linterp import locate, linterp
from alf_constants import tiny_number, ALF_HOME
from vacairconv import airtovac
from velbroad import velbroad
import time
from functools import partial

"""
read in and set up all the arrays
"""

__all__ = ['setup']

# ---------------------------------------------------------------- #
# -------- parallelize all velbroad part -------- #
# ---------------------------------------------------------------- #
def worker(inlam, sigma0, lam_lo, lam_hi, smooth_arr, velbroad_simple, inarr):
    """
    - use partial; the *last variable* has to be the input spectrum
    """
    return velbroad(inlam, inarr, sigma0, lam_lo, lam_hi, smooth_arr, velbroad_simple)


# ---------------------------------------------------------------- #
def setup(alfvar, onlybasic = False, pool=None):
    """
    !read in and set up all the arrays
    """
    l1um = 1e4; t13 = 1.3; t23 = 2.3; sig0 = 99.;
    lam = np.zeros(alfvar.nl)
    dumi = np.zeros(alfvar.nl)
    shift = 100
    ntrans = 22800
    ltrans = np.zeros(ntrans)
    ftrans_h2o = np.zeros(ntrans)
    ftrans_o2  = np.zeros(ntrans)
    strans     = np.zeros(ntrans)

    velbroad_simple = alfvar.fit_hermite


    nstart, nl = alfvar.nstart, alfvar.nl
    nimf, nimfoff = alfvar.nimf, alfvar.nimfoff
    # ---------------------------------------------------------------!
    # ---------------------------------------------------------------!
    #try:
    #    ALF_HOME = os.environ['ALF_HOME']
    #except:
    #    print('ALF ERROR: ALF_HOME environment variable not set!')

    # -- correction factor between Salpeter, Kroupa and flat intrabin weights
    # -- for non-parametric IMF

    if alfvar.nonpimf_alpha == 0:
        alfvar.corr_bin_weight[:] = 0.0
        alfvar.npi_alphav[:]      = 0.0
        alfvar.npi_renorm[:]      = 1.0

    elif alfvar.nonpimf_alpha == 1:
        alfvar.corr_bin_weight = np.array([1.455, 1.093, 0.898, 0.755, 0.602,
                                           0.434, 0.290, 0.164, 0.053])
        alfvar.npi_alphav = np.array([1.3, 1.3, 1.3, 1.3, 2.3,
                                      2.3, 2.3, 2.3, 2.3])
        alfvar.npi_renorm = np.array([2.0, 2.0, 2.0, 2.0, 1.0,
                                      1.0, 1.0, 1.0, 1.0])

    elif alfvar.nonpimf_alpha == 2:
        alfvar.corr_bin_weight = np.array([2.122, 1.438, 1.083, 0.822, 0.615,
                                           0.443, 0.296, 0.168, 0.054])
        alfvar.npi_alphav[:] = 2.3
        alfvar.npi_renorm[:] = 1.0
    else:
        print('SETUP ERROR: nonpimf_alpha invalid value: ',alfvar.nonpimf_alpha)


    charz  = ['m1.5','m1.0','m0.5','p0.0','p0.2']
    charz2 = ['-1.50','-1.00','-0.50','+0.00','+0.25']
    charm  = ['0.08','0.10','0.15','0.20','0.25','0.30','0.35','0.40']
    chart  = ['t01.0','t03.0','t05.0','t07.0','t09.0','t11.0','t13.5']
    chart2 = ['t01','t03','t05','t09','t13']

    alfvar.sspgrid.logssp[:]  = tiny_number
    alfvar.sspgrid.logsspm[:]  = tiny_number

    # ---- if the data has not been read in, then we need to manually
    # ---- define the lower and upper limits for the instrumental resolution
    # ---- broadening.  Currently this is only triggered if write_a_model is
    # ---- being called (or if an error is made in READ_DATA).
    if alfvar.nlint == 0:
        lamlo = 3.8e3
        lamhi = 2.4e4
    else:
        lamlo = alfvar.l1[0]-500
        lamhi = alfvar.l2[-1]+500  

    # -- read in filter transmission curves
    try:
        f15 = np.loadtxt('{0}infiles/filters.dat'.format(ALF_HOME))
        alfvar.filters[:,0:3] = np.copy(f15[alfvar.nstart-1:alfvar.nend, 1:4])
    except:
        print('SETUP ERROR: filter curves not found')


    # -------------------------------------------------------------------------!
    # -----------------read in the theoretical response functions--------------!
    # ---------------- Line 89 in setup.f90 -----------------------------------!
    # ---------------- atlas_ssp_{1}_Z{2}.abund.{3}.s100 ----------------------!
    # -------------------------------------------------------------------------!
    f20_read_dict_attr = ['lam','solar','nap','nam','cap','cam','fep','fem',
                          'cp','cm','d1','np','nm','ap','tip','tim','mgp','mgm',
                          'sip','sim','teffp','teffm','crp','mnp','bap','bam',
                          'nip','cop','eup','srp','kp','vp','cup','nap6','nap9']

    for k in range(alfvar.nzmet):
        for j in range(alfvar.nage_rfcn):
            filename = "{0}infiles/atlas_ssp_{1}_Z{2}.abund."\
                       "{3}.s100".format(ALF_HOME, chart2[j], charz[k], alfvar.atlas_imf)
            #f20 = np.array(pd.read_csv(filename, delim_whitespace=True, header=None, comment='#'))
            f20 = np.loadtxt(filename, comments='#')

            for icol, iattr in enumerate(f20_read_dict_attr):
                if iattr == 'd1':
                    continue
                else:
                    temcol = f20[alfvar.nstart-1:alfvar.nend, icol]
                    if iattr == 'lam':
                        alfvar.sspgrid.__setattr__(iattr, temcol)
                    else:
                        getattr(alfvar.sspgrid, iattr)[:,j,k] = temcol



    # ------------------------------------------------------------------------- #
    # -- Replace the [Z/H]=+0.2 models with the +0.0 models
    # -- as the former are broken
    nzmet = alfvar.nzmet
    temlist = ['solar','nap','nam','cap','cam','fep',
               'fem','cp','cm','np','nm','ap','tip','tim',
               'mgp','mgm','sip','sim','teffp','teffm','crp',
               'mnp','bap','bam','nip','cop','eup','srp','kp','vp',
               'cup','nap6','nap9']

    for iattr in temlist:
        getattr(alfvar.sspgrid, iattr)[:,:,alfvar.nzmet-1] = getattr(alfvar.sspgrid, iattr)[:,:,alfvar.nzmet-2]

    lam = alfvar.sspgrid.lam
    alfvar.sspgrid.logagegrid_rfcn = np.log10(np.array([1.0, 3.0, 5.0, 9.0, 13.0]))


    # ------------------------------------------------------------------------- #
    # create fake response functions by shifting  
    # the wavelengths by n pixels
    if alfvar.fake_response == 1:
        dumi = alfvar.sspgrid.crp/alfvar.sspgrid.solar
        alfvar.sspgrid.crp[:,:,:] = 1.0
        alfvar.sspgrid.crp[shift-1 : nl-1,:,:] = dumi[: nl-shift, :, :]
        alfvar.sspgrid.crp *= alfvar.sspgrid.solar

        dumi = alfvar.sspgrid.mnp/alfvar.sspgrid.solar
        alfvar.sspgrid.mnp[:,:,:] = 1.0
        alfvar.sspgrid.mnp[shift-1 : nl-1,:,:] = dumi[: nl-shift, :, :]
        alfvar.sspgrid.mnp *= alfvar.sspgrid.solar

        dumi = alfvar.sspgrid.cop/alfvar.sspgrid.solar
        alfvar.sspgrid.cop[:,:,:] = 1.0
        alfvar.sspgrid.cop[shift-1 : nl-1,:,:] = dumi[: nl-shift, :, :]
        alfvar.sspgrid.cop *= alfvar.sspgrid.solar


    #-------------------------------------------------------------------------!
    #-----read in empirical spectra as a function of age and metallicity------!
    #----------- Line 198 in setup.f90 ---------------------------------------!
    #-------------------------------------------------------------------------!

    alfvar.sspgrid.logagegrid = np.log10(np.array([1.0, 3.0, 5.0, 7.0, 9.0, 11.0, 13.5]))
    alfvar.sspgrid.logzgrid   = np.array([-1.5, -1.0, -0.5, 0.0, 0.25])
    # ---- below is to make things work when nzmet3=1 for cvd
    alfvar.sspgrid.logzgrid2[:alfvar.nzmet3] = np.array([-0.5, 0.0, 0.25])[:alfvar.nzmet3]

    # ---- read in two parameter IMF models ---- #
    # ---- {0}infiles/VCJ_v8_mcut0.08_{1}_Z{2}.ssp.imf_varydoublex.s100 ---- #
    print('\nbegin: read in two parameter IMF models, t, z for chart[t], charz[z]')
    for (t,z), _ in np.ndenumerate(alfvar.sspgrid.logssp[0,0,0]):
        if alfvar.ssp_type == 'vcj':
            filename = "{0}infiles/VCJ_v8_mcut0.08_{1}"\
            "_Z{2}.ssp.imf_varydoublex.s100".format(ALF_HOME, chart[t], charz[z])
            #f22 = np.array(pd.read_csv(filename, delim_whitespace=True, header=None, comment='#'))
            f22 = np.loadtxt(filename, comments='#')
        elif alfvar.ssp_type == 'CvD':
            filename = "{0}infiles/CvD_v7s_mcut0.08_{1}"\
            "_Zp0.0.ssp.imf_varydoublex.s100".format(ALF_HOME, chart[t])
            #f22 = np.array(pd.read_csv(filename, delim_whitespace=True, header=None, comment='#'))
            f22 = np.loadtxt(filename, comments='#')
            
        else:
            print('wrong alfvar.ssp_type')
            return 
        
        tmp = f22[alfvar.nstart-1:alfvar.nend, 1:]
        ii = 0
        for j in range(nimf+nimfoff):
            for k in range(nimf+nimfoff):
                if k >= nimfoff and j >= nimfoff:
                    alfvar.sspgrid.logssp[:, j-nimfoff, k-nimfoff, t, z] = np.copy(tmp[:,ii])
                ii+=1        
        

    # ---- read in 3 parameter IMF models ---- #
    # ---- LINE 247 in setup.f90 ---- #
    # ---- VCJ_v8_mcut{1}_{2}_Z{3}.ssp.imf_varydoublex.s100 ---- #

    if alfvar.imf_type in [2, 3] and alfvar.sspgrid.logsspm.shape[0] == 1:
        alfvar.sspgrid.logsspm = np.zeros((
            alfvar.nl,
            alfvar.nimf,
            alfvar.nimf,
            alfvar.nage,
            alfvar.nmcut,
            alfvar.nzmet3,
        ))
    
    if alfvar.imf_type in [2, 3]:
        print('\nread in 3 parameter IMF models, t, m, z for charm[m],chart[t],charz[z+2]:')
        for (t, m, z), _ in np.ndenumerate(alfvar.sspgrid.logsspm[0,0,0]):
            filename = "{0}infiles/VCJ_v8_mcut{1}_{2}_Z{3}"\
                       ".ssp.imf_varydoublex.s100".format(ALF_HOME,charm[m],chart[t],charz[z+2])
            f22 = np.loadtxt(filename, comments='#')
            tmp = f22[alfvar.nstart-1:alfvar.nend, 1:]
            ii=0
            for j in range(nimf+nimfoff):
                for k in range(nimf+nimfoff):
                    if k >= nimfoff and j >= nimfoff:
                        alfvar.sspgrid.logsspm[:,j-nimfoff,k-nimfoff,t,m,z] = np.copy(tmp[:,ii])
                    ii +=1



    # ---- read in non-parametric IMF models ---- #
    # ---- {0}infiles/VCJ_v8_{1}_Z{2}.ssp.imf_nonpara_flat.s100 ---- #
    if alfvar.imf_type == 4:
        for (t, z), _ in np.ndenumerate(alfvar.sspgrid.sspnp[0,0]):
            if alfvar.nonpimf_alpha ==0:
                filename = "{0}infiles/VCJ_v8_{1}_Z{2}"\
                           ".ssp.imf_nonpara_flat.s100".format(ALF_HOME,chart[t],charz[z])
            elif alfvar.nonpimf_alpha == 1:
                filename = "{0}infiles/VCJ_v8_{1}_Z{2}"\
                           ".ssp.imf_nonpara_krpa.s100".format(ALF_HOME,chart[t],charz[z])
            elif alfvar.nonpimf_alpha == 2:
                filename = "{0}infiles/VCJ_v8_{1}_Z{2}"\
                           ".ssp.imf_nonpara_x2.3.s100".format(ALF_HOME,chart[t],charz[z])
            else:
                print('SETUP ERROR: nonpimf_alpha=',alfvar.nonpimf_alpha)
            
            #f22 = np.array(pd.read_csv(filename, delim_whitespace=True, header=None, comment='#'))
            f22 = np.loadtxt(filename, comments='#')
            tmp = f22[alfvar.nstart-1:alfvar.nend, 1:]
            for ii in range(9):
                alfvar.sspgrid.sspnp[:, ii, t, z] = np.copy(tmp[:,ii])


    # -- values of IMF parameters at the grid points
    alfvar.sspgrid.imfx1 = np.array([0.5 + (i + nimfoff)/5.0 for i in range(nimf)])
    alfvar.sspgrid.imfx2 = np.copy(alfvar.sspgrid.imfx1)
    alfvar.sspgrid.imfx3 = np.array([0.08,0.10,0.15,0.2,0.25,0.3,0.35,0.4])

    # -- find indices of the reference IMF
    alfvar.imfr1 = locate(alfvar.sspgrid.imfx1, t13 + 1e-3)
    alfvar.imfr2 = locate(alfvar.sspgrid.imfx2, t23 + 1e-3)
    alfvar.imfr3 = locate(alfvar.sspgrid.imfx3, alfvar.imflo + 1e-3)

    #-------------------------------------------------------------------------!
    #------------------------set up nuisance features-------------------------!
    #-------------------------------------------------------------------------!

    # -- read in hot stars
    for j in range(alfvar.nzmet):
        
        #f24 = np.array(pd.read_csv("{0}infiles/hotteff_feh{1}.dat".format(ALF_HOME, charz2[j]), 
        #                           delim_whitespace=True, header=None, comment='#'))
        filename_ = f"{ALF_HOME}infiles/hotteff_feh{charz2[j]}.dat"
        f24 = np.loadtxt(filename_, comments='#')
        alfvar.sspgrid.hotspec[:,:,j] = f24[alfvar.nstart-1:alfvar.nend, 1:]
        
        """
        f24 = np.array(pd.read_csv("{0}infiles/at12_feh{1}_afe+0.0_t08000g4.00"\
                                   ".spec.s100".format(ALF_HOME, charz2[j]), 
                                   delim_whitespace=True, header=None, comment='#'))
        alfvar.sspgrid.hotspec[:,0,j] = f24[alfvar.nstart-1:alfvar.nend, 1]

        
        f25 = np.array(pd.read_csv("{0}infiles/at12_feh{1}_afe+0.0_t10000g4.00"\
                                   ".spec.s100".format(ALF_HOME, charz2[j]), 
                                   delim_whitespace=True, header=None, comment='#'))
        alfvar.sspgrid.hotspec[:,1,j] = f25[alfvar.nstart-1:alfvar.nend, 1]

        
        f26 = np.array(pd.read_csv("{0}infiles/at12_feh{1}_afe+0.0_t20000g4.00"\
                                   ".spec.s100".format(ALF_HOME, charz2[j]), 
                                   delim_whitespace=True, header=None, comment='#'))
        alfvar.sspgrid.hotspec[:,2,j] = f26[alfvar.nstart-1:alfvar.nend, 1]

        
        f27 = np.array(pd.read_csv("{0}infiles/at12_feh{1}_afe+0.0_t30000g4.00"\
                                   ".spec.s100".format(ALF_HOME, charz2[j]), 
                                   delim_whitespace=True, header=None, comment='#'))
        alfvar.sspgrid.hotspec[:,3,j] = f27[alfvar.nstart-1:alfvar.nend, 1]
        """

        # -- normalize to a 13 Gyr SSP at 1um (same norm for the M7III param)
        # -- NB: this normalization was changed on 7/20/15.  Also, a major
        # -- bug was found in which the indices of the array were reversed.
        vv = locate(alfvar.sspgrid.lam, l1um)
        for i in range(alfvar.nhot):
            part1 = alfvar.sspgrid.hotspec[:,i,j]/alfvar.sspgrid.hotspec[vv,i,j]
            part2 = alfvar.sspgrid.logssp[vv, alfvar.imfr1, alfvar.imfr2, alfvar.nage-1, alfvar.nzmet-2]
            alfvar.sspgrid.hotspec[:,i,j] = part1*part2


    #hot star Teff in kK
    #alfvar.sspgrid.teffarrhot = np.array([8.0, 10.0, 20.0, 30.0])
    alfvar.sspgrid.teffarrhot = np.array([8.,10.,12., 14.,16.,18., 20.,22.,24.,26., 28., 30.0])

    #read in M7III star, normalized to a 13 Gyr SSP at 1um
    f23 = np.loadtxt("{0}infiles/M7III.spec.s100".format(ALF_HOME))
    #d1 = f23['col1']
    alfvar.sspgrid.m7g = f23[alfvar.nstart-1:alfvar.nend, 1]

    #normalization provided here as opposed to in external IDL routine on 5/13/16
    part1 = alfvar.sspgrid.m7g/alfvar.sspgrid.m7g[vv]
    part2 = alfvar.sspgrid.logssp[vv,alfvar.imfr1,alfvar.imfr2,alfvar.nage-1,alfvar.nzmet-2]
    alfvar.sspgrid.m7g = part1*part2

    #-------------------------------------------------------------------------!
    #-------------------------------------------------------------------------!
    #-------------------------------------------------------------------------!

    if alfvar.apply_temperrfcn ==1:
        # ---- read in template error function (computed from SDSS stacks)
        # ---- NB: this hasn't been used in years!
        print( 'WARNING: this option has not been tested in a long time!!')
        f28 = np.loadtxt('{0}infiles/temperrfcn.s350'.format(ALF_HOME))
        alfvar.sspgrid.temperrfcn[:] = f28[alfvar.nstart-1:alfvar.nend:,1]

    #-------------------------------------------------------------------------!
    #------------set up the atm transmission function & sky lines-------------!
    #----------- Line 450 in setup.f90 ---------------------------------------!
    #-------------------------------------------------------------------------!

    print('\nbegin: set up the atm transmission function & sky lines')
    f29 = np.loadtxt("{0}infiles/atm_trans.dat".format(ALF_HOME))
    ltrans = np.copy(f29[:, 0])
    ftrans_h2o = np.copy(f29[:, 1])
    ftrans_o2 = np.copy(f29[:, 2])

    # ---- smooth the trans curve here before interpolation to the main grid
    datmax = alfvar.datmax
    data_ires_max = np.nanmax(alfvar.data.ires)
    if  data_ires_max > 0: # CHECK
        strans = linterp(xin = alfvar.data.lam,
                         yin = alfvar.data.ires,
                         xout = ltrans)
        strans[np.where(strans<0)] = 0
        strans[np.where(strans>data_ires_max)] = data_ires_max
    else:
        # -- force the instrumental resolution to 100 km/s if not explicitly set
        # -- only done here b/c the transmission function is tabulated at high res
        strans[:] = 100.0

    # -- add all the terms in quad, including a floor of 10 km/s
    strans = np.sqrt(np.square(strans) + np.square(alfvar.smooth_trans) + 10.**2)
    
    # -- use the simple version which allows for arrays of arbitrary length
    d1 = alfvar.velbroad_simple
    ftrans_h2o = velbroad(lam = ltrans, spec = ftrans_h2o,
                          sigma = sig0, minl=lamlo, maxl=lamhi,
                          ires = strans, velbroad_simple = 1)

    ftrans_o2 = velbroad(lam = ltrans, spec = ftrans_o2,
                         sigma = sig0, minl=lamlo, maxl=lamhi,
                         ires = strans, velbroad_simple = 1)


    # -- interpolate onto the main wavelength grid.  Force transmission
    # -- to be 1.0 outside of the bounds of the tabulated function
    alfvar.sspgrid.atm_trans_h2o[:] = 1.0
    alfvar.sspgrid.atm_trans_o2[:]  = 1.0
    alfvar.sspgrid.atm_trans_h2o = linterp(xin = ltrans, yin = ftrans_h2o, xout = lam)
    alfvar.sspgrid.atm_trans_o2  = linterp(xin = ltrans, yin = ftrans_o2, xout = lam)

    temind = np.logical_or(lam < ltrans[0], lam > ltrans[ntrans-1])
    alfvar.sspgrid.atm_trans_h2o[temind] = 1.0
    alfvar.sspgrid.atm_trans_o2[temind] = 1.0

    # -- sky lines
    try:
        f29 = np.loadtxt("{0}infiles/radiance_lines.dat".format(ALF_HOME))
        alfvar.lsky, alfvar.fsky = f29[:,0], f29[:,1]
    except:
        print('SETUP ERROR: sky lines file not found')


    # ---------------- smooth by the instrumental resolution ---------------- #
    # ---- use the simple version which allows for arrays of arbitrary length #
    strans_ = linterp(ltrans, strans, alfvar.lsky)
    alfvar.fsky = velbroad(lam = alfvar.lsky, spec = alfvar.fsky,
                           sigma = sig0, minl = lamlo, maxl = lamhi,
                           ires = strans_, velbroad_simple = 1)
    alfvar.fsky = alfvar.fsky / np.nanmax(alfvar.fsky)

    #-------------------------------------------------------------------------!
    #---------smooth the models to the input instrumental resolution----------!
    #----------- Line 516 in setup.f90 ---------------------------------------!
    #-------------------------------------------------------------------------!

    print('begin: smooth the models to the input instrumental resolution')
    
    if np.nanmax(alfvar.data.ires) > 10. and onlybasic == False:
        # ---- the interpolation here is a massive extrapolation beyond the range
        # ---- of the data.  This *should't* matter since we dont use the model
        # ---- beyond the range of the data, but I should double check this at some point
        smooth = linterp(alfvar.data.lam,alfvar.data.ires,alfvar.sspgrid.lam)

        temind = np.where(alfvar.sspgrid.lam<alfvar.data.lam[0])
        smooth[temind] = alfvar.data.ires[0]

        temind = np.where(alfvar.sspgrid.lam>alfvar.data.lam[-1])
        smooth[temind] = alfvar.data.ires[datmax-1]

        # ---- smooth the response functions ---- #
        # ---- line 534 ---- #
        print('-------- smooth the response functions --------')
        temlist = ['solar','nap','nam','cap','cam','fep','fem',
                   'cp','cm','np','nm','ap','tip','tim','mgp','mgm',
                   'sip','sim','teffp','teffm','crp','mnp','bap','bam',
                   'nip','cop','eup','srp','kp','vp','cup','nap6','nap9']
              
        tstart = time.time()
        pwork = partial(worker, alfvar.sspgrid.lam, sig0, lamlo, lamhi, smooth, velbroad_simple)
        
        # ---- define partial worker for all velbroad---- #
        for iattr in temlist:
            print('\n smooth response func ', iattr, end=':')
            index_list = [(j,k) for (j, k), _ in np.ndenumerate(getattr(alfvar.sspgrid, iattr)[0])]
            temarr = pool.map(pwork, [getattr(alfvar.sspgrid, iattr)[:,j,k] for (j, k), _ in np.ndenumerate(getattr(alfvar.sspgrid, iattr)[0])] )
            for ii, (j, k) in enumerate(index_list):
                getattr(alfvar.sspgrid, iattr)[:,j,k] = np.copy(temarr[ii])   



        print('\n smooth hotspec ')
        index_list = [(i, j) for (i, j), _ in np.ndenumerate(alfvar.sspgrid.hotspec[0])]
        temarr = pool.map(pwork, [alfvar.sspgrid.hotspec[:,i,j] for (i, j), _ in np.ndenumerate(alfvar.sspgrid.hotspec[0])] )
        for ii, (i, j) in enumerate(index_list):
            alfvar.sspgrid.hotspec[:,i,j]= np.copy(temarr[ii])

            
        alfvar.sspgrid.m7g = velbroad(alfvar.sspgrid.lam, alfvar.sspgrid.m7g,
                                          sig0, lamlo, lamhi, smooth, 
                                      velbroad_simple = velbroad_simple)

        
        # ---- smooth the standard two-part power-law IMF models ---- #
        print('-------- smooth the standard two-part power-law IMF models')
        
        index_list = [(k, j, t, z) for (k, j, t, z), _ in np.ndenumerate(alfvar.sspgrid.logssp[0])]
        temarr = pool.map(pwork, [alfvar.sspgrid.logssp[:,k,j,t,z] for (k, j, t, z), _ in np.ndenumerate(alfvar.sspgrid.logssp[0])] )
        for ii, (k, j, t, z) in enumerate(index_list):
            alfvar.sspgrid.logssp[:,k,j,t,z] = np.copy(temarr[ii])


        # ---- smooth the 3-part IMF models ---- #
        print('-------- smooth the 3-part IMF models')
        if alfvar.imf_type in [2, 3]:
            index_list = [(k, j, t, m, z) for (k, j, t, m, z), _ in np.ndenumerate(alfvar.sspgrid.logsspm[0])]
            temarr = pool.map(pwork, [alfvar.sspgrid.logsspm[:,k,j,t,m,z] for (k, j, t, m, z), _ in np.ndenumerate(alfvar.sspgrid.logsspm[0])] )
            for ii, (k, j, t, m, z) in enumerate(index_list):
                alfvar.sspgrid.logsspm[:,k,j,t,m,z] = np.copy(temarr[ii])            


        # ---- smooth the non-parametric IMF models ---- #
        print('-------- smooth the non-parametric IMF models')
        if alfvar.imf_type == 4:
            index_list = [(j, t, z) for (j, t, z), _ in np.ndenumerate(alfvar.sspgrid.sspnp[0])]
            temarr = pool.map(pwork, [alfvar.sspgrid.sspnp[:,j,t,z] for (j, t, z), _ in np.ndenumerate(alfvar.sspgrid.sspnp[0])] )
            for ii, (j, t, z) in enumerate(index_list):
                alfvar.sspgrid.sspnp[:,j,t,z] = np.copy(temarr[ii]) 
                    
        ndur = time.time() - tstart

        print('\nparallelized velbroad {:.2f}min'.format(ndur/60.))
    alfvar.sspgrid.logssp  = np.log10(alfvar.sspgrid.logssp + tiny_number)
    alfvar.sspgrid.logsspm = np.log10(alfvar.sspgrid.logsspm + tiny_number)

    #----------------------------------------------------------------!
    #-----------------read in index definitions----------------------!
    #---- Line 624 --------------------------------------------------!
    #----------------------------------------------------------------!
    print('begin:read in index definitions')
    f99 = np.genfromtxt("{0}infiles/allindices.dat".format(ALF_HOME), 
    comments='#', dtype=None, encoding='utf-8', usecols=(0,1,2,3,4,5))

    
    alfvar.indxdef = np.copy(f99[:,:].T)
    for i in range(6):
        alfvar.indxdef[i,:21] = airtovac(alfvar.indxdef[i,:21])

    #----------------------------------------------------------------!
    alfvar.lam7 = locate(lam, 7000.0)

    #----------------------------------------------------------------!
    alfvar.indxcat[0] = np.array([8484.0,8522.0,8642.0])
    alfvar.indxcat[1] = np.array([8513.0,8562.0,8682.0])
    alfvar.indxcat[2] = np.array([8474.0,8474.0,8619.0])
    alfvar.indxcat[3] = np.array([8484.0,8484.0,8642.0])
    alfvar.indxcat[4] = np.array([8563.0,8563.0,8700.0])
    alfvar.indxcat[5] = np.array([8577.0,8577.0,8725.0])

    return alfvar
