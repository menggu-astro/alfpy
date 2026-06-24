import os, sys, copy, pickle, numpy as np
import matplotlib.pyplot as plt
import emcee, time
import dynesty
from tofit_parameters import tofit_params
from func import func
from str2arr import fill_param, str2arr
from post_process import calm2l_dynesty, worker_m2l
from alf_vars import ALFVAR
from alf_constants import tiny_number
from priors import TopHat, ClippedNormal
from read_data import read_data
from linterp import locate, linterp
from setup import setup
from set_pinit_priors import set_pinit_priors
from scipy.optimize import differential_evolution
import multiprocessing as mp

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"

_G_CALCULATOR = None

def _init_worker(calculator):
    global _G_CALCULATOR
    _G_CALCULATOR = calculator

def _log_prob_worker(theta):
    return _G_CALCULATOR.log_prob(theta)

def _m2l_worker(theta):
    return worker_m2l(_G_CALCULATOR.alfvar, _G_CALCULATOR.keys, theta)

def _save_emcee_text_outputs(ALFPY_HOME, filename, tag, alfvar, use_keys, res, prob, prloarr, prhiarr, nwalkers, nburn, nmcmc, ncpu, acceptance_fraction, elapsed_seconds, pool=None):
    outdir = f"{ALFPY_HOME}results_emcee"
    outstem = filename if tag == "" else f"{filename}_{tag}"
    pos2d = res.reshape(res.shape[0] * res.shape[1], res.shape[2])
    prob1d = prob.reshape(prob.shape[0] * prob.shape[1])
    chi2 = -2.0 * prob1d
    posfull = np.array([fill_param(irow, use_keys) for irow in pos2d])
    chunksize = max(1, len(pos2d) // (max(1, ncpu) * 8))
    m2l = np.array(pool.map(_m2l_worker, pos2d, chunksize)) if pool is not None else np.array([worker_m2l(alfvar, use_keys, irow) for irow in pos2d])
    np.savetxt(f"{outdir}/{outstem}.mcmc", np.column_stack([chi2, posfull, m2l]), fmt=["%12.5E"] + ["%11.4f"] * (posfull.shape[1] + m2l.shape[1]))
    ibest = np.nanargmax(prob1d)
    best_params = pos2d[ibest]
    best_chi2 = chi2[ibest]
    _, best_mspec = func(alfvar, best_params, use_keys, funit=True)
    np.savetxt(f"{outdir}/{outstem}.bestspec", np.transpose(best_mspec), delimiter="     ", fmt="   %12.4f   %12.4E   %12.4E   %12.4E   %12.4E   %12.4E")
    combined = np.column_stack([posfull, m2l])
    zeros_m2l = np.zeros(m2l.shape[1])
    sum_rows = np.vstack([
        np.r_[best_chi2, np.nanmean(combined, axis=0)],
        np.r_[best_chi2, posfull[ibest], zeros_m2l],
        np.r_[0.0, np.nanstd(combined, axis=0)],
        np.r_[0.0, np.nanpercentile(combined, 2.5, axis=0)],
        np.r_[0.0, np.nanpercentile(combined, 16.0, axis=0)],
        np.r_[0.0, np.nanpercentile(combined, 50.0, axis=0)],
        np.r_[0.0, np.nanpercentile(combined, 84.0, axis=0)],
        np.r_[0.0, np.nanpercentile(combined, 97.5, axis=0)],
        np.r_[0.0, prloarr, zeros_m2l],
        np.r_[0.0, prhiarr, zeros_m2l],
    ])
    header = "\n".join([
        f"   Elapsed Time: {elapsed_seconds / 3600.0:6.2f} hr",
        f"    ssp_type  = {alfvar.ssp_type}",
        f"    fit_type  = {alfvar.fit_type:2d}",
        f"    imf_type  = {alfvar.imf_type:2d}",
        f"  fit_hermite = {alfvar.fit_hermite:2d}",
        f" fit_two_ages = {alfvar.fit_two_ages:2d}",
        f"     nonpimf  = {alfvar.nonpimf_alpha:2d}",
        f"   obs_frame  = {alfvar.observed_frame:2d}",
        f"    fit_poly  = {alfvar.fit_poly:2d}",
        f"       mwimf  = {alfvar.mwimf:2d}",
        f"   age-dep Rf = {alfvar.use_age_dep_resp_fcns:2d}",
        f"     Z-dep Rf = {alfvar.use_z_dep_resp_fcns:2d}",
        f"   Nwalkers   = {nwalkers:6d}",
        f"   Nburn      = {nburn:6d}",
        f"   Nchain     = {nmcmc:6d}",
        f"   Nsample    = {1:6d}",
        f"   Nwave      = {alfvar.nl:6d}",
        f"   Ncores     = {ncpu:6d}",
        f"   facc: {acceptance_fraction:6.3f}",
        "   rows: mean posterior, pos(chi^2_min), 1 sigma errors, 2.5%, 16%, 50%, 84%, 97.5% CL, lower priors, upper priors ",
    ])
    np.savetxt(f"{outdir}/{outstem}.sum", sum_rows, fmt=["%12.5E"] + ["%11.4f"] * (sum_rows.shape[1] - 1), header=header)

# -------------------------------------------------------- #
def func_2min(inarr):
    """Minimization function for the first 4 parameters."""
    return func(global_alfvar,
                inarr,
                use_keys[:len(inarr)],
                prhiarr=global_prhiarr,
                prloarr=global_prloarr,
               )

# -------------------------------------------------------- #
def build_alf_model(filename, tag='', pool_type='multiprocessing', run_de=False):
    """
    Build an ALFVAR model based on the specified input file.
    Parameters:
        - filename: Name of the input file.
        - tag: Tag for the output file.
        - pool_type: Multiprocessing or MPI pool type.
    - based on [alf.f90](https://github.com/cconroy20/alf/blob/master/src/alf.f90)
    """
    ALFPY_HOME = os.environ['ALFPY_HOME']
    for ifolder in ['results_emcee', 'results_dynesty', 'subjobs']:
        if os.path.exists(ALFPY_HOME + ifolder) is not True:
            os.makedirs(ALFPY_HOME + ifolder)

    alfvar = ALFVAR()
    global use_keys
    use_keys = [k for k, (v1, v2) in tofit_params.items() if v1 == True]

    #---------------------------------------------------------------!
    #---------------------------Setup-------------------------------!
    #---------------------------------------------------------------!
    alfvar.fit_indices = 0  #flag specifying if fitting indices or spectra

    # ---- flag determining the level of complexity
    # ---- 0=full, 1=simple, 2=super-simple.  See sfvars for details
    alfvar.fit_type = 0  # do not change; use use_keys to specify parameters

    # ---- fit h3 and h4 parameters
    alfvar.fit_hermite = 0

    # ---- type of IMF to fit
    # ---- 0=1PL, 1=2PL, 2=1PL+cutoff, 3=2PL+cutoff, 4=non-parametric IMF
    alfvar.imf_type = 1

    # ---- are the data in the original observed frame?
    alfvar.observed_frame = 0
    alfvar.mwimf = 0  #force a MW (Kroupa) IMF

    if alfvar.mwimf:
        alfvar.imf_type = 1

    # ---- fit two-age SFH or not?  (only considered if fit_type=0)
    alfvar.fit_two_ages = 1

    # ---- IMF slope within the non-parametric IMF bins
    # ---- 0 = flat, 1 = Kroupa, 2 = Salpeter
    alfvar.nonpimf_alpha = 2

    # ---- turn on/off the use of an external tabulated M/L prior
    alfvar.extmlpr = 0

    # ---- set initial params, step sizes, and prior ranges
    _, prlo, prhi = set_pinit_priors(alfvar.imf_type)

    # ---- change the prior limits to kill off these parameters
    prhi.logm7g = -5.0
    prhi.teff   =  2.0
    prlo.teff   = -2.0

    # ---- mass of the young component should always be sub-dominant
    prhi.logfy = -0.5

    # ---------------------------------------------------------------!
    # --------------Do not change things below this line-------------!
    # ---------------unless you know what you are doing--------------!
    # ---------------------------------------------------------------!
    # ---- regularize non-parametric IMF (always do this)
    alfvar.nonpimf_regularize = 1

    # ---- dont fit transmission function in cases where the input
    # ---- spectrum has already been de-redshifted to ~0.0
    if alfvar.observed_frame == 0 or alfvar.fit_indices == 1:
        alfvar.fit_trans = 0
        prhi.logtrans = -5.0
        prhi.logsky   = -5.0
    else:
        alfvar.fit_trans = 1

    # ---- extra smoothing to the transmission spectrum.
    # ---- if the input data has been smoothed by a gaussian
    # ---- in velocity space, set the parameter below to that extra smoothing
    alfvar.smooth_trans = 0.0

    if (alfvar.ssp_type == 'cvd'):
        # ---- always limit the [Z/H] range for CvD since
        # ---- these models are actually only at Zsol
        prhi.zh =  0.01
        prlo.zh = -0.01
        if (alfvar.imf_type > 1):
            print('ALF ERROR, ssp_type=cvd but imf>1')

    if alfvar.fit_type in [1,2]:
        alfvar.mwimf=1

    #---------------------------------------------------------------!

    if filename is None:
        print('ALF ERROR: You need to specify an input file')
        teminput = input("Name of the input file: ")
        if len(teminput.split(' '))==1:
            filename = teminput
        elif len(teminput.split(' '))>1:
            filename = teminput[0]
            tag = teminput[1]


    # ---- write some important variables to screen
    print(" ************************************")
    if alfvar.fit_indices == 1:
        print(" ***********Index Fitter*************")
    else:
        print(" **********Spectral Fitter***********")
    print(" ************************************")
    print("   ssp_type  =", alfvar.ssp_type)
    print("   fit_type  =", alfvar.fit_type)
    print("   imf_type  =", alfvar.imf_type)
    print(" fit_hermite =", alfvar.fit_hermite)
    print("fit_two_ages =", alfvar.fit_two_ages)
    if alfvar.imf_type == 4:
        print("   nonpimf   =", alfvar.nonpimf_alpha)
    print("  obs_frame  =",  alfvar.observed_frame)
    print("      mwimf  =",  alfvar.mwimf)
    print("  age-dep Rf =",  alfvar.use_age_dep_resp_fcns)
    print("    Z-dep Rf =",  alfvar.use_z_dep_resp_fcns)
    #print("  Ncores     = ",  ntasks)
    print("  filename   = ",  filename, ' ', tag)
    print(" ************************************")
    #print('\n\nStart Time ',datetime.now())

    #---------------------------------------------------------------!

    # ---- read in the data and wavelength boundaries
    alfvar.filename = filename
    alfvar.tag = tag

    if alfvar.fit_indices == 0:
        alfvar = read_data(alfvar)
        # ---- read in the SSPs and bandpass filters
        # ------- setting up model arry with given imf_type ---- #

        pool = setup_pool(pool_type)

        print('\nsetting up model arry with given imf_type and input data\n')
        tstart = time.time()
        alfvar = setup(alfvar, onlybasic = False, pool = pool)
        #alfvar = setup(alfvar, onlybasic = True, pool = pool)  # use onlybasic for test purpose
        ndur = time.time() - tstart
        print('\n Total time for setup {:.2f}min'.format(ndur/60))


        ## ---- This part requires alfvar.sspgrid.lam ---- ##
        lam = np.copy(alfvar.sspgrid.lam)
        # ---- interpolate the sky emission model onto the observed wavelength grid
        # ---- moved to read_data
        if alfvar.observed_frame == 1:
            alfvar.data.sky = linterp(alfvar.lsky, alfvar.fsky, alfvar.data.lam)
            alfvar.data.sky[alfvar.data.sky<0] = 0.
        else:
            alfvar.data.sky[:] = tiny_number
        alfvar.data.sky[:] = tiny_number  # ?? why?

        # ---- we only compute things up to 500A beyond the input fit region
        alfvar.nl_fit = min(max(locate(lam, alfvar.l2[-1]+500.0),0),alfvar.nl-1)
        ## ---- define the log wavelength grid used in velbroad.f90
        alfvar.dlstep = (np.log(alfvar.sspgrid.lam[alfvar.nl_fit])-
                         np.log(alfvar.sspgrid.lam[0]))/(alfvar.nl_fit+1)

        for i in range(alfvar.nl_fit):
            alfvar.lnlam[i] = i*alfvar.dlstep + np.log(alfvar.sspgrid.lam[0])


    # ---- convert the structures into their equivalent arrays
    prloarr = str2arr(switch=1, instr = prlo)
    prhiarr = str2arr(switch=1, instr = prhi)

    # ---- this is the master process
    # ---- estimate velz ---- #
    print("  Fitting ",alfvar.nlint," wavelength intervals")
    nlint = alfvar.nlint
    l1, l2 = alfvar.l1, alfvar.l2
    print('wavelength bourdaries: ', l1, l2)
    if l2[-1]>np.nanmax(lam) or l1[0]<np.nanmin(lam):
        print('ERROR: wavelength boundaries exceed model wavelength grid')
        print(l2[nlint-1],lam[nl-1],l1[0],lam[0])

    global global_alfvar, global_prloarr, global_prhiarr
    global_alfvar = copy.deepcopy(alfvar)
    global_prloarr = copy.deepcopy(prloarr)
    global_prhiarr = copy.deepcopy(prhiarr)
    # ---- optimize the first four parameters
    # ---- using differential evolution
    # ---- then shrink the prior based on the optimization results
    # ---- although the updated prior range has not been extensively tested
    de_keys = ['velz', 'sigma', 'logage', 'zh']
    len_optimize = len(de_keys)
    all_key_list = list(tofit_params.keys())
    prloarr_usekeys = np.array([global_prloarr[i_] for i_, k_ in enumerate(all_key_list) if k_ in use_keys])
    prhiarr_usekeys = np.array([global_prhiarr[i_] for i_, k_ in enumerate(all_key_list) if k_ in use_keys])

    prior_bounds = list(zip(prloarr_usekeys[:len_optimize], prhiarr_usekeys[:len_optimize]))
    prrange = [10, 10, 0.1, 0.1]  # Assumed range adjustments

    if run_de:
        print('will narrow prior for the following four parameters: \n', use_keys[:len_optimize])
        print(f'prior_bounds for the first four parameters: {prior_bounds}\n')
        optimize_res = differential_evolution(
            func_2min,
            bounds = prior_bounds,
            disp=True,
            polish=False,
            updating='deferred',
            workers=1)
        print('optimized parameters', optimize_res)
        optimize_res_x = optimize_res.x
        global_all_prior = [ClippedNormal(
            np.array(optimize_res_x)[i], prrange[i],
            global_prloarr[i],
            global_prhiarr[i]) for i in range(len_optimize)] + [
                TopHat(global_prloarr[i+len_optimize],
                              global_prhiarr[i+len_optimize]) for i in range(len(all_key_list)-len_optimize)]
    else:
        optimize_res_x = None
        global_all_prior = [TopHat(global_prloarr[i], global_prhiarr[i]) for i in range(len(all_key_list))]

    pool.close()
    return [alfvar, prloarr, prhiarr, global_all_prior, optimize_res_x, run_de]


# -------- #
def setup_pool(pool_type, ncpu=4):
    """Set up the multiprocessing pool."""
    if pool_type == 'multiprocessing':
        import multiprocessing
        return multiprocessing.Pool(processes=ncpu)
    raise ValueError(f"unsupported pool_type {pool_type!r}; only 'multiprocessing' is supported")


# -------------------------------------------------------- #
class LogProbCalculator:
    """
    use a class instead of relying on global variables
    """
    def __init__(self, alfvar, prloarr, prhiarr, all_prior, keys):
        self.alfvar = alfvar
        self.prloarr = prloarr
        self.prhiarr = prhiarr
        self.all_prior = all_prior
        self.keys = keys


    def log_prob(self, inarr):
        """Log-probability function for emcee."""
        log_p = func(self.alfvar,
                     inarr,
                     self.keys,
                     prhiarr=self.prhiarr,
                     prloarr=self.prloarr)
        if not np.isfinite(log_p):
            return -np.inf
        return -0.5 * log_p

    def log_prob_nested(self, posarr):
        """Log-probability function for dynesty."""
        res_ = func(self.alfvar,
                    posarr,
                    usekeys=self.keys,
                    prhiarr=self.prhiarr,
                    prloarr=self.prloarr)
        if not np.isfinite(res_):
            return -np.inf
        return -0.5 * res_


    def prior_transform(self, unit_coords):
        """Transform unit coordinates to prior ranges for dynesty."""
        all_key_list = list(tofit_params.keys())
        res_ = np.array([self.all_prior[all_key_list.index(ikey)].unit_transform(unit_coords[i]) for i, ikey in enumerate(self.keys)])
        return res_

# -------------------------------------------------------- #
def alf(filename,
        tag='',
        nwalkers = 128,
        nburn = 500,
        nmcmc = 100,
        run='emcee',
        pool_type='multiprocessing',
        ncpu=1,
        nested_post_process=False,
        model=None):
    """
    Main function to perform ALF fitting using either emcee or dynesty.
    - based on alf.f90, `https://github.com/cconroy20/alf/blob/master/src/alf.f90`
    Master program to fit the absorption line spectrum, or indices,
    #  of a quiescent (>1 Gyr) stellar population
    # Some important points to keep in mind:
    # 1. The prior bounds on the parameters are specified in set_pinit_priors.
    #    Always make sure that the output parameters are not hitting a prior.
    # 2. Make sure that the chain is converged in all relevant parameters
    #    by plotting the chain trace (parameter vs. chain step).
    # 3. Do not use this code blindly.  Fitting spectra is a
    #    subtle art and the code can easily fool you if you don't know
    #    what you're doing.  Make sure you understand *why* the code is
    #    settling on a particular parameter value.
    # 4. Wavelength-dependent instrumental broadening is included but
    #    will not be accurate in the limit of modest-large redshift b/c
    #    this is implemented in the model restframe at code setup time
    # 5. The code can fit for the atmospheric transmission function but
    #    this will only work if the input data are in the original
    #    observed frame; i.e., not de-redshifted.
    # 6. I've found that Nwalkers=1024 and Nburn=~10,000 seems to
    #    generically yield well-converged solutions, but you should test
    #    this yourself by fitting mock data generated with write_a_model
    # To Do: let the Fe-peak elements track Fe in simple mode
    """
    ALFPY_HOME = os.environ['ALFPY_HOME']
    alfvar, prloarr, prhiarr, all_prior, optimize_res_x, run_de = model

    # Initialize log probability calculator
    use_keys = [k for k, (v1, v2) in tofit_params.items() if v1 == True]
    npar = len(use_keys)
    all_key_list = list(tofit_params.keys())
    log_prob_calculator = LogProbCalculator(alfvar, prloarr, prhiarr, all_prior, use_keys)
    if run == 'emcee':
        pool = mp.get_context("spawn").Pool(
            processes=ncpu, initializer=_init_worker, initargs=(log_prob_calculator,))
        with pool:
            # Initialize walkers
            pos_emcee_in = np.zeros(shape=(nwalkers, npar))
            prrange = [10, 10, 0.1, 0.1]
            for i in range(npar):
                if run_de and i < 4:
                    min_ = max(prloarr[i], np.array(optimize_res_x)[i] - prrange[i])
                    max_ = min(prhiarr[i], np.array(optimize_res_x)[i] + prrange[i])
                    pos_emcee_in[:, i] = np.array([np.random.uniform(min_, max_, nwalkers)])
                else:
                    tem_prior = np.take(all_prior, all_key_list.index(use_keys[i]))
                    print(tem_prior.range[0], tem_prior.range[1])
                    pos_emcee_in[:, i] = np.array([np.random.uniform(tem_prior.range[0], tem_prior.range[1], nwalkers)])

            print(pos_emcee_in[0])
            print(f'Initializing emcee with nwalkers={nwalkers}, npar={npar}')
            print(f"Fitting parameters: {use_keys}")
            print(f"Shape of initialized positions: {pos_emcee_in.shape}")
            print(f"Mean positions across walkers: {np.nanmean(pos_emcee_in, axis=0)}")
            print(f"Min positions across walkers: {np.nanmin(pos_emcee_in, axis=0)}")
            print(f"Max positions across walkers: {np.nanmax(pos_emcee_in, axis=0)}")
            print("try func on mean initial values:",
                  func(alfvar, np.nanmean(pos_emcee_in, axis=0),
                       use_keys,
                       prhiarr=prhiarr,
                       prloarr=prloarr))

            tstart = time.time()
            sampler = emcee.EnsembleSampler(
            nwalkers, npar, _log_prob_worker, pool=pool)
            sampler.run_mcmc(pos_emcee_in, nburn + nmcmc, progress=True)

            print(f'mean acc fraction {np.nanmean(sampler.acceptance_fraction):.3f}')
            ndur = time.time() - tstart
            print(f'\n Total time for emcee {ndur/60:.2f}min')
            res = sampler.get_chain(discard = nburn)
            prob = sampler.get_log_prob(discard = nburn)

            pickle.dump(res, open(f'{ALFPY_HOME}results_emcee/res_emcee_{filename}_{tag}.p', "wb"))
            pickle.dump(prob, open(f'{ALFPY_HOME}results_emcee/prob_emcee_{filename}_{tag}.p', "wb"))
            print('writing emcee outputs and M/L columns...')
            _save_emcee_text_outputs(ALFPY_HOME, filename, tag, alfvar, use_keys, res, prob, prloarr, prhiarr, nwalkers, nburn, nmcmc, ncpu, np.nanmean(sampler.acceptance_fraction), ndur, pool=pool)
            print('EMCEE run complete.')
            pool.close()
    # ---------------------------------------------------------------- #
    elif run == 'dynesty':
        # Run dynesty
        tstart = time.time()
        print(f"Fitting parameters: {use_keys}")
        if pool_type == 'multiprocessing' and ncpu > 1:
            with dynesty.pool.Pool(ncpu,
                                   log_prob_calculator.log_prob_nested,
                                   log_prob_calculator.prior_transform) as pool:
                dsampler = dynesty.NestedSampler(pool.loglike, pool.prior_transform,
                                                 pool=pool, ndim=npar, nlive=int(50*npar),
                                                 sample='rslice', bootstrap=0)
                dsampler.run_nested(dlogz=0.5)
        else:
            dsampler = dynesty.NestedSampler(log_prob_calculator.log_prob_nested,
                                             log_prob_calculator.prior_transform,
                                             ndim=npar, nlive=int(50*npar),
                                             sample='rslice', bootstrap=0)
            dsampler.run_nested(dlogz=0.5)
        ndur = time.time() - tstart
        print(f'\n Total time for dynesty {ndur/60./60.:.2f}hrs')

        # Save results
        results = dsampler.results
        pickle.dump(results, open(f'{ALFPY_HOME}results_dynesty/res_dynesty_{filename}_{tag}.p', "wb"))
        print('Dynesty run complete.')

        # ---- post process ---- #
        if nested_post_process:
            results = pickle.load(open(f'{ALFPY_HOME}results_dynesty/res_dynesty_{filename}_{tag}.p', "rb" ))
            calm2l_dynesty(results, alfvar, use_keys=use_keys,
                               outname=f"{filename}_{tag}")


# -------------------------------- #
# ---- command line arguments ---- #
# -------------------------------- #


if __name__ == "__main__":
    argv_l = sys.argv
    n_argv = len(argv_l)
    filename = argv_l[1]
    tag = argv_l[2] if n_argv >= 3 else ''

    pool_type = "multiprocessing"
    model = build_alf_model(filename, tag, pool_type=pool_type)
    alf(filename, tag,
        run = "dynesty",
        pool_type = pool_type,
        ncpu=8,
        model=model)
