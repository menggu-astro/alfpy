import os, sys, pickle, numpy as np
import matplotlib.pyplot as plt
import emcee, time
import dynesty
from tofit_parameters import tofit_params
from func import func
from str2arr import fill_param
from post_process import calm2l_dynesty
from alf_build_model import setup_pool
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
        ln_prior = self.lnprior(posarr)
        if not np.isfinite(ln_prior):
            return -np.inf
        res_ = func(self.alfvar, 
                    posarr, 
                    usekeys=self.keys, 
                    prhiarr=self.prhiarr,
                    prloarr=self.prloarr)
        return ln_prior - 0.5 * res_


    def prior_transform(self, unit_coords):
        """Transform unit coordinates to prior ranges for dynesty."""
        all_key_list = list(tofit_params.keys())
        res_ = np.array([self.all_prior[all_key_list.index(ikey)].unit_transform(unit_coords[i]) for i, ikey in enumerate(self.keys)])
        return res_


    def lnprior(self, in_arr):
        """
        Log-prior function for dynesty.
        - INPUT: npar arr (same length as use_keys)
        - all_priors: priors for all 46 parameter
        """
        full_arr = fill_param(in_arr, self.keys)
        lnp = 0.0
        for i in range(len(self.all_prior)):
            lnp += self.all_prior[i].lnp(full_arr[i])
        #lnp = sum([self.all_prior[i_].lnp(iarr_) for i_, iarr_ in enumerate(full_arr)])
        return lnp if np.isfinite(lnp) else -np.inf
        #return 0 if np.isfinite(lnp) else lnp


# -------------------------------------------------------- #
def alf(filename, 
        tag='', 
        nwalkers = 128, 
        nburn = 500, 
        nmcmc = 100,
        run='dynesty', 
        pool_type='multiprocessing', 
        emcee_save_chains = False, 
        ncpu=1, 
        nested_post_process=False):
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
    pickle_model_name = f"{ALFPY_HOME}alfvar_models/alfvar_model_{filename}_{tag}.p"

    # Load model
    try:
        print(f"loading pickle file {pickle_model_name}")
        alfvar, prloarr, prhiarr, all_prior, optimize_res_x = pickle.load(open(pickle_model_name, "rb" ))
    except:
        print(f'Cannot find model_arr at {pickle_model_name}. Please run alf_build_model first.')
        return 

    # Initialize log probability calculator
    use_keys = [k for k, (v1, v2) in tofit_params.items() if v1 == True]
    npar = len(use_keys)
    all_key_list = list(tofit_params.keys())
    log_prob_calculator = LogProbCalculator(alfvar, prloarr, prhiarr, all_prior, use_keys)
    if run == 'emcee' or run == 'emcee_test':
        pool = mp.get_context("spawn").Pool(
            processes=ncpu, initializer=_init_worker, initargs=(log_prob_calculator,))
        with pool:
            # Initialize walkers
            pos_emcee_in = np.zeros(shape=(nwalkers, npar))
            prrange = [10, 10, 0.1, 0.1]
            for i in range(npar):
                if i < 4:
                    min_ = max(prloarr[i], np.array(optimize_res_x)[i] - prrange[i])
                    max_ = min(prhiarr[i], np.array(optimize_res_x)[i] + prrange[i])
                    pos_emcee_in[:, i] = np.array([np.random.uniform(min_, max_, nwalkers)])                
                else:
                    tem_prior = np.take(all_prior, all_key_list.index(use_keys[i]))
                    print(tem_prior.range[0], tem_prior.range[1])
                    pos_emcee_in[:, i] = np.array([np.random.uniform(tem_prior.range[0], tem_prior.range[1], nwalkers)])
                
            print(pos_emcee_in[0])
            print(f'Initializing emcee with nwalkers={nwalkers}, npar={npar}')
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
            if emcee_save_chains:
                backends_fname = f"{ALFPY_HOME}results_emcee/backend_{filename}_{tag}.p"
                backend = emcee.backends.HDFBackend(backends_fname)
                backend.reset(nwalkers, npar)

            # ---------------------------------------------------------------- #
            if run == 'emcee':
                # Run emcee
                sampler = emcee.EnsembleSampler(
                nwalkers, npar, _log_prob_worker, pool=pool)
                sampler.run_mcmc(pos_emcee_in, nburn + nmcmc, progress=True)

                # Save results
                print(f'mean acc fraction {np.nanmean(sampler.acceptance_fraction):.3f}')
                ndur = time.time() - tstart
                print(f'\n Total time for emcee {ndur/60:.2f}min')
                res = sampler.get_chain(discard = nburn) 
                prob = sampler.get_log_prob(discard = nburn)

            # ---------------------------------------------------------------- #
            elif run == 'emcee_test':
                sampler = emcee.EnsembleSampler(nwalkers, npar, log_prob_calculator.log_prob,
                                            moves=[emcee.moves.StretchMove(a=1.5)],
                                            backend=backend)
                old_tau = np.inf
                converged = False
                for j, sample in enumerate(sampler.sample(pos_emcee_in, iterations=60000, progress=True)):
                    it = j+1
                    if it%100: continue
                    ndur = (time.time() - tstart)/60
                    # use the tau (without discarding) to determine how much to remove
                    tau = sampler.get_autocorr_time(discard=int(np.max(old_tau)) if \
                                                np.all(np.isfinite(old_tau)) else 0,
                                                tol=0) 

                    print(f'iter = {it}, ' +
                      f"tau = {np.max(tau):.0f}, " +
                      f"acceptance fraction = {sampler.acceptance_fraction.mean():.2f}, " +
                      f"dtau = {np.max((tau-old_tau)/tau):.2f}, " +
                      f"it/tau = {np.min(it/tau):.1f}, "+
                      f"time={ndur:.2f}min",
                      flush=True)

                    converged = np.all(tau * 20 < it)
                    converged &= np.all((tau-old_tau)/tau < 0.01)
                    old_tau = tau

                    if converged: break
                    if it % 1000:  continue
                samples = sampler.get_chain(thin=int(np.max(tau) / 2))
                fig, axes = plt.subplots(npar, figsize=(10, 20), sharex=True)
                for i in range(samples.shape[2]):
                    ax = axes[i]
                    ax.plot(samples[:, :, i], "k", alpha=0.3)
                    ax.set_ylabel(use_keys[i])
                    ax.yaxis.set_label_coords(-0.1, 0.5)
                axes[-1].set_xlabel("step number")
                fig.savefig(f'{ALFPY_HOME}results_emcee/check_chains_{filename}_{tag}.png',
                            dpi=100, bbox_inches="tight")
                fig.clf()
                del fig

                ndur = time.time() - tstart
                print('\n Total time for emcee {:.2f}min'.format(ndur/60))
                prob = sampler.get_log_prob(discard=int(5 * np.max(tau)), thin=int(np.max(tau) / 2))
                res = sampler.get_chain(discard=int(5 * np.max(tau)), thin=int(np.max(tau) / 2))



            pickle.dump(res, open(f'{ALFPY_HOME}results_emcee/res_emcee_{filename}_{tag}.p', "wb"))
            pickle.dump(prob, open(f'{ALFPY_HOME}results_emcee/prob_emcee_{filename}_{tag}.p', "wb"))
            print('EMCEE run complete.')
            best_params = res[np.where(prob == prob.max())][0]
            _, best_mspec = func(alfvar, best_params, use_keys, funit=True)
            np.savetxt(f'{ALFPY_HOME}results_emcee/bestspec_{filename}_{tag}.dat',
                       np.transpose(best_mspec), 
                       delimiter="     ", 
                       fmt='   %12.4f   %12.4E   %12.4E   %12.4E   %12.4E   %12.4E')
            pool.close()
    # ---------------------------------------------------------------- #
    elif run == 'dynesty':
        # Run dynesty
        tstart = time.time()
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
    tag = argv_l[2] if n_argv >= 2 else ''
    # pool_type: multiprocessing or emcee
    alf(filename, tag, 
        run = "dynesty", 
        pool_type = "multiprocessing", 
        ncpu=8)
