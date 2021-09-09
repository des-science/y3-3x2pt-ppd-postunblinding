import numpy as np
import numba
from astropy.io import fits
import os
import utils_6x2 as util
from utils_ppd import get_realizations, get_resampled_pvalue, load_chains #get_data_obs
import matplotlib as mpl
import matplotlib.pyplot as plt
from utils_ppd_plots import print_pval
import scipy.stats
import multiprocessing
from scipy.special import gammainc
from tqdm.auto import trange, tqdm
import functools
import multiprocessing

def _map_f(args):
    f, i = args
    return f(i)

def redef_map(func, iter, ordered=True):

    pool = multiprocessing.Pool()

    inputs = ((func,i) for i in iter) #use a generator, so that nothing is computed before it's needed :)

    try :
        n = len(iter)
    except TypeError : # if iter is a generator
        n = None

    res_list = []

    if ordered:
        pool_map = pool.imap
    else:
        pool_map = pool.imap_unordered

    with tqdm(total=n, desc='# castor.parallel.map') as pbar:
        for res in pool_map(_map_f, inputs):
            try :
                pbar.update()
                res_list.append(res)
            except KeyboardInterrupt:
                pool.terminate()

    pool.close()
    pool.join()

    return res_list
#

gammainc_funcs = [functools.partial(scipy.special.gammainc, k/2.) for k in range(600)]

def draw_conditional_multivariate_normal(mu, cov, a2, ind2, size):
    # Using notations from https://en.wikipedia.org/wiki/Multivariate_normal_distribution#Conditional_distributions
    import scipy.stats
    
    ind1 = np.setdiff1d(np.arange(len(mu)), ind2)
    sig11 = cov[ind1,:][:,ind1]
    sig12 = cov[ind1,:][:,ind2]
    sig21 = cov[ind2,:][:,ind1]
    sig22inv = np.linalg.inv(cov[ind2,:][:,ind2])

    mubar = mu[ind1] + np.dot(sig12,np.dot(sig22inv,(a2-mu[ind2])))
    sigbar = sig11 - np.dot(sig12,np.dot(sig22inv,sig21))
    
    return np.random.multivariate_normal(mean=mubar, cov=sigbar, size=size)


# @numba.jit(nopython=True)
# def _get_p_U_y(y, p_U):
#     return np.dot(y, p_U)

@numba.jit(nopython=True)
def dot(A,B):
    return np.dot(A,B)

@numba.jit(nopython=True, parallel=True)
def _3D_2D_dot(a,b):
    c = np.zeros((a.shape[0], a.shape[1], b.shape[1]))
    for i in numba.prange(len(c)):
        c[i] = np.dot(a[i],b)
    return c

# @numba.jit()
# def get_std_multivariate_normal(n,m):
#     res = np.zeros((n,m))
#     for i in range(n):
#         for j in range(m):
#             res[i,j] = np.random.normal()
#     return np.ascontiguousarray(res)

@numba.jit(nopython=True)
def get_std_multivariate_normal_2d(n,m):
    res = np.zeros((n,m))
    for i in range(n):
        for j in range(m):
            res[i,j] = np.random.normal()
    return np.ascontiguousarray(res)

@numba.jit(nopython=True)
def get_std_multivariate_normal_1d(n):
    res = np.zeros(n)
    for i in range(n):
        res[i] = np.random.normal()
    return np.ascontiguousarray(res)

# @numba.jit(nopython=True)
# def symmatrix_sqrt(A):
#     eigenvals, u = np.linalg.eigh(A)
#     return np.multiply(u, np.sqrt(eigenvals))

@numba.jit(nopython=True)
def _symmatrix_sqrt_inv_1(A):
    return np.linalg.eigh(A)

def symmatrix_sqrt_inv(A):
    # eigenvals, u = np.linalg.eigh(A)
    eigenvals, u = _symmatrix_sqrt_inv_1(A)
    if np.any(eigenvals<=0.):
        emin = np.min(eigenvals[eigenvals>0.])
        print("Some ({}) eigenvalues are non-positive. Clipping to minimum positive one = {}".format(np.sum(eigenvals<=0.), emin))
        eigenvals = np.clip(eigenvals, emin, np.inf)
    return np.multiply(u, 1./np.sqrt(eigenvals))

@numba.jit(nopython=True, parallel=True)
def get_chi2_rsd_sub(p_U_means, p_U_dvs):
    n = len(p_U_means)
    m = len(p_U_dvs)
    w = np.zeros((n,m))
    for i in numba.prange(n):
        w_i = np.zeros(m)
        for j in range(m):
            w_i[j] = np.sum(np.square(p_U_means[i]-p_U_dvs[j]))
        w[i] = w_i
    return w

def get_chi2_rsd(_xs, _xth, _cov):
    # print(" - Number of sims DV = ", _xs.shape[0])
    # print(" - Number of chain DV = ", _xth.shape[0])
    # print(" - Dimension of DV = ", _xth.shape[1])
    assert _cov.shape[0]==_cov.shape[1]==_xs.shape[1]==_xth.shape[1]
    # eigenvals, u = np.linalg.eigh(_cov)
    # p_U = np.multiply(u, 1./np.sqrt(eigenvals))
    p_U = symmatrix_sqrt_inv(_cov)

    p_U_dvs = dot(_xs, p_U)
    p_U_means = dot(_xth, p_U)
    return get_chi2_rsd_sub(p_U_means, p_U_dvs)

@numba.jit(nopython=True, parallel=True)#nopython=True, parallel=True)
def get_chi2_one_sub(p_U_means, p_U_dvs):
    n = len(p_U_means)
    w = np.zeros(n)
    for i in numba.prange(n):
        w[i] = np.sum(np.square(p_U_means[i]-p_U_dvs[i]))
    return w

def get_chi2_one(_xs, _xth, _cov):
    # print(_xs.shape)
    # print(_xth.shape)
    # print(_cov.shape)
    assert _cov.shape[0]==_cov.shape[1]==_xs.shape[1]==_xth.shape[1]
    assert len(_xs)==len(_xth)
    p_U = symmatrix_sqrt_inv(_cov)
    p_U_dvs = dot(_xs, p_U)
    p_U_means = dot(_xth, p_U)
    return get_chi2_one_sub(p_U_means, p_U_dvs)

@numba.jit()
def _get_weighted_average(x,w):
    return np.sum(x*w)/np.sum(w)

@numba.jit(nopython=True, parallel=True)
def _get_2D_weighted_average(x,w):
    n, m = x.shape
    res = np.zeros(m)
    for i in numba.prange(m):
        res[i] = np.sum(x[:,i]*w[:,i])/np.sum(w[:,i])
    return res

@numba.jit()
def _hist(a, bins, range):
    return np.histogram(a, bins=bins, range=range) 

@numba.jit(nopython=True, parallel=True)
def _2D_clip(a, amax):
    n,m = a.shape
    b = np.zeros_like(a)
    for i in numba.prange(n):
        for j in range(m):
            b[i,j] = min(a[i,j], amax)
    return b

@numba.jit(nopython=True, parallel=True)
def _get_is_weight(chi2_rsd_, chi2_data_, weights_, clip=1.):
    n, m = chi2_rsd_.shape
    # print(chi2_rsd_.shape)
    # print(chi2_data_.shape)
    # print(weights_.shape)
    log_is_weights = np.zeros((n, m))
    for i in numba.prange(n):
        for j in range(m): 
            log_is_weights[i,j] = -0.5*(chi2_rsd_[i,j] - chi2_data_[i])
            
    for j in numba.prange(m): 
        max_j = np.max(log_is_weights[:,j])
        for i in range(n):
            log_is_weights[i,j] -= max_j
        
    is_weights = np.exp(log_is_weights)

    # reweight
    for i in numba.prange(n):
        for j in range(m): 
            is_weights[i,j] *= weights_[i]
    
    # normalize before clipping
    for j in numba.prange(m): 
        sum_j = np.sum(is_weights[:,j])
        for i in range(n):
            is_weights[i,j] /= sum_j

    # clip
    is_weights = _2D_clip(is_weights, clip)

    # normalize after posterior sample reweighting
    for j in numba.prange(m): 
        sum_j = np.sum(is_weights[:,j])
        for i in range(n):
            is_weights[i,j] /= sum_j

    return is_weights


@numba.jit(nopython=True, parallel=True)
def get_mubar_y_2_summed(theory_dprime, p_U_sigbar, resampled_data_d, theory_d, c11_ic22):
    c11_ic22Tp_U_sigbar = np.dot(c11_ic22.T, p_U_sigbar)
    theory_dprime_y = np.dot(theory_dprime, p_U_sigbar)
    n = len(resampled_data_d)
    m = len(theory_d)
    res = np.zeros((n,m))
    for i in numba.prange(n):
        for j in range(m):
            res[i,j] = np.sum(np.square(theory_dprime_y[j] + np.dot(resampled_data_d[i]-theory_d[j], c11_ic22Tp_U_sigbar)))
    return res


@numba.jit(nopython=True, parallel=True)
def _sub_get_chi2_resample_cond(nsamples, nchain, iL_u_sqrtcov_p_U, iL_cond_mui, iL_cond_muj, iL_mu1):
    chi2_ = np.zeros((nsamples, nchain))
    for j in numba.prange(nsamples):    
        # if j%100==0:
        #     print(j)
        chi2_j = np.zeros(nchain)
        perm = np.random.permutation(nchain)
        for i in range(nchain):
            iL_x = iL_u_sqrtcov_p_U[perm[i]] + iL_cond_mui[i] + iL_cond_muj[j] 
            chi2_j[i] = np.sum(np.square(iL_x-iL_mu1[i]))
        chi2_[j] = chi2_j
    return chi2_.T
        
def get_chi2_resample_cond(mu, cov, a2, ind2, cov11_for_chi2):
    # print(" - Number of sims DV = ", a2.shape[0])
    # print(" - Number of chain DV = ", mu.shape[0])
    # print(" - Dimension of DV (d+d') = ", mu.shape[1])
    # print(" - Dimension of DV (d) = ", a2.shape[1])
    assert mu.shape[1] == cov.shape[0] == cov.shape[1]
    assert a2.shape[1] == ind2.shape[0]
    nchain,_ = mu.shape
    nsamples,ndim2 = a2.shape
    
    # indices
    ind1 = np.setdiff1d(np.arange(cov.shape[0]), ind2)
    ndim1 = len(ind1)
    
    # get covariance matrices
    sig11 = cov[ind1,:][:,ind1]
    sig12 = cov[ind1,:][:,ind2]
    sig21 = cov[ind2,:][:,ind1]
    sig22inv = np.linalg.inv(cov[ind2,:][:,ind2])
    sigbar = sig11 - np.dot(sig12,np.dot(sig22inv,sig21))
    
    # get (inv) sqrt
    # L22bar = symmatrix_sqrt(sigbar) # sqrt of dprime_rep covariance with conditioning
    # seems that the line above returns weird results: probably because down below we perform dot products
    # where we sort out invert the order but symmatrix_sqrt doesn't return a symmetric sqrt while the one below does
    L22bar = np.ascontiguousarray(scipy.linalg.sqrtm(sigbar))  
    # iL11 = symmatrix_sqrt_inv(sig11) # inv sqrt of dprime_rep without conditioning
    iL11 = symmatrix_sqrt_inv(cov11_for_chi2) # inv sqrt of dprime_rep without conditioning
    
    mu1 = mu[:,ind1] # dprime theory
    mu2 = mu[:,ind2] # d theory
    
    # We use the fact that chi2 = (x-mu)*C-1*(x-mu) = || iL*(x-mu) ||^2 where iL = Cholesky of inverse cov
    iL_mu1 = dot(mu1, iL11) # iL*mu
    
    # Shifted mean with conditioning
    # i indices posterior samples/realizations
    # j indices simulated data vectors
    cond_mui = mu1 + dot(sig12,dot(sig22inv, -mu2.T)).T
    cond_muj = dot(sig12,dot(sig22inv, a2.T)).T
    
    iL_cond_mui = dot(cond_mui, iL11)
    iL_cond_muj = dot(cond_muj, iL11)
    
    # To sample drep, we note that we can sample a multivariate gaussian with x=Lu+mu where L is the sqrt
    # of the cov (or its Cholesky decompostion), u is a unit multivariate gaussian and mu the mean
    # To speed things up, we generate a batch of u's to use for everyone
    L22bar_iL11 = dot(L22bar, iL11)
    u = get_std_multivariate_normal_2d(nchain,ndim1)
    # Then we also need to multiply by iL11 to get chi2's as explained above
    iL_u_sqrtcov_p_U = dot(u, L22bar_iL11)
    
    # The big for loop over posterior samples and simulated DV's
    return _sub_get_chi2_resample_cond(nsamples, nchain, iL_u_sqrtcov_p_U, iL_cond_mui, iL_cond_muj, iL_mu1)

def get_resampled_pvalue_2d(comp_avg, weights, N=1000):
    print(comp_avg.shape)
    print(weights.shape)
    print(np.random.binomial(1, p=comp_avg).shape)
    print(np.average(np.random.binomial(1, p=comp_avg), weights=weights, axis=0).shape)
    _p = []
    for _ in trange(N):
        _p.append(np.average(np.random.binomial(1, p=comp_avg), weights=weights, axis=0))
    return np.mean(_p, axis=0)

def calibrate_pvals(path_ppd, path_chain, path_dv, RUN_NAME, RUN_NAME_PPD, DATAFILE, fiducial_dv,
                    data_sets_d, data_sets_dprime,
                    #dico_indices_dprime, dico_indices_d,
                    #theory_dprime, realization_dprime, true_dprime, chain_weights,
                    #theory_d, true_d,
                    N, title, pval_data, use_logit=False, legend_loc=1, yscale='linear', sample_from='fiducial_dv', use_pm=True, clip_is=0.1, ndraws=1, experimental_is=False, chunks=0, size_chunk=None,
                    pvals_subsets={_x:None for _x in ['xip', 'xim', '1x2', 'gammat', 'wtheta', '2x2']}, get_zbin_pcal=False, get_zbin_pair_pcal=False, additional_subsets=None):

    # File paths
    dv_file = os.path.join(path_dv, DATAFILE)
    fiducial_dv_file = os.path.join(path_dv, fiducial_dv)
    ppd_chain_file = os.path.join(path_ppd, RUN_NAME_PPD, 'ppd_chain_'+RUN_NAME+'_'+RUN_NAME_PPD+'.txt')
    chain_file = os.path.join(path_chain, 'chain_'+RUN_NAME+'.txt')

    # Chains
    print("\n###############################")
    print("## Loading chains #############")
    print("###############################")
    # ppd_chain, _ = util.load_cosmosis_chain(ppd_chain_file, verbose=False, params_lambda=lambda _x:True, read_nsample=False)
    # _mcmc_chain, weights = util.load_cosmosis_chain(chain_file, verbose=False, params_lambda=lambda _x:True, read_nsample=False)
    ppd_chain, _mcmc_chain, weights, ppd_chain_file = load_chains(path_ppd, path_chain, RUN_NAME, RUN_NAME_PPD, verbose=False, chunks=chunks, size_chunk=size_chunk)

    # d and dprime indices
    print("\n###############################")
    print("## Loading scale/bin cuts #####")
    print("###############################")
    dico_indices_d = util.get_scale_cuts_new(ppd_chain_file, dv_file, print_out=True, return_zbins=True, file_mode='r',
                                       like_module='2pt_d_like', data_sets_realizations=data_sets_d, num_observables_tot=4)
    print("")
    dico_indices_dprime = util.get_scale_cuts_new(ppd_chain_file, dv_file, print_out=True, return_zbins=True, file_mode='r',
                                       like_module='2pt_dprime_like', data_sets_realizations=data_sets_dprime, num_observables_tot=4)                
 
    idx_full_dprime = np.concatenate([np.concatenate([dico_indices_dprime[_obs][ij]['idx_full'] for ij in dico_indices_dprime[_obs]]) for _obs in dico_indices_dprime.keys() if _obs in ['xip','xim','gammat','wtheta']])
    idx_full_d = np.concatenate([np.concatenate([dico_indices_d[_obs][ij]['idx_full'] for ij in dico_indices_d[_obs]]) for _obs in dico_indices_d.keys() if _obs in ['xip','xim','gammat','wtheta']])
    idx_full_d_plus_dprime = np.concatenate([idx_full_d, idx_full_dprime])


    # Indices for subsets
    subsets_obs = {'xip':['xip'], 'xim':['xim'], '1x2':['xip','xim'], 'gammat':['gammat'], 'wtheta':['wtheta'], '2x2':['gammat','wtheta']}
    subsets = list(np.intersect1d(list(subsets_obs.keys()), list(pvals_subsets.keys())))
    subsets_idx_full_dprime = {}
    subsets_idx_sub_dprime = {}
    for subset in ['xip', 'xim', '1x2', 'gammat', 'wtheta', '2x2']:
        if subset in subsets:
            try:
                subsets_idx_full_dprime[subset] = np.concatenate([np.concatenate([dico_indices_dprime[_obs][ij]['idx_full'] for ij in dico_indices_dprime[_obs].keys()]) for _obs in dico_indices_dprime.keys() if _obs in subsets_obs[subset]])
            except:
                subsets.remove(subset)
                continue
            assert np.all(np.isin(subsets_idx_full_dprime[subset], idx_full_dprime))
            assert len(subsets_idx_full_dprime[subset])>0
            subsets_idx_sub_dprime[subset] = np.intersect1d(subsets_idx_full_dprime[subset], idx_full_dprime, return_indices=True)[2]
    # print("\nUsing subsets", subsets)

    # if get_zbin_pair_pcal:
    #     for _obs in dico_indices_dprime.keys():
    #         assert _obs in subsets_idx_full_dprime.keys()
    #         for ij in dico_indices_dprime[_obs].keys():
    #             _i,_j=ij
    #             new_subset = _obs +'_{}_{}'.format(_i,_j)
    #             subsets_idx_full_dprime[new_subset] = dico_indices_dprime[_obs][ij]['idx_full']
    #             subsets_idx_sub_dprime[new_subset] = np.intersect1d(subsets_idx_full_dprime[new_subset], idx_full_dprime, return_indices=True)[2]
    #             subsets.append(new_subset)

    def np_append(a,b):
        a = np.concatenate([a,b])

    if get_zbin_pair_pcal or get_zbin_pcal:
        for _obs in dico_indices_dprime.keys():
            assert _obs in subsets_idx_full_dprime.keys()
            for ij in dico_indices_dprime[_obs].keys():
                _i,_j=ij
                new_subset = _obs +'_{}_{}'.format(_i,_j)
                subsets_idx_full_dprime[new_subset] = dico_indices_dprime[_obs][ij]['idx_full']
                subsets_idx_sub_dprime[new_subset] = np.intersect1d(subsets_idx_full_dprime[new_subset], idx_full_dprime, return_indices=True)[2]
                if get_zbin_pair_pcal:
                    subsets.append(new_subset)
                if get_zbin_pcal:
                    new_subset_i = _obs +'_{}_all'.format(_i)
                    new_subset_j = _obs +'_all_{}'.format(_j)
                    for new_subset_ij in [new_subset_i, new_subset_j]:
                        if new_subset_ij in subsets_idx_full_dprime.keys():
                            np_append(subsets_idx_full_dprime[new_subset_ij], subsets_idx_full_dprime[new_subset])
                            np_append(subsets_idx_sub_dprime[new_subset_ij], subsets_idx_sub_dprime[new_subset])
                        else:
                            subsets_idx_full_dprime[new_subset_ij] = subsets_idx_full_dprime[new_subset]
                            subsets_idx_sub_dprime[new_subset_ij] = subsets_idx_sub_dprime[new_subset]
                    if new_subset_i not in subsets:
                        subsets.append(new_subset_i)
                    if new_subset_j not in subsets:
                        subsets.append(new_subset_j)

    print("\nUsing subsets", subsets)

    if additional_subsets is not None:
        for k,v in additional_subsets.items():
            subsets_idx_full_dprime[k] = v['full']
            subsets_idx_sub_dprime[k] = v['sub']
            subsets.append(k)


    if np.array_equal(idx_full_dprime, idx_full_d):
        print("\nUsing same indices for d and dprime -> conditioning turned off")
        conditioning = False
    else:
        assert len(np.intersect1d(idx_full_d, idx_full_dprime))==0, "idx_full_d and idx_full_dprime are different but overlap, something is wrong"
        print("Using non-overlapping indices for d and dprime -> conditioning turned on")
        conditioning = True

    
    print("\n###############################")
    print("## Loading realizations #######")
    print("###############################")
    ppd_output_file_basename = os.path.join(path_ppd, RUN_NAME_PPD, 'ppd_'+RUN_NAME+'_'+RUN_NAME_PPD)

    if chunks==0:
        # theory_dprime = get_realizations(ppd_output_file_basename+'_dprime_theory.txt', chain_file)
        # realization_dprime = get_realizations(ppd_output_file_basename+'_dprime_real.txt', chain_file) # <-- needed for now to recompute chi2 without pm-cov
        # theory_d = get_realizations(ppd_output_file_basename+'_d_theory.txt', chain_file)
        theory_dprime = get_realizations(ppd_output_file_basename+'_dprime_theory.npy', ppd_chain)
        realization_dprime = get_realizations(ppd_output_file_basename+'_dprime_real.npy', ppd_chain) # <-- needed for now to recompute chi2 without pm-cov
        theory_d = get_realizations(ppd_output_file_basename+'_d_theory.npy', ppd_chain)
        # true_d = get_data_obs(ppd_output_file_basename+'_d_theory.txt')
        # true_dprime = get_data_obs(ppd_output_file_basename+'_dprime_theory.txt')
        true_d = np.load(ppd_output_file_basename+'_d_true.npy')
        true_dprime = np.load(ppd_output_file_basename+'_dprime_true.npy')
    else:
        # Theory and PPD realizations
        theory_dprime_chunks = []
        realization_dprime_chunks = []
        theory_d_chunks = []
        for ichunk in trange(chunks):
            theory_dprime_chunks.append(get_realizations(ppd_output_file_basename+'_{}_dprime_theory.npy'.format(ichunk), ppd_chain, ichunk=ichunk, size_chunk=size_chunk))
            realization_dprime_chunks.append(get_realizations(ppd_output_file_basename+'_{}_dprime_real.npy'.format(ichunk), ppd_chain, ichunk=ichunk, size_chunk=size_chunk))
            theory_d_chunks.append(get_realizations(ppd_output_file_basename+'_{}_d_theory.npy'.format(ichunk), ppd_chain, ichunk=ichunk, size_chunk=size_chunk))
        theory_dprime = np.concatenate(theory_dprime_chunks)
        realization_dprime = np.concatenate(realization_dprime_chunks)
        theory_d = np.concatenate(theory_d_chunks)
        # True data
        # true_d = get_data_obs(ppd_output_file_basename+'_0_d_theory.txt')
        # true_dprime = get_data_obs(ppd_output_file_basename+'_0_dprime_theory.txt')
        true_d = np.load(ppd_output_file_basename+'_0_d_true.npy')
        true_dprime = np.load(ppd_output_file_basename+'_0_dprime_true.npy')

    # Sample full data vector
    print("\n###############################")
    print("## Sampling data ##############")
    print("###############################")
    data_dv = fits.open(dv_file)
    full_cov = (data_dv['COVMAT'].data[:,:][:,:]).astype(float)    # Ami removed hard-coded 900,900 indices for RM
    cov_d = full_cov[:,idx_full_d][idx_full_d,:]
    cov_dprime = full_cov[:,idx_full_dprime][idx_full_dprime,:]

    fid_dv = fits.open(fiducial_dv_file)

    if use_pm:
        # if chunks>0:
        #     cov_d = np.load(ppd_output_file_basename+'_0_d_inv_cov_pm.npy'))
        #     cov_pm_d = np.linalg.inv(np.load(ppd_output_file_basename+'_0_d_inv_cov_pm.npy'))
        #     cov_pm_dprime = np.linalg.inv(np.load(ppd_output_file_basename+'_0_dprime_inv_cov_pm.npy'))
        #     cov_dprime = np.linalg.inv(np.load(ppd_output_file_basename+'_0_dprime_inv_cov_pm.npy'))
        #     # subsets_cov_pm_dprime = {}
        #     # for subset in subsets:
        #     #     subsets_cov_pm_dprime[subset] = np.linalg.inv(np.load(ppd_output_file_basename+'_0_{}_inv_cov_pm.npy'.format(subset)))
        # else:
        #     cov_pm_d = np.linalg.inv(np.load(ppd_output_file_basename+'_d_inv_cov_pm.npy'))
        #     cov_pm_dprime = np.linalg.inv(np.load(ppd_output_file_basename+'_dprime_inv_cov_pm.npy'))
        #     # subsets_cov_pm_dprime = {}
        #     # for subset in subsets:
        #     #     subsets_cov_pm_dprime[subset] = np.linalg.inv(np.load(ppd_output_file_basename+'_{}_inv_cov_pm.npy'.format(subset)))
        _chunk_suffix = '_0' if chunks>0 else ''

        icov_pm_d = np.load(ppd_output_file_basename+_chunk_suffix+'_d_inv_cov_pm.npy')
        cov_pm_d = np.linalg.inv(icov_pm_d)
        icov_pm_dprime = np.load(ppd_output_file_basename+_chunk_suffix+'_dprime_inv_cov_pm.npy')
        cov_pm_dprime = np.linalg.inv(icov_pm_dprime)

        icov_dprime = np.linalg.inv(cov_dprime)
        icov_pm_dprime_corr = icov_pm_dprime - icov_dprime
        subsets_cov_pm_dprime = {}

        for subset in subsets:
            _idx = subsets_idx_sub_dprime[subset]
            subsets_cov_pm_dprime[subset] = np.linalg.inv(np.linalg.inv(cov_dprime[:,_idx][_idx,:]) + icov_pm_dprime_corr[:,_idx][_idx,:])

    else:
        cov_pm_d = cov_d
        cov_pm_dprime = cov_dprime
        subsets_cov_pm_dprime = {}
        for subset in subsets:
            subsets_cov_pm_dprime[subset] = full_cov[:,subsets_idx_full_dprime[subset]][subsets_idx_full_dprime[subset],:]

    if sample_from=='fiducial_dv':
        print("Sampling from", fiducial_dv_file)
        full_dv = np.concatenate([fid_dv[_obs].data['VALUE'] for _obs in ['xip','xim','gammat','wtheta']])[idx_full_d_plus_dprime]
        resampled_data = np.random.multivariate_normal(mean=full_dv, cov=full_cov[idx_full_d_plus_dprime,:][:,idx_full_d_plus_dprime], size=N)

    elif sample_from=='best-fit':
        # resampled_data_d = resampled_data[:,idx_full_d]
        # resampled_data_d = np.random.multivariate_normal(mean=theory_d[np.argmax(_mcmc_chain['post'])], cov=cov_d, size=N) # using best-fit dv
        # chi2_rsd_d = get_chi2_rsd(np.ascontiguousarray(resampled_data_d), theory_d, cov_d)

        # i_bf = np.argmax(_mcmc_chain['post'])
        try:
            i_bf = np.argmax(_mcmc_chain['post'])
        except KeyError:
            i_bf = np.argmax(ppd_chain['post'])

        print("Sampling at best-fit with params")
        # comenting this for maglim unblinding:
        for key in _mcmc_chain.keys():
            if ("--" in key) and key.lower()==key and (_mcmc_chain[key][0] != _mcmc_chain[key][1]):
                print("{:20} = {:.3}".format(key, _mcmc_chain[key][i_bf]))
        if not conditioning:
            _mean = theory_d[i_bf]
            _cov = full_cov[:,idx_full_d][idx_full_d,:]
        else:
            _idx_fullddprime = np.concatenate([idx_full_d, idx_full_dprime])
            _mean = np.concatenate([theory_d[i_bf], theory_dprime[i_bf]]) 
            _cov = full_cov[:,_idx_fullddprime][_idx_fullddprime,:]
        resampled_data = np.random.multivariate_normal(mean=_mean, cov=_cov, size=N)

    print("\n###############################")
    print("## Compute chi2 for d #########")
    print("###############################")
    resampled_data_d = resampled_data[:,:len(idx_full_d)]
    # chi2_rsd_d = get_chi2_rsd(np.ascontiguousarray(resampled_data_d), theory_d, cov_d)
    if chunks==0:
        chi2_rsd_d = get_chi2_rsd(np.ascontiguousarray(resampled_data_d), theory_d, cov_pm_d)
    else:
        chi2_rsd_d = np.concatenate([get_chi2_rsd(np.ascontiguousarray(resampled_data_d), _th_d, cov_pm_d) for _th_d in tqdm(theory_d_chunks)])



    # if not conditioning:
    #     print("Full")
    #     chi2_rsd_dprime = chi2_rsd_d
    #     # subsets
    #     sub_chi2_rsd_dprime = {}
    #     for subset in subsets:
    #         print(subset)
    #         sub_idx = subsets_idx_sub_dprime[subset]
    #         sub_chi2_rsd_dprime[subset] = get_chi2_rsd(np.ascontiguousarray(resampled_data_d[:,sub_idx]), theory_dprime[:,sub_idx], subsets_cov_pm_dprime[subset])
    # else:
        # print("\n###############################")
        # print("## Compute chi2 for dprime ####")
        # print("###############################")
        # print("Full")
        # resampled_data_dprime = resampled_data[:,len(idx_full_d):]
        # # chi2_rsd_dprime = get_chi2_rsd(np.ascontiguousarray(resampled_data_dprime), theory_dprime, cov_dprime)
        # chi2_rsd_dprime = get_chi2_rsd(np.ascontiguousarray(resampled_data_dprime), theory_dprime, cov_pm_dprime)
        # # subsets
        # sub_chi2_rsd_dprime = {}
        # for subset in subsets:
        #     print(subset)
        #     sub_idx = subsets_idx_sub_dprime[subset]
        #     sub_chi2_rsd_dprime[subset] = get_chi2_rsd(np.ascontiguousarray(resampled_data_dprime[:,sub_idx]), theory_dprime[:,sub_idx], subsets_cov_pm_dprime[subset]) #[:,sub_idx][sub_idx,:]

    
    print("\n###############################")
    print("## Compute IS weights for d ###")
    print("###############################")
    # is_weights = _get_is_weight(chi2_rsd_d, ppd_chain['PPD--CHI2_D_DATA'], weights) <-- old version: using pre-computed chi2 for d (with pm-cov)
    # chi2_data_d = get_chi2_rsd(true_d[None,:], theory_d, cov_d)[:,0] # <-- new version: recomputing chi2 for d (without pm-cov so far...)
    # chi2_data_d = get_chi2_rsd(theory_d[None,i_bf], theory_d, cov_d)[:,0] # <-- test using bf to smooth IS...
    # is_weights = _get_is_weight(chi2_rsd_d, chi2_data_d, weights, clip=clip_is)

    if chunks==0:
        chi2_data_d = get_chi2_rsd(true_d[None,:], theory_d, cov_pm_d)[:,0] # <-- new version: recomputing chi2 for d (without pm-cov so far...)
    else:
        chi2_data_d = np.concatenate([get_chi2_rsd(true_d[None,:], _th_d, cov_pm_d)[:,0] for _th_d in tqdm(theory_d_chunks)])

    # return chi2_rsd_d, chi2_data_d

    _w_weights_positive = weights>0.
    is_weights = np.zeros_like(chi2_rsd_d)
    is_weights[_w_weights_positive,:] += _get_is_weight(chi2_rsd_d[_w_weights_positive,:], chi2_data_d[_w_weights_positive], weights[_w_weights_positive], clip=clip_is)
    print(is_weights.shape)

    # if experimental_is:
    #     i_bf_rsd = np.argmin(chi2_rsd_d, axis=0)
    #     chi2_rsd_d_bf = get_chi2_rsd(theory_d[i_bf_rsd,:], theory_d, cov_d)
    #     chi2_data_d_bf = get_chi2_rsd(theory_d[None,i_bf], theory_d, cov_d)[:,0]
    #     is_weights = _get_is_weight(chi2_rsd_d_bf, chi2_data_d_bf, weights, clip=clip_is)

    neff = np.sum(is_weights, axis=0)**2/np.sum(is_weights**2, axis=0)
    neff_mask = np.isfinite(neff)
    print("Neff = {:.1f} (median), 16th={:.1f}, 84th={:.1f} ({:d} were nan's)".format(np.median(neff[neff_mask]), np.percentile(neff[neff_mask], 16), np.percentile(neff[neff_mask], 84), len(neff_mask)-np.sum(neff_mask)))

    print("\n###############################")
    print("## Computing p-values #########")
    print("###############################")
    sub_pvals_rsd = {}
    if ndraws==1:
        if not conditioning:
            # Goodness-of-fit case, no conditioning
            chi2_rsd_dprime = chi2_rsd_d

            if chunks==0:
                chi2_real_dprime = get_chi2_one(realization_dprime, theory_dprime, cov_pm_dprime)
            else:
                chi2_real_dprime = np.concatenate([get_chi2_one(_r_dp, _th_dp, cov_pm_dprime) for _r_dp,_th_dp in tqdm(zip(realization_dprime_chunks,theory_dprime_chunks))])

            pvals_rsd = _get_2D_weighted_average((chi2_real_dprime[:,None]>chi2_rsd_dprime).astype(float), is_weights)

            for subset in subsets:
                print(subset)
                sub_idx = subsets_idx_sub_dprime[subset]
                sub_chi2_real_dprime = get_chi2_one(realization_dprime[:,sub_idx], theory_dprime[:,sub_idx], subsets_cov_pm_dprime[subset]) 
                # sub_pvals_rsd[subset] = _get_2D_weighted_average((sub_chi2_real_dprime[:,None]>sub_chi2_rsd_dprime[subset]).astype(float), is_weights)
                sub_chi2_rsd_dprime = get_chi2_rsd(np.ascontiguousarray(resampled_data_d[:,sub_idx]), theory_dprime[:,sub_idx], subsets_cov_pm_dprime[subset])
                sub_pvals_rsd[subset] = _get_2D_weighted_average((sub_chi2_real_dprime[:,None]>sub_chi2_rsd_dprime).astype(float), is_weights)

                if pvals_subsets.get(subset, None) is None:
                    sub_chi2_data_dprime = get_chi2_rsd(true_dprime[None,sub_idx], theory_dprime[:,sub_idx], subsets_cov_pm_dprime[subset])[:,0]
                    _p = _get_weighted_average((sub_chi2_real_dprime>sub_chi2_data_dprime).astype(float), weights)
                    print("Appending pval for", subset, _p)
                    pvals_subsets[subset] = _p
                # print(' -    sub_pvals_rsd = ', sub_pvals_rsd)
        else:
            # Consistency case, with conditioning, ie we draw dprime_rep from theory conditioned on simulated d
            print("Full")
            resampled_data_dprime = resampled_data[:,len(idx_full_d):]
            chi2_rsd_dprime = get_chi2_rsd(np.ascontiguousarray(resampled_data_dprime), theory_dprime, cov_pm_dprime)

            chi2_real_dprime = get_chi2_resample_cond(mu=np.concatenate([theory_d, theory_dprime], axis=1).astype(float), # stack theory DV's
                                                    cov=full_cov[:,idx_full_d_plus_dprime][idx_full_d_plus_dprime,:].astype(float), # d+dprime cov 
                                                    a2=resampled_data_d.astype(float), #
                                                    ind2=np.arange(len(idx_full_d)),
                                                    cov11_for_chi2=cov_pm_dprime)
            pvals_rsd = _get_2D_weighted_average((chi2_real_dprime>chi2_rsd_dprime).astype(float), is_weights)

            for subset in subsets:
                print(subset)
                sub_idx = subsets_idx_sub_dprime[subset]
                idx_full_d_plus_sub_dprime = np.concatenate([idx_full_d, idx_full_dprime[sub_idx]])
                print("get_chi2_resample_cond")
                sub_chi2_real_dprime = get_chi2_resample_cond(mu=np.concatenate([theory_d, theory_dprime[:,sub_idx]], axis=1).astype(float), # stack theory DV's
                                                    cov=full_cov[:,idx_full_d_plus_sub_dprime][idx_full_d_plus_sub_dprime,:].astype(float), # d+dprime cov 
                                                    a2=resampled_data_d.astype(float), #
                                                    ind2=np.arange(len(idx_full_d)),
                                                    # cov11_for_chi2=cov_pm_dprime[sub_idx,:][:,sub_idx])
                                                    cov11_for_chi2=subsets_cov_pm_dprime[subset])
                # sub_pvals_rsd[subset] = _get_2D_weighted_average((sub_chi2_real_dprime>sub_chi2_rsd_dprime[subset]).astype(float), is_weights)
                print("get_chi2_rsd")
                sub_chi2_rsd_dprime = get_chi2_rsd(np.ascontiguousarray(resampled_data_dprime[:,sub_idx]), theory_dprime[:,sub_idx], subsets_cov_pm_dprime[subset]) #[:,sub_idx][sub_idx,:]
                # if subset=='gammat_no56':
                #     return np.ascontiguousarray(resampled_data_dprime[:,sub_idx]), theory_dprime[:,sub_idx], subsets_cov_pm_dprime[subset]
                # print(subset)
                # print(sub_chi2_real_dprime)
                # print(sub_chi2_rsd_dprime)
                # print(sub_chi2_rsd_dprime)
                # print(subsets_cov_pm_dprime[subset])
                # print(np.linalg.slogdet(subsets_cov_pm_dprime[subset]))
                sub_pvals_rsd[subset] = _get_2D_weighted_average((sub_chi2_real_dprime>sub_chi2_rsd_dprime).astype(float), is_weights)

                if pvals_subsets.get(subset, None) is None:
                    sub_chi2_real_dprime = get_chi2_one(realization_dprime[:,sub_idx], theory_dprime[:,sub_idx], subsets_cov_pm_dprime[subset]) 
                    sub_chi2_data_dprime = get_chi2_rsd(true_dprime[None,sub_idx], theory_dprime[:,sub_idx], subsets_cov_pm_dprime[subset])[:,0]
                    _p = _get_weighted_average((sub_chi2_real_dprime>sub_chi2_data_dprime).astype(float), weights)
                    print("Appending pval for", subset, _p)
                    pvals_subsets[subset] = _p

    elif ndraws>1:
        if not conditioning:
            chi2_real_dprime = None
            # dprime are drawn from theory_d, so their chi2 wrt theory_d is chi2 distributed
            # scipy.stats.chi2.sf(k,x) is replaced by 1.-gammainc(k/2,x/2)
            # pvals_rsd = _get_2D_weighted_average(1.-gammainc(len(idx_full_dprime)/2., chi2_rsd_dprime/2.), is_weights)
            # c = 1.-gammainc(len(idx_full_dprime)/2., chi2_rsd_dprime/2.)
            # pvals_rsd = get_resampled_pvalue_2d(c, is_weights, N=ndraws)
            args = np.array_split(chi2_rsd_dprime/2., multiprocessing.cpu_count())
            res = redef_map(gammainc_funcs[len(idx_full_dprime)], args)
            c = 1. - np.concatenate(res)
            pvals_rsd = _get_2D_weighted_average(c, is_weights)
        else:
            _p = []
            c=None
            for _ in trange(ndraws):
                chi2_real_dprime = get_chi2_resample_cond(mu=np.concatenate([theory_d, theory_dprime], axis=1).astype(float), # stack theory DV's
                                                  cov=full_cov[:,idx_full_d_plus_dprime][idx_full_d_plus_dprime,:].astype(float), # d+dprime cov 
                                                  a2=resampled_data_d.astype(float), #
                                                  ind2=np.arange(len(idx_full_d)),
                                                  cov11_for_chi2=cov_pm_dprime)
                _p.append(_get_2D_weighted_average((chi2_real_dprime>chi2_rsd_dprime).astype(float), is_weights))
            pvals_rsd = np.mean(_p, axis=0)

    else:
        raise ValueError

    if False: # test using chi2 CDF instead of random draws
        if not conditioning:
            # dprime are drawn from theory_d, so their chi2 wrt theory_d is chi2 distributed
            # scipy.stats.chi2.sf(k,x) is replaced by 1.-gammainc(k/2,x/2)
            pvals_rsd = _get_2D_weighted_average(1.-gammainc(len(idx_full_dprime)/2., chi2_rsd_dprime/2.), is_weights)
        
        else:
            # dprime are drawn conditional distrib, which amounts to multivariate gaussian with moving means and modified but cst covariance
            cov_ddprime = full_cov[:,idx_full_d][idx_full_dprime,:]
            c11_ic22 = np.dot(cov_ddprime, np.linalg.inv(cov_d))
            sigbar = cov_dprime - np.dot(c11_ic22, cov_ddprime.T)
            eigenvals, u = np.linalg.eigh(sigbar)
            p_U_sigbar = np.multiply(u, 1./np.sqrt(eigenvals))
            # mubar = theory_dprime[None,:,:] + np.dot(resampled_data_d[:,None,:]-theory_d[None,:,:], c11_ic22.T)
            # mubar = theory_dprime[None,:,:] + _3D_2D_dot(resampled_data_d[:,None,:]-theory_d[None,:,:], c11_ic22.T)
            # mubar_y = _3D_2D_dot(mubar, p_U_sigbar)

            # theory_dprime_y = _get_p_U_y(theory_dprime, p_U_sigbar)
            # mubar_2_y = _3D_2D_dot(resampled_data_d[:,None,:]-theory_d[None,:,:], np.dot(c11_ic22.T, p_U_sigbar))
            # mubar_y = theory_dprime_y[None,:,:] + mubar_2_y
            # print(mubar_y.shape)
            # nc = np.sum(mubar_y**2, axis=2)
            # print(nc.shape)
            df = len(idx_full_dprime)

            nc = get_mubar_y_2_summed(theory_dprime, p_U_sigbar, resampled_data_d, theory_d, c11_ic22)
            print(nc.shape)
            
            _ncx2 = np.sum(np.dot(resampled_data_dprime, p_U_sigbar)**2, axis=1)
            print(_ncx2.shape)
            _p = scipy.stats.ncx2.sf(_ncx2[:,None], df=df, nc=nc)
            print(_p.shape)
            pvals_rsd = _get_2D_weighted_average(_p.astype(float).T, is_weights)
            # pvals_rsd = np.random.rand(*(is_weights.shape))
    
    print("\n###############################")
    print("## Calibrated p-values ########")
    print("###############################")
    def get_pcal(p_rsd, p_dat):
        _w = np.isfinite(p_rsd)
        # return float(np.sum(p_rsd[_w]<p_dat))/float(np.sum(_w))
        return float(np.sum(p_rsd[_w]<=p_dat))/float(np.sum(_w))

    pcal = get_pcal(pvals_rsd, pval_data)
    sub_pcals = {}
    sub_pcals['full'] = pcal
    print('     - Full         calibrated pval={:.4f} (raw pval={:.4f})'.format(pcal, pval_data))
    for subset in subsets:
        sub_pcals[subset] = get_pcal(sub_pvals_rsd[subset], pvals_subsets[subset])
        print('     - {:<8}     calibrated pval={:.4f} (raw pval={:.4f})'.format(subset, sub_pcals[subset], pvals_subsets[subset]))

    print("\n###############################")
    print("## Plotting ###################")
    print("###############################")
    plt.figure(figsize=(4,3))
    if use_logit:
        plt.hist(pvals_rsd, np.geomspace(use_logit[0],use_logit[1],101), color='dodgerblue', alpha=.5, label='Simulated DV PPD', density=True);
        plt.axvline(x=pval_data, ls='--', c='orangered', label='Data PPD '+print_pval(pval_data))
        plt.xlabel('$p$-value')
        plt.xlim(use_logit[0],use_logit[1])
        plt.xscale('log')
    else:
        counts, bins = _hist(pvals_rsd, 100, (0.,1.))
        centroids = (bins[1:] + bins[:-1]) / 2
        counts_, bins_, _ = plt.hist(centroids, bins=len(counts), weights=counts, range=(min(bins), max(bins)), color='dodgerblue', alpha=.5, label='Simulated DV PPD', density=True);
        plt.xlim(0,1)
        plt.xlabel('$p$-value')
        plt.axvline(x=pval_data, ls='--', c='orangered', label='Data PPD '+print_pval(pval_data))
    plt.plot([], [], ' ', label='Calibrated $p$={:.3f}'.format(pcal))
    plt.yscale(yscale)
    plt.yticks([])
    
    handles, labels = plt.gca().get_legend_handles_labels()
    order = [2,0,1]
    plt.legend([handles[idx] for idx in order],[labels[idx] for idx in order], frameon=False, loc=legend_loc)
    plt.savefig('figs/'+title+'_meta_pval.pdf', bbox_inches='tight')
    plt.savefig('figs/'+title+'_meta_pval.png', bbox_inches='tight', dpi=300)
    
    # if ndraws==1:
    #     return pvals_rsd, is_weights, neff, chi2_rsd_dprime, chi2_rsd_d, chi2_real_dprime, cov_pm_d, cov_pm_dprime, chi2_data_d
    # else:
    #     return pvals_rsd, is_weights, neff, chi2_rsd_dprime, chi2_rsd_d, chi2_real_dprime, cov_pm_d, cov_pm_dprime, c, chi2_data_d #, ppd_chain['PPD--CHI2_D_DATA'], idx_full_dprime, idx_full_d, theory_d, resampled_data_d, _p #pvals_rsd_2 chi2_real_dprime
    return pvals_rsd, sub_pvals_rsd, pcal, sub_pcals

# @numba.jit(nopython=True)
# def get_bin_edges(a, bins):
#     bin_edges = np.zeros((bins+1,), dtype=np.float64)
#     a_min = a.min()
#     a_max = a.max()
#     delta = (a_max - a_min) / bins
#     for i in range(bin_edges.shape[0]):
#         bin_edges[i] = a_min + i * delta

#     bin_edges[-1] = a_max  # Avoid roundoff error on last point
#     return bin_edges


# @numba.jit(nopython=True)
# def compute_bin(x, bin_edges):
#     # assuming uniform bins for now
#     n = bin_edges.shape[0] - 1
#     a_min = bin_edges[0]
#     a_max = bin_edges[-1]

#     # special case to mirror NumPy behavior for last bin
#     if x == a_max:
#         return n - 1 # a_max always in last bin

#     bin = int(n * (x - a_min) / (a_max - a_min))

#     if bin < 0 or bin >= n:
#         return None
#     else:
#         return bin


# @numba.jit(nopython=True)
# def numba_histogram(a, bins):
#     hist = np.zeros((bins,), dtype=np.intp)
#     bin_edges = get_bin_edges(a, bins)

#     for x in a.flat:
#         bin = compute_bin(x, bin_edges)
#         if bin is not None:
#             hist[int(bin)] += 1

#     return hist, bin_edges



# @numba.jit()
# def get_chi2_one_resample(p_U_xreal, p_U_xth, p_U):
#     # print(_xreal.shape)
#     # assert _xreal.shape == _xth.shape
    
#     #p_U_xreal = _get_p_U_y(_xreal, p_U)
#     #p_U_xth = _get_p_U_y(_xth, p_U)
    
#     chi2 = np.zeros(len(p_U_xreal))
#     for i in range(len(p_U_xreal)):
#         chi2[i] = np.sum(np.square(p_U_xreal[i]-p_U_xth[i]))
        
#     return chi2

# @numba.jit()
# def get_std_multivariate_normal(n,m):
#     res = np.zeros((n,m))
#     for i in range(n):
#         for j in range(m):
#             res[i,j] = np.random.normal()
#     return np.ascontiguousarray(res)

# @numba.jit()
# def get_chi2_resample_cond_sub(mu1, mu2, cov11, a2, ind2, ind1, sig12, sig22inv, sqrtcov):
#     nchain,ndim1 = mu1.shape
#     nsamples,ndim2 = a2.shape

#     ndim1 = len(ind1)

#     eigenvals, u = np.linalg.eigh(cov11)
#     p_U = np.multiply(u, 1./np.sqrt(eigenvals))
    
#     p_U_xth = _get_p_U_y(mu1, p_U)

#     chi2_ = np.zeros((nchain, nsamples))
#     for i in range(nsamples):
#         if i%10==0:
#             print(i)
#         mubar = np.zeros((nchain, ndim1))
#         for j in range(nchain):
#             mubar[j,:] = mu1[j] + np.dot(sig12,np.dot(sig22inv,(a2[i]-mu2[j])))
#         u = get_std_multivariate_normal(nchain,ndim1) #np.random.multivariate_normal(np.zeros(ndim1), np.eye(ndim1), size=(nchain,))
#         #t = np.dot(u, sqrtcov) + mubar
#         p_U_xreal = u + _get_p_U_y(mubar, p_U)

#         chi2_[:,i] = get_chi2_one_resample(p_U_xreal, p_U_xth, p_U)

#     return chi2_

# def get_chi2_resample_cond(mu, cov, a2, ind2):
#     print(mu.shape)
#     print(cov.shape)
#     print(a2.shape)
#     print(ind2.shape)
#     ind1 = np.setdiff1d(np.arange(cov.shape[0]), ind2)
#     sig11 = cov[ind1,:][:,ind1]
#     sig12 = cov[ind1,:][:,ind2]
#     sig21 = cov[ind2,:][:,ind1]
#     sig22inv = np.linalg.inv(cov[ind2,:][:,ind2])
#     sigbar = sig11 - np.dot(sig12,np.dot(sig22inv,sig21))
#     sqrtcov = np.ascontiguousarray(scipy.linalg.sqrtm(sigbar))
    
#     mu1 = mu[:,ind1]
#     mu2 = mu[:,ind2]
    
#     cov11 = cov[:,ind1][ind1,:]
    
#     return get_chi2_resample_cond_sub(mu1, mu2, cov11, a2, ind2, ind1, sig12, sig22inv, sqrtcov)