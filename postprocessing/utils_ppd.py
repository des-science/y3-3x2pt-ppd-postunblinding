import os
import numpy as np
import utils_6x2 as util
from tqdm.auto import tqdm, trange
from astropy.io import fits
from collections import OrderedDict


# def get_realizations(filename_realizations, filename_chain, return_match=False, ichunk=None):
#     print("Loading realizations from ", filename_realizations)
#     # Load chain
#     d_chain = np.loadtxt(filename_chain)
    
#     # Load realization file
#     d_real = np.genfromtxt(filename_realizations, skip_header=1)
    
#     # Check Om and As match between both 
#     x = d_chain[:,[0,4]]
#     y = d_real[:,:2]
#     if ichunk is not None:
#         x = x[ichunk*len(y):(ichunk+1)*len(y)]
#     np.testing.assert_array_almost_equal(x, y)
    
#     print(" - data vector size =", d_real.shape[1]-2)
#     print(" - number of realizations =", d_real.shape[0])
#     return d_real[:,2:]

def get_realizations(filename_realizations, ppd_chain, return_match=False, ichunk=None, size_chunk=None):
    if not ichunk:
        print("Loading realizations from ", filename_realizations)
    # Load chain
    #d_chain = np.loadtxt(filename_chain)
    
    # Load realization file
    # d_real = np.genfromtxt(filename_realizations, skip_header=1)
    d_real = np.load(filename_realizations)
    
    # Check Om and As match between both 
    # x = d_chain[:,[0,4]]
    x = np.array([ppd_chain['cosmological_parameters--omega_m'], ppd_chain['cosmological_parameters--a_s']]).T
    y = d_real[:,:2]
    if ichunk is not None:
        assert size_chunk is not None
        x = x[ichunk*size_chunk:(ichunk+1)*size_chunk]
    np.testing.assert_array_almost_equal(x, y)
    
    if not ichunk:
        print(" - data vector size =", d_real.shape[1]-2)
        print(" - number of realizations =", d_real.shape[0])
    return d_real[:,2:]
    

# def get_data_obs(filename_realizations):
#     with open(filename_realizations) as f:
#         line = f.readline()
#         d_obs = np.fromstring(line, sep = ' ')
#     return d_obs

# def get_data_obs(filename_realizations):
#     return np.loadtxt(filename_realizations.replace('real'))

def get_resampled_pvalue(comp_avg, weights, N=1000):
    _p = []
    for _ in range(N):
        _p.append(np.average(np.random.binomial(1, p=comp_avg), weights=weights))
    _p = np.array(_p)
    return np.percentile(_p, 16), np.percentile(_p, 50), np.percentile(_p, 84)

def load_chains(path_ppd, path_chain, RUN_NAME, RUN_NAME_PPD, chunks=0, size_chunk=None, verbose=True):
    if chunks==0:
        # MCMC chain
        chain_file = os.path.join(path_chain, 'chain_'+RUN_NAME+'.txt')
        _mcmc_chain, weights = util.load_cosmosis_chain(chain_file, verbose=verbose, params_lambda=lambda _x:True, read_nsample=False)
        # PPD chain
        ppd_chain_file = os.path.join(path_ppd, RUN_NAME_PPD, 'ppd_chain_'+RUN_NAME+'_'+RUN_NAME_PPD+'.txt')
        ppd_chain, _ = util.load_cosmosis_chain(ppd_chain_file, verbose=verbose, params_lambda=lambda _x:True, read_nsample=False)
    else:
        # PPD chain
        ppd_chain_list = []
        for ichunk in trange(chunks):
            ppd_chain_file = os.path.join(path_ppd, RUN_NAME_PPD, 'ppd_chain_'+RUN_NAME+'_'+RUN_NAME_PPD+'_{}.txt'.format(ichunk))
            if os.path.exists(ppd_chain_file):
                if os.path.getsize(ppd_chain_file):
                    ppd_chain_sub, _w = util.load_cosmosis_chain(ppd_chain_file, verbose=(ichunk==0), params_lambda=lambda _x:True, read_nsample=False)
                    ppd_chain_list.append(ppd_chain_sub)
                    # print(ichunk)
        ppd_chain = OrderedDict()
        for k in ppd_chain_list[0].keys():
            ppd_chain[k] = np.concatenate([ppd_chain_sub[k] for ppd_chain_sub in ppd_chain_list])
        # MCMC chain
        chain_file = os.path.join(path_chain, 'chain_'+RUN_NAME+'.txt')
        _mcmc_chain, weights = util.load_cosmosis_chain(chain_file, verbose=verbose, params_lambda=lambda _x:True, read_nsample=False)
        # _n = chunks * len(_w)
        # _mcmc_chain = {_k:_mcmc_chain_[_k][:_n] for _k in _mcmc_chain_.keys()}
        # weights = weights_[:_n]
        print("Using {} chunks of size {} ({}) for a total of {} posterior samples".format(chunks, [len(_p[k]) for _p in ppd_chain_list], size_chunk, len(weights)))
    return ppd_chain, _mcmc_chain, weights, ppd_chain_file

def load_run(path_ppd, path_chain, path_dv,
             #RUN_NAME, DATAFILE, SCALE_CUTS, DEMODEL,
             RUN_NAME, RUN_NAME_PPD, DATAFILE,
             like_module='2pt_dprime_like', data_sets=['xip','xim','wtheta','gammat'],
             get_pval_only=False, verbose=True,
             get_pval_all_obs=True, get_cov_pm_dprime=False, chunks=0, size_chunk=None, load_dvs=True, Nresamplepval=1000
            ):
    
    if get_pval_only:
        verbose=False
    
    # ppd_realization_file = os.path.join(path_ppd, RUN_NAME, "ppd_realizations_{}_{}_{}_{}.txt".format(RUN_NAME,DATAFILE,SCALE_CUTS,DEMODEL))
    # ppd_theory_file = os.path.join(path_ppd, RUN_NAME, "ppd_theory_{}_{}_{}_{}.txt".format(RUN_NAME,DATAFILE,SCALE_CUTS,DEMODEL))
    # ppd_chain_file = os.path.join(path_ppd, RUN_NAME, "ppd_chain_{}_{}_{}_{}.txt".format(RUN_NAME,DATAFILE,SCALE_CUTS,DEMODEL))
    # chain_file = os.path.join(path_ppd, RUN_NAME, "chain_{}_{}_{}_{}.txt".format(RUN_NAME,DATAFILE,SCALE_CUTS,DEMODEL))

    dv_file = os.path.join(path_dv, DATAFILE)
    ppd_output_file_basename = os.path.join(path_ppd, RUN_NAME_PPD, 'ppd_'+RUN_NAME+'_'+RUN_NAME_PPD)

    if not get_pval_only:
        print("\n###############################")
        print("## Loading chains #############")
        print("###############################")

    # if chunks==0:
    #     # MCMC chain
    #     chain_file = os.path.join(path_chain, 'chain_'+RUN_NAME+'.txt')
    #     _mcmc_chain, weights = util.load_cosmosis_chain(chain_file, verbose=verbose, params_lambda=lambda _x:True, read_nsample=False)
    #     # PPD chain
    #     ppd_chain_file = os.path.join(path_ppd, RUN_NAME_PPD, 'ppd_chain_'+RUN_NAME+'_'+RUN_NAME_PPD+'.txt')
    #     ppd_chain, _ = util.load_cosmosis_chain(ppd_chain_file, verbose=verbose, params_lambda=lambda _x:True, read_nsample=False)
    # else:
    #     # PPD chain
    #     ppd_chain_list = []
    #     for ichunk in trange(chunks):
    #         ppd_chain_file = os.path.join(path_ppd, RUN_NAME_PPD, 'ppd_chain_'+RUN_NAME+'_'+RUN_NAME_PPD+'_{}.txt'.format(ichunk))
    #         if os.path.exists(ppd_chain_file):
    #             if os.path.getsize(ppd_chain_file):
    #                 ppd_chain_sub, _w = util.load_cosmosis_chain(ppd_chain_file, verbose=False, params_lambda=lambda _x:True, read_nsample=False)
    #                 ppd_chain_list.append(ppd_chain_sub)
    #     ppd_chain = OrderedDict()
    #     for k in ppd_chain_list[0].keys():
    #         ppd_chain[k] = np.concatenate([ppd_chain_sub[k] for ppd_chain_sub in ppd_chain_list])
    #     # MCMC chain
    #     chain_file = os.path.join(path_chain, 'chain_'+RUN_NAME+'.txt')
    #     _mcmc_chain_, weights_ = util.load_cosmosis_chain(chain_file, verbose=verbose, params_lambda=lambda _x:True, read_nsample=False)
    #     _n = chunks * len(_w)
    #     _mcmc_chain = {_k:_mcmc_chain_[_k][:_n] for _k in _mcmc_chain_.keys()}
    #     weights = weights_[:_n]
    #     print("Using {} chunks of size {} for a total of {} posterior samples".format(chunks, len(_w), _n))
    ppd_chain, _mcmc_chain, weights, ppd_chain_file = load_chains(path_ppd, path_chain, RUN_NAME, RUN_NAME_PPD, verbose=verbose, chunks=chunks, size_chunk=size_chunk)

    _key = list(ppd_chain.keys())[0]
    # Check compatibility
    np.testing.assert_array_almost_equal(ppd_chain[_key], _mcmc_chain[_key], decimal=6) # assert np.all(ppd_chain[_key]==_mcmc_chain[_key])

    print("\n###############################")
    print("## Getting p-values ###########")
    print("###############################")
    pval = np.average((ppd_chain['PPD--CHI2_DPRIME_REALIZATION']>ppd_chain['PPD--CHI2_DPRIME_DATA']).astype(float), weights=weights)

    if not get_pval_only:
        try:
            comp_avg = ppd_chain['PPD--CHI2_DPRIME_COMP_AVG']
            low, med, upp = get_resampled_pvalue(comp_avg, weights, N=Nresamplepval)
            print('     - Full     pval={:.4f} +{:.4f}/-{:.4f}  [16={:.4f}, 50={:.4f}, 84={:.4f}] ({})'.format(med, upp-med, med-low, low, med, upp, pval))
        except:
            print('     - Full     pval={:.4f} ({})'.format(pval, pval))

    if get_pval_only:
        return pval

    if get_pval_all_obs:
        pval = {'full':med}
        for _obs in ['XIP', 'XIM', '1X2', 'GAMMAT', 'WTHETA', '2X2']:
            _p = np.average((ppd_chain['PPD--CHI2_'+_obs+'_REALIZATION']>ppd_chain['PPD--CHI2_'+_obs+'_DATA']).astype(float), weights=weights)
            try:
                comp_avg = ppd_chain['PPD--CHI2_'+_obs+'_COMP_AVG']
                low, med, upp = get_resampled_pvalue(comp_avg, weights, N=Nresamplepval)
                print('     - {:<8} pval={:.4f} +{:.4f}/-{:.4f}  [16={:.4f}, 50={:.4f}, 84={:.4f}] ({})'.format(_obs, med, upp-med, med-low, low, med, upp, _p))
                pval[_obs.lower()] = med
            except:
                print("     - {:<8} pval = {:.4f} ({})".format(_obs, _p, _p))
    
    print("\n###############################")
    print("## Loading scale/bin cuts #####")
    print("###############################")
    # dico_indices = util.get_scale_cuts(ppd_chain_file, data_file, print_out=True, return_zbins=True, file_mode='r',
    #                                    like_module=like_module, data_sets_realizations=data_sets, num_observables=4)
    dico_indices = util.get_scale_cuts_new(ppd_chain_file, dv_file, print_out=True, return_zbins=True, file_mode='r',
                                       like_module=like_module, data_sets_realizations=data_sets, num_observables_tot=4)
    
    if load_dvs:
        print("\n###############################")
        print("## Loading realizations #######")
        print("###############################")
        if chunks==0:
            ppd_realization_file = os.path.join(path_ppd, RUN_NAME_PPD, 'ppd_'+RUN_NAME+'_'+RUN_NAME_PPD+'_dprime_real.npy')
            ppd_theory_file = os.path.join(path_ppd, RUN_NAME_PPD, 'ppd_'+RUN_NAME+'_'+RUN_NAME_PPD+'_dprime_theory.npy')
            # ppd_realizations = get_realizations(ppd_realization_file, chain_file)
            # ppd_theory = get_realizations(ppd_theory_file, chain_file)
            ppd_realizations = get_realizations(ppd_realization_file, ppd_chain)
            ppd_theory = get_realizations(ppd_theory_file, ppd_chain)
            # ppd_data_obs = get_data_obs(ppd_realization_file)
            ppd_data_obs = np.load(os.path.join(path_ppd, RUN_NAME_PPD, 'ppd_'+RUN_NAME+'_'+RUN_NAME_PPD+'_dprime_true.npy'))
        else:
            ppd_realizations = []
            ppd_theory = []
            for ichunk in trange(chunks):
                ppd_realization_file = os.path.join(path_ppd, RUN_NAME_PPD, 'ppd_'+RUN_NAME+'_'+RUN_NAME_PPD+'_{}_dprime_real.npy'.format(ichunk))
                ppd_theory_file = os.path.join(path_ppd, RUN_NAME_PPD, 'ppd_'+RUN_NAME+'_'+RUN_NAME_PPD+'_{}_dprime_theory.npy'.format(ichunk))
                # ppd_realizations.append(get_realizations(ppd_realization_file, chain_file, ichunk=ichunk))
                # ppd_theory.append(get_realizations(ppd_theory_file, chain_file, ichunk=ichunk))
                ppd_realizations.append(get_realizations(ppd_realization_file, ppd_chain, ichunk=ichunk, size_chunk=size_chunk))
                ppd_theory.append(get_realizations(ppd_theory_file, ppd_chain, ichunk=ichunk, size_chunk=size_chunk))
            ppd_realizations = np.concatenate(ppd_realizations)
            ppd_theory = np.concatenate(ppd_theory)
            # ppd_data_obs = get_data_obs(ppd_realization_file)
            ppd_data_obs = np.load(os.path.join(path_ppd, RUN_NAME_PPD, 'ppd_'+RUN_NAME+'_'+RUN_NAME_PPD+'_0_dprime_true.npy'))


    # if get_cov_pm_dprime:
    #     cov_pm_dprime = np.loadtxt(os.path.join(path_ppd, RUN_NAME_PPD, 'ppd_'+RUN_NAME+'_'+RUN_NAME_PPD+'_dprime_cov_pm.txt'))
    #     return pval, dico_indices, ppd_data_obs, ppd_realizations, ppd_theory, weights, ppd_chain, cov_pm_dprime
    
    # else:
    #     return pval, dico_indices, ppd_data_obs, ppd_realizations, ppd_theory, weights, ppd_chain

        return pval, dico_indices, ppd_data_obs, ppd_realizations, ppd_theory, weights, ppd_chain, ppd_output_file_basename

    else:
        return pval, dico_indices, weights, ppd_chain, ppd_output_file_basename