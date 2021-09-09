import numpy as np
import os
import pdb
from cosmosis.datablock import names, option_section

## Developpers Eric Baxter, Cyrille Doux et al.

def setup(options):
    statistic = options.get_string(option_section, "statistic", "chi2")
    d_names_string = options.get_string(option_section, "ppd_d_names", "xip")
    dprime_names_string = options.get_string(option_section, "ppd_dprime_names", "xip")
    condition_on_d = options.get_bool(option_section, "condition_on_d", False)
    use_like_cuts = options.get_int(option_section, "use_like_cuts", -1)
    assert use_like_cuts==2, "[ppd] This is hacky but for Y3 runs, we will always run 3 likelihood modules, so better to use use_like_cuts=2"
    ndraws = options.get_int(option_section, "ndraws", 1)
    assert ndraws>0
    # assert 0<=use_like_cuts<=2
    # if use_like_cuts:
    #     assert condition_on_d, "[ppd] Using fancy cuts but not conditioning, what are you doing??"

    d_names = d_names_string.split()
    dprime_names = dprime_names_string.split()

    # ppd_output_file = None
    # if options.has_value(option_section,"ppd_output_file"):
    #     ppd_output_file = options.get_string(option_section, "ppd_output_file")
    # theory_output_file = None
    # if options.has_value(option_section,"theory_output_file"):
    #     theory_output_file = options.get_string(option_section, "theory_output_file")

    # config = {'statistic': statistic, 'd_names':d_names, 'dprime_names': dprime_names, 'ppd_output_file':ppd_output_file, 'theory_output_file':theory_output_file, 'condition_on_d':condition_on_d, 'use_like_cuts':use_like_cuts}

    config = {'statistic': statistic, 'd_names':d_names, 'dprime_names': dprime_names, 'condition_on_d':condition_on_d, 'use_like_cuts':use_like_cuts, 'ndraws':ndraws}
    return config

def get_indices(block, like_full_name, like_sub_name, allow_missing=False):
    n_full = len(block[names.data_vector, like_full_name+'_theory'])
    n_sub = len(block[names.data_vector, like_sub_name+'_theory'])

    keys = ['angle','bin1','bin2','dataset_indices']
    A_full = np.array([block[names.data_vector, like_full_name+'_'+x] for x in keys]).T
    A_sub = np.array([block[names.data_vector, like_sub_name+'_'+x] for x in keys]).T

    assert n_full==A_full.shape[0]
    assert n_sub==A_sub.shape[0]

    idx = []
    for i in range(n_sub):
        temp = np.where(np.all(np.abs(A_full.astype(float)-A_sub[i].astype(float))<1e-4, axis=1))[0]
        assert len(temp)<=1
        if not allow_missing:
            assert len(temp)==1
            idx.append(temp[0])
        else:
            if len(temp)==1:
                idx.append(temp[0])
            else:
                continue
    return np.array(idx)


def execute(block, config):
    '''
        We want to test the consistency of d and d' under some model.  We'll generate
        realizations of d' conditioned on d.  This is the posterior predictive distribution.
        We'll use some statistic (e.g. chi^2) to compare the simulated realizations with
        the true d'.

        We assume that .ini is being run on some previous chain from the analysis of d.
    '''

    statistic = config['statistic']
    d_names = config['d_names']
    dprime_names = config['dprime_names']
    ndraws = config['ndraws']
    # ppd_output_file = config['ppd_output_file']
    # theory_output_file = config['theory_output_file']

    # Get observed data, theory at current parameters, and covariance
    obs = block[names.data_vector, '2pt_data']
    theory = block[names.data_vector, '2pt_theory']
    cov = block[names.data_vector, '2pt_covariance']
    # cov = np.linalg.inv(block[names.data_vector, '2pt_inverse_covariance']) <-- this is wrong: this was intended to take PM-marginalization into account, but that corresponds to an effective covariance after marginalization. The covariance of the data remains unchanged.
    npts = len(obs)

    # # Get info about data vector
    dataset_names = block[names.data_vector, '2pt_dataset_names'].split(',')
    dataset_indices = block[names.data_vector, '2pt_dataset_indices']
    # dataset_all_names = np.array(dataset_names)[dataset_indices]

    # # Get indices of dprime and d in data_vector
    # d_indices = []
    # for ii in xrange(0,len(d_names)):
    #     matches = np.where(dataset_all_names == d_names[ii])[0]
    #     d_indices.append(matches)
    # d_indices = np.concatenate(d_indices)
    # dprime_indices = []
    # for ii in xrange(0,len(dprime_names)):
    #     matches = np.where(dataset_all_names == dprime_names[ii])[0]
    #     dprime_indices.append(matches)
    # dprime_indices = np.concatenate(dprime_indices)

    if config['use_like_cuts'] == 0:
        # Get indices of dprime and d in data_vector
        d_indices = []
        for ii in xrange(0,len(d_names)):
            matches = np.where(dataset_all_names == d_names[ii])[0]
            d_indices.append(matches)
        d_indices = np.concatenate(d_indices)
        dprime_indices = []
        for ii in xrange(0,len(dprime_names)):
            matches = np.where(dataset_all_names == dprime_names[ii])[0]
            dprime_indices.append(matches)
        dprime_indices = np.concatenate(dprime_indices)
    elif config['use_like_cuts'] == 1:
        dprime_indices = get_indices(block, '2pt', '2pt_dprime')
        d_indices = np.setdiff1d(np.arange(npts), dprime_indices)
    elif config['use_like_cuts'] == 2:
        d_indices = get_indices(block, '2pt', '2pt_d')
        dprime_indices = get_indices(block, '2pt', '2pt_dprime')
    else:
        raise ValueError("That should not happen")
    # print("[ppd] d_indices", d_indices.shape, d_indices)
    # print("[ppd] dprime_indices", dprime_indices.shape, dprime_indices)
        
    # Get relevant covariances and subplots
    cov_dprime = (cov[dprime_indices, :])[:, dprime_indices]
    cov_ddprime = (cov[dprime_indices, :])[:, d_indices]
    cov_d = (cov[d_indices, :])[:, d_indices]
    inv_cov_d = np.linalg.inv(cov_d)
    inv_cov_dprime = np.linalg.inv(cov_dprime)

    # Gaussian random realization conditioned on observed vector - see evidence notes
    dprime_realization_all = []
    if (config['condition_on_d']):
        assert len(np.intersect1d(d_indices, dprime_indices))==0, "[ppd] Conditioning is activated but d and dprime have overlapping indices."
        mean_dprime_conditioned = theory[dprime_indices] + np.dot(cov_ddprime, np.dot(inv_cov_d, (obs - theory)[d_indices]))
        cov_dprime_conditioned = cov_dprime - np.dot(cov_ddprime, np.dot(inv_cov_d, cov_ddprime.transpose()))
        for _ in range(ndraws):
            dprime_realization = np.random.multivariate_normal(mean_dprime_conditioned, cov_dprime_conditioned)
            dprime_realization_all.append(dprime_realization)
    else:
        mean_dprime = theory[dprime_indices]
        for _ in range(ndraws):
            dprime_realization = np.random.multivariate_normal(mean_dprime, cov_dprime)
            dprime_realization_all.append(dprime_realization)

    dprime_true = obs[dprime_indices]
    dprime_theory = theory[dprime_indices]

    d_true = obs[d_indices]
    d_theory = theory[d_indices]

    # Write ppd outputs to the block
    block['ppd', 'dprime_true'] = dprime_true
    block['ppd', 'dprime_realization'] = dprime_realization
    block['ppd', 'dprime_theory'] = dprime_theory

    block['ppd', 'd_true'] = d_true
    block['ppd', 'd_theory'] = d_theory

    if (statistic == 'values'):
        block['ppd', 'values_data'] = dprime_true
        block['ppd', 'values_realization'] = dprime_realization

    inv_cov_dprime_chi2 = block[names.data_vector, '2pt_dprime_inverse_covariance'] 
    inv_cov_d_chi2 = block[names.data_vector, '2pt_d_inverse_covariance']

    if (statistic == 'chi2'):
        # Actual data
        diff_data = dprime_true - dprime_theory
        chi2_data = np.dot(diff_data, np.dot(inv_cov_dprime_chi2, diff_data))
        # Realization
        diff_realization = dprime_realization - dprime_theory
        chi2_realization = np.dot(diff_realization, np.dot(inv_cov_dprime_chi2, diff_realization))

        diff_d = d_true - d_theory
        chi2_d = np.dot(diff_d, np.dot(inv_cov_d_chi2, diff_d))

        block['ppd', 'chi2_dprime_data'] = chi2_data
        block['ppd', 'chi2_dprime_realization'] = chi2_realization
        block['ppd', 'chi2_d_data'] = chi2_d

        c = 0.
        for i in range(ndraws):
            diff_realization = dprime_realization_all[i] - dprime_theory
            chi2_realization = np.dot(diff_realization, np.dot(inv_cov_dprime_chi2, diff_realization))
            if chi2_realization>chi2_data:
                c += 1.
        c /= ndraws
        block['ppd', 'chi2_dprime_comp_avg'] = c


        for like_name in ['xip', 'xim', '1x2', 'gammat', 'wtheta', '2x2']:
            print(like_name)
            try:
                _like_indices = get_indices(block, '2pt_dprime', '2pt_'+like_name, allow_missing=False)
            except:
                print("[ppd] getting indices from {} within 2pt_dprime failed".format(like_name))
                _like_indices = []
            if len(_like_indices)>0:
                print("[ppd] using {} with {} points".format(like_name, len(_like_indices)))
                _d = dprime_true[_like_indices]
                _th = dprime_theory[_like_indices]
                _icov = block[names.data_vector, '2pt_{}_inverse_covariance'.format(like_name)]
                _chi2_d = np.dot(_d-_th, np.dot(_icov, _d-_th))
                block['ppd', 'chi2_'+like_name+'_data'] = _chi2_d
                _c = 0.
                for i in range(ndraws):
                    _r = dprime_realization_all[i][_like_indices]
                    _chi2_r = np.dot(_r-_th, np.dot(_icov, _r-_th))
                    if _chi2_r>_chi2_d:
                        _c += 1.
                _c /= ndraws
                block['ppd', 'chi2_'+like_name+'_comp_avg'] = _c
                block['ppd', 'chi2_'+like_name+'_realization'] = _chi2_r
            else:
                block['ppd', 'chi2_'+like_name+'_data'] = -1.
                block['ppd', 'chi2_'+like_name+'_realization'] = -1.




    return 0

