import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import scipy.stats
from astropy.io import fits

def weighted_quantile(values, quantiles, sample_weight=None, 
                      values_sorted=False, old_style=False):
    """ Very close to numpy.percentile, but supports weights.
    NOTE: quantiles should be in [0, 1]!
    :param values: numpy.array with data
    :param quantiles: array-like with many quantiles needed
    :param sample_weight: array-like of the same length as `array`
    :param values_sorted: bool, if True, then will avoid sorting of
        initial array
    :param old_style: if True, will correct output to be consistent
        with numpy.percentile.
    :return: numpy.array with computed quantiles.
    """
    values = np.array(values)
    quantiles = np.array(quantiles)
    if sample_weight is None:
        sample_weight = np.ones(len(values))
    sample_weight = np.array(sample_weight)
    assert np.all(quantiles >= 0) and np.all(quantiles <= 1), \
        'quantiles should be in [0, 1]'

    if not values_sorted:
        sorter = np.argsort(values)
        values = values[sorter]
        sample_weight = sample_weight[sorter]

    weighted_quantiles = np.cumsum(sample_weight) - 0.5 * sample_weight
    if old_style:
        # To be convenient with numpy.percentile
        weighted_quantiles -= weighted_quantiles[0]
        weighted_quantiles /= weighted_quantiles[-1]
    else:
        weighted_quantiles /= np.sum(sample_weight)
    return np.interp(quantiles, weighted_quantiles, values)

def calc_chi2(x, icov, xmean=None):
    if xmean is not None :
        y = x - xmean
    else :
        y = x
    # icov = np.linalg.inv(cov)
    return np.dot(y.T, np.dot(icov, y))
    
def print_pval(pval):
    # f = mticker.ScalarFormatter(useOffset=False, useMathText=True)
    # if pval<0.01:
    #     return "$p={}$".format(f._formatSciNotation('%1.1e' % pval))
    # elif pval>0.99:
    #     return "$p=1-{}$".format(f._formatSciNotation('%1.1e' % (1.-pval)))
    # else:
    #     return '$p={:.2f}$'.format(pval)
    return '$p={:.2f}$'.format(pval)

def print_pcal(pcal):
    # if pcal > 0.01:
    #     return r'$\tilde{{p}}={:.2f}$'.format(pcal)
    # else:
    #     return r'$\tilde{{p}}<0.01$'
    if pcal > 0.01:
        return r'$p={:.2f}$'.format(pcal)
    else:
        return r'$p<0.01$'    
def step_x_bins(x):
    y = np.zeros(len(x)+1)


    
def ppd_plot(ax,
             obs, zi, zj,
             xlim,
             dico_indices, all_data, all_best_fit, all_realizations, all_theory, weights,
             sigma_levels=[1,2], theta_bins = np.geomspace(2.5,250.,21), ylim=3.9,
             blind_y=False,
             ppd_output_file_basename=None,
             pcal={},
             pval={},
             scale_bestfit=True,
             scale_relative=False,
             theta_factor=0
            ):
    
    #One liner to find the index in the full covariance matrix where obs starts
    # idx0 = np.array([0]+[len(all_data[d].data) for d in dico_indices.keys()]).cumsum()[list(dico_indices.keys()).index(obs)]
    hdr = all_data['COVMAT'].header
    # idx0 = [hdr['STRT_'+str(i)] for i in range(6) if hdr['NAME_'+str(i)]==obs][0]
    idx0 = [hdr['STRT_'+str(i)] for i in range(4) if hdr['NAME_'+str(i)]==obs][0]
    
    idx_cov_sub = dico_indices[obs][zi,zj]['idx_full']
    idx_real_sub  = dico_indices[obs][zi,zj]['idx_sub']
    idx_theta_all = np.where(np.logical_and(all_data[obs].data['BIN1']==zi,all_data[obs].data['BIN2']==zj))[0]
    idx_cov_all = idx_theta_all+idx0
    idx_theta_sub = idx_cov_sub-idx0
    idx_theta_sub_bins = idx_theta_sub % 20


    theta = all_data[obs].data['ANG']
    data = all_data[obs].data['VALUE']
    bf = all_best_fit[obs].data['VALUE']
    
    cov = all_data['COVMAT'].data
    std_sub = np.sqrt(np.diagonal(cov[idx_cov_sub,:][:,idx_cov_sub]))
    std_all = np.sqrt(np.diagonal(cov[idx_cov_all,:][:,idx_cov_all]))
    
    mean = np.array([weighted_quantile(all_realizations[:,_idx], 0.5, sample_weight=weights) for _idx in idx_real_sub])
    # print(mean.shape)
    
    if blind_y:
        idx_theta_all = idx_theta_sub
        for i in range(len(mean)):
            bf[idx_theta_sub[i]] = mean[i]
            # print("fwaef")
            # print(mean)
            # print(bf[idx_theta_sub])
            # print([idx_theta_sub])
    
    # Plot data
    if scale_bestfit:
        if not blind_y:
            ax.plot(theta[idx_theta_all], (data[idx_theta_all]-bf[idx_theta_all])/std_all,
                ls=' ', c='orangered', marker='o', alpha=0.2)
        ax.plot(theta[idx_theta_sub], (data[idx_theta_sub]-bf[idx_theta_sub])/std_sub,
                ls=' ', c='orangered', marker='o', zorder=9999)
    elif scale_relative:
        ax.plot(theta[idx_theta_sub], (data[idx_theta_sub]/mean) -1.,
                ls=' ', c='orangered', marker='o', zorder=9999)
    else:
        if not blind_y:
            ax.plot(theta[idx_theta_all], theta[idx_theta_all]**theta_factor * data[idx_theta_all],
                ls=' ', c='orangered', marker='o', alpha=0.2)
        ax.plot(theta[idx_theta_sub], theta[idx_theta_sub]**theta_factor * data[idx_theta_sub],
                ls=' ', c='orangered', marker='o', zorder=9999)
    
    # Plot realizations quantiles
    if scale_bestfit:
        ax.plot(np.repeat(theta_bins[np.append(idx_theta_sub_bins, idx_theta_sub_bins[-1]+1)],2)[1:-1], np.repeat((mean-bf[idx_theta_sub])/std_sub, 2), color='dodgerblue', lw=2, zorder=999) #color='#2b91b3'
    elif scale_relative:
        ax.plot(np.repeat(theta_bins[np.append(idx_theta_sub_bins, idx_theta_sub_bins[-1]+1)],2)[1:-1], np.repeat(mean/mean - 1., 2), color='dodgerblue', lw=2, zorder=999) #color='#2b91b3'
    else:
        ax.plot(np.repeat(theta_bins[np.append(idx_theta_sub_bins, idx_theta_sub_bins[-1]+1)],2)[1:-1],
                np.repeat(theta[idx_theta_sub]**theta_factor * mean, 2), color='dodgerblue', lw=2, zorder=999) #color='#2b91b3'

    for sl in sigma_levels:
        high = np.array([weighted_quantile(all_realizations[:,_idx], scipy.stats.norm.cdf(+sl), sample_weight=weights) for _idx in idx_real_sub])
        low  = np.array([weighted_quantile(all_realizations[:,_idx], scipy.stats.norm.cdf(-sl), sample_weight=weights) for _idx in idx_real_sub])
        if scale_bestfit:
            y_low = (low-bf[idx_theta_sub])/std_sub
            y_high = (high-bf[idx_theta_sub])/std_sub
        elif scale_relative:
            y_low = low/mean - 1.
            y_high = high/mean - 1.
        else:
            y_low = low * theta[idx_theta_sub]**theta_factor
            y_high = high * theta[idx_theta_sub]**theta_factor
        ax.bar(theta[idx_theta_sub], height=(y_high-y_low), bottom=y_low, width=(theta_bins[1:]-theta_bins[:-1])[idx_theta_sub_bins], color='dodgerblue', alpha=0.4) # color='#2b91b3'# '#60c975'
        
    # try:
    # pcal = pcal['{}_{}_{}'.format(obs,zi,zj)]
    # ax.text(0.05, 0.1, print_pcal(pcal), ha='left', va='bottom', transform=ax.transAxes, fontsize=12)
    # except:
    #     try:
    #         pval = pval['{}_{}_{}'.format(obs,zi,zj)]
    #         ax.text(0.05, 0.05, print_pval(pval), ha='left', va='bottom', transform=ax.transAxes, fontsize=12)
    #     except:
    #         print("Recomputing pval for", obs, zi, zj)
    #         # Compute chi2 and p-values
    #         if ppd_output_file_basename is None:
    #             icov = np.linalg.inv(cov[idx_cov_sub,:][:,idx_cov_sub])
    #         else:
    #             # icov = np.linalg.inv(cov_pm_dprime)
    #             cov = np.loadtxt(ppd_output_file_basename+'_dprime_cov.txt')
    #             icov_pm = np.loadtxt(ppd_output_file_basename+'_dprime_inv_cov_pm.txt')
    #             # print("cov.shape", cov.shape)
    #             # print(icov_pm.shape)
    #             # print(idx_real_sub.shape)
    #             # print(idx_real_sub)
    #             icov_full = np.linalg.inv(cov)
    #             # cov_pm = np.linalg.inv(icov_pm)
    #             icov_pm_corr = icov_pm - icov_full

    #             icov = np.linalg.inv(cov[idx_real_sub,:][:,idx_real_sub]) + icov_pm_corr[idx_real_sub,:][:,idx_real_sub]

    #         chi2_ppd = np.array([calc_chi2(all_realizations[i,idx_real_sub], icov, all_theory[i,idx_real_sub]) for i in range(len(all_realizations))])
    #         chi2_obs = np.array([calc_chi2(data[idx_theta_sub], icov, all_theory[i,idx_real_sub]) for i in range(len(all_realizations))])
    #         # chi2_ppd = np.array([calc_chi2(all_realizations[i,idx_real_sub], icov, bf[idx_theta_sub]) for i in range(len(all_realizations))])
    #         # chi2_obs = np.array([calc_chi2(data[idx_theta_sub], icov, bf[idx_theta_sub]) for i in range(len(all_realizations))])
    #         pval = np.average((chi2_ppd>chi2_obs).astype(float), weights=weights) # float(np.sum(chi2_ppd>chi2_obs))/len(chi2_ppd)
    #         ax.text(0.05, 0.05, print_pval(pval), ha='left', va='bottom', transform=ax.transAxes, fontsize=12) # fontsize=8 if len(print_pval(pval))>8 else 12)

    
    if scale_bestfit:
        ax.axhline(y=0., c='0.8', zorder=99)
        ax.axhline(y=-1, c='0.8', zorder=99, lw=1, ls='--')
        ax.axhline(y=+1, c='0.8', zorder=99, lw=1, ls='--')
    else:
        ax.axhline(y=0., c='0.8', ls='--', zorder=-9999)
    ax.set_xlim(xlim)
    ax.set_xscale('log')
    if scale_bestfit or scale_relative:
        if np.isscalar(ylim):
            ax.set_ylim(-ylim, +ylim)
        else:
            ax.set_ylim(ylim[0],ylim[1])
    
    plt.setp(ax.get_xminorticklabels(), visible=False)
    
    ax.text(0.05, 0.95, '{},{}'.format(zi,zj), ha='left', va='top', transform=ax.transAxes, fontsize=12)

    return idx_real_sub, idx_theta_sub, idx_cov_sub



def ppd_plot_all_obs(obs, nrows, ncols, ylabel, prefix, xlim,
                     dico_indices, all_data, all_best_fit, all_realizations,
                     all_theory, weights, invert_zizj=False, coeff_figsize=1., width=None, get_chi2=True,
                     indices_real=None, indices_data=None, indices_cov=None, xlabel_only_bottom=False, xlabel_only_one_central=False, ylim=3.9, blind_y=False,
                     ppd_output_file_basename=None,
                     pcal={},
                     pval={},
                     scale_bestfit=True,
                     scale_relative=False,
                     theta_factor=0
                    ):
        
    if ppd_output_file_basename is None:
        print("Using DV.fits covariance for chi2")
    else:
        print("Using pm cov in", ppd_output_file_basename)

    if width is None:
        width = np.log(xlim[1]/xlim[0])/np.log(250./2.5)
        
    figsize=((1.+coeff_figsize*width*4*ncols)*1.,(1.+coeff_figsize*3*nrows)*1.) 
    left = 0.8/figsize[0]
    right = 1. - 0.2/figsize[0]
    bottom = 0.8/figsize[1]
    top = 1-0.2/figsize[1]

    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=figsize, squeeze=False)
    
    zis = np.sort(np.unique([x[0] for x in dico_indices[obs].keys()]))
    zjs = np.sort(np.unique([x[1] for x in dico_indices[obs].keys()]))
    # print(zis, zjs)
    # assert nrows==len(zis)
    # assert ncols==len(zjs)
    
    idx_real = []
    idx_data = []
    idx_cov = []
    
    fontsize_axis_label = 14
    
    if nrows>1:
        used_axes = []
        for i, zi in enumerate(zis):
            for j, zj in enumerate(zjs):
                ax = axes[j,i] if invert_zizj else axes[i,j]
                if (zi,zj) in dico_indices[obs].keys():
                    _idx_real, _idx_data, _idx_cov = ppd_plot(ax, obs, zi, zj, xlim, dico_indices, all_data, all_best_fit, all_realizations, all_theory,
                                                              weights, ylim=ylim, blind_y=blind_y, ppd_output_file_basename=ppd_output_file_basename, pcal=pcal, pval=pval, scale_bestfit=scale_bestfit, scale_relative=scale_relative, theta_factor=theta_factor)
                    idx_real.append(_idx_real)
                    idx_data.append(_idx_data)
                    idx_cov.append(_idx_cov)
                    ax.set_ylabel(ylabel, fontsize=fontsize_axis_label)
                    used_axes.append(ax)
                else:
                    ax.axis('off')
        
        for i in range(nrows):
            for j in range(ncols):
                ax = axes[i,j]
                # x-ticks
                if i<nrows-1:
                    if axes[i+1,j] in used_axes:
                        ax.set_xticks([])
                    else:
                        ax.set_xlabel('$\\theta$ (arcmin)', fontsize=fontsize_axis_label)
                if i==nrows-1:
                    ax.set_xlabel('$\\theta$ (arcmin)', fontsize=fontsize_axis_label)
                if j>0:
                    if axes[i,j-1] in used_axes:
                        ax.set_yticks([])
                        ax.set_ylabel('', fontsize=fontsize_axis_label)
    else:
        used_axes = axes
        for i, (zi,zj) in enumerate(dico_indices[obs].keys()):
            ax = axes[0,i]
            _idx_real, _idx_data, _idx_cov = ppd_plot(ax, obs, zi, zj, xlim, dico_indices, all_data, all_best_fit, all_realizations, all_theory,
                                                      weights, ylim=ylim, blind_y=blind_y, ppd_output_file_basename=ppd_output_file_basename, pcal=pcal, pval=pval, scale_bestfit=scale_bestfit, scale_relative=scale_relative, theta_factor=theta_factor)
            idx_real.append(_idx_real)
            idx_data.append(_idx_data)
            idx_cov.append(_idx_cov)
            if i==0:
                ax.set_ylabel(ylabel, fontsize=fontsize_axis_label)
            else:
                ax.set_yticks([])
            ax.set_xlabel('$\\theta$ [arcmin]', fontsize=fontsize_axis_label)
    
    for ax in axes.ravel():
        ax.tick_params(axis='both', which='major', labelsize=11)
        
    if xlabel_only_bottom:
        for ax in axes[:-1,:].ravel():
            ax.set_xlabel('')
    
    if xlabel_only_one_central:
        for ax in axes.ravel():
            ax.set_xlabel('')
        fig.add_subplot(111, frameon=False)
        # hide tick and tick label of the big axis
        plt.tick_params(labelcolor='none', top=False, bottom=False, left=False, right=False)
        plt.xlabel('$\\theta$ [arcmin]', fontsize=fontsize_axis_label)

    if not (scale_bestfit or scale_relative):
        for i in range(nrows):
            ymin = min([axes[i,j].get_ylim()[0] for j in range(ncols) if axes[i,j] in used_axes])
            ymax = max([axes[i,j].get_ylim()[1] for j in range(ncols) if axes[i,j] in used_axes])
            for j in range(ncols):
                axes[i,j].set_ylim(ymin,ymax)
    
    extra = []
    if get_chi2:
        idx_real = np.concatenate(idx_real)
        idx_data = np.concatenate(idx_data)
        idx_cov = np.concatenate(idx_cov)
        
        # print(idx_real)
        # print(idx_cov)
        # print(idx_data)
        
        if ppd_output_file_basename is None:
            # raise ValueError
            cov_ = all_data['COVMAT'].data[idx_cov,:][:,idx_cov]
            icov_ = np.linalg.inv(cov_)
        else:
            # # Load saved covariances
            # cov_ = np.loadtxt(ppd_output_file_basename+'_dprime_cov.txt')
            # icov_pm_ = np.loadtxt(ppd_output_file_basename+'_dprime_inv_cov_pm.txt')
            # # # Compute icov pm correction
            # # icov_full_ = np.linalg.inv(cov_)
            # cov_pm_ = np.linalg.inv(icov_pm_)
            # # icov_pm_corr_ = icov_pm_ - icov_full_
            # # # # Compute icov
            # # icov_ = np.linalg.inv(cov_[idx_real,:][:,idx_real]) + icov_pm_corr_[idx_real,:][:,idx_real]
            # # # cov_pm_corr_ = np.linalg.inv(icov_pm_)-cov_
            # # # icov_ = np.linalg.inv(cov_[idx_real,:][:,idx_real]+cov_pm_corr_[idx_real,:][:,idx_real])
            # icov_ = np.linalg.inv(cov_pm_[idx_real,:][:,idx_real])
            print("Loading", ppd_output_file_basename+'_{}_inv_cov_pm.npy'.format(obs))
            icov_ = np.load(ppd_output_file_basename+'_{}_inv_cov_pm.npy'.format(obs))

        data_ = all_data[obs].data['VALUE'][idx_data]
        real_ = all_realizations[:,idx_real]
        theory_ = all_theory[:,idx_real]
        
        # print(real_.shape)
        
        chi2_ppd = np.array([calc_chi2(real_[i], icov_, theory_[i]) for i in range(len(real_))])
        chi2_obs = np.array([calc_chi2(data_, icov_, theory_[i]) for i in range(len(real_))])
        pval = np.average((chi2_ppd>chi2_obs).astype(float), weights=weights) # float(np.sum(chi2_ppd>chi2_obs))/len(chi2_ppd)
        print("## Observable p-value")
        print("- ", obs, "p={:.6f}".format(pval))
        
        extra = (chi2_ppd, chi2_obs)

    plt.subplots_adjust(left=left, right=right, top=top, bottom=bottom, wspace=0, hspace=0)
    
    plt.savefig('./figs/'+prefix+'_'+obs+'.pdf', transparent=True)
    
    plt.show()
    
    return fig, axes, extra