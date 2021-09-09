import numpy as np
from astropy.io import fits
import pdb
from collections import OrderedDict

def as_list(x):
    if type(x) in [list, np.ndarray]:
        return x
    else:
        return [x]
    
#Giving a chain file and a data file, get the indices in the data file that correspond to 
#scale cuts used in chain
def get_scale_cuts_new(chain_file, data_file, print_out=False, return_zbins=False, file_mode='rb', like_module='2pt_like', data_sets_realizations=None, num_observables_tot=4):

    fits_data = fits.open(data_file)

    #Get observables
    # if num_observables is None:
        # num_observables = len(fits_data)-4
    data_sets = []
    start_indices = []
    header = fits_data['COVMAT'].header
    len_data = fits_data['COVMAT'].data.shape[0]
    for oi in range(num_observables_tot):
        data_sets.append(header['NAME_' + str(oi)])
        start_indices.append(header['STRT_' + str(oi)])
    print("Found observables in data_file", data_sets)
    
    #Current hardcoded
    min_theta = 2.5
    max_theta = 250.0

    #First we use chain to get scale cuts and bin cuts
    cut_bins = {x:{} for x in data_sets}
    cut_scales = {x:{} for x in data_sets}
    with open(chain_file, file_mode) as file:
        likelihood_module_reached = False
        for idx, line in enumerate(file):
            if likelihood_module_reached and line.startswith("## ["):
                break
            if not likelihood_module_reached and not line.startswith("## ["+like_module+"]"):
                continue
            else:
                if not likelihood_module_reached:
                    if print_out:
                        print("Reading options for likelihood module: ", like_module)
                likelihood_module_reached = True
                
            #If line is a bin cut line
            if "## cut_" in line:
                # print(line)
                line_sp = line.split(' ')
                observable = line_sp[1][4:]
                cut_bins[observable] = []
                for _x in line_sp[3:]:
                    if len(_x)>0:
                        xi, xj = _x.split(',')
                        if print_out:
                            print("Found bin cut for",observable,int(xi),int(xj))
                        cut_bins[observable].append((int(xi),int(xj)))
                
            #If line is a scale cut line
            if ('## angle_range_' in line):
                line_sp = line.split()#' ')
                observable = line_sp[1][12:-4]
                if observable in data_sets:
                    bin_string_sp = line_sp[1].split('_')
                    bini = int(bin_string_sp[-2])
                    binj = int(bin_string_sp[-1])
                    min_theta_cut = float(line_sp[3])
                    max_theta_cut = float(line_sp[4])
                    cut_scales[observable][bini,binj]=(min_theta_cut,max_theta_cut)
                    if int(print_out)>1:
                        print("Found scale cut for ",observable,bini,binj,min_theta_cut,max_theta_cut)

        if int(print_out)==1:
            print("Found scale cuts for ", {k:list(cut_scales[k].keys()) for k in cut_scales.keys()})
    
    #Loop over observables in data vector
    dico_indices = {}
    i_sub = 0
    for di, data_set in enumerate(data_sets):
        if data_set in data_sets_realizations:
            if print_out:
                print("Getting indices for", data_set)
            dico_indices[data_set] = {}
            data_obs = fits_data[data_set].data

            # Loop over data points for given observable
            for i in range(len(data_obs)):
                bin1 = data_obs[i]['BIN1']
                bin2 = data_obs[i]['BIN2']

                if (bin1,bin2) not in dico_indices[data_set].keys():
                    dico_indices[data_set][bin1,bin2] = {'idx_full':[], 'idx_sub':[]}

                # Check if bin is cut
                # if data_set in cut_bins.keys():
                if (bin1,bin2) in cut_bins[data_set]:
                    continue

                # Check if scale is cut
                # if data_set in cut_scales.keys():
                if (bin1,bin2) in cut_scales[data_set]:
                    tmin, tmax = cut_scales[data_set][bin1,bin2]
                    if data_obs[i]['ANG'] < tmin or data_obs[i]['ANG'] > tmax:
                        continue

                dico_indices[data_set][bin1,bin2]['idx_full'].append(start_indices[di]+i)
                dico_indices[data_set][bin1,bin2]['idx_sub'].append(i_sub)
                i_sub += 1
            
            
            # Remove empty lists
            for k in list(dico_indices[data_set].keys()):
                if len(dico_indices[data_set][k]['idx_full']) == 0:
                    assert len(dico_indices[data_set][k]['idx_sub']) == 0
                    dico_indices[data_set].pop(k)
                else:
                    dico_indices[data_set][k]['idx_full'] = np.array(dico_indices[data_set][k]['idx_full'])
                    dico_indices[data_set][k]['idx_sub'] = np.array(dico_indices[data_set][k]['idx_sub'])
    
    return dico_indices
            

def load_cosmosis_chain(filename, params_lambda=lambda s:s.upper().startswith('COSMO'), verbose=True, read_nsample=True):
    """
    Loading a cosmosis chain

    Parameters
    ----------
    filename : 
    params_lambda : function that takes cosmosis parameter's name and returns True/False whether it should be used.
    verbose :
    """
    from collections import OrderedDict

    def get_nsample(filename):
        with open(filename,"r") as fi:
            for ln in fi:
                if ln.startswith("#nsample="):
                    nsamples = int(ln[9:])
                    break
        return nsamples

    
    with open(filename, 'r') as file:
        # Read all parameters names
        s = file.readline()
        s_a = s[1:].split()

        # Read sampler        
        s = file.readline()
        # print(s)
        if s == '#sampler=multinest\n':
            print("Loading Multinest chain at")
            print(filename)
            if read_nsample:
                list_s = file.read().splitlines()
                nsample = int(list_s[-3].replace('#nsample=',''))
            else:
                nsample = 0
        elif s == '#sampler=polychord\n':
            print("Loading Polychord chain at")
            print(filename)
            # list_s = file.read().splitlines()
            # nsample = int(list_s[-3].replace('#nsample=',''))
            if read_nsample:
                list_s = file.read().splitlines()
                nsample = int(list_s[-3].replace('#nsample=',''))
            else:
                nsample = 0
        elif s == '#sampler=emcee\n':
            print("Loading emcee chain at")
            print(filename)
            nsample = 0
        elif s == '#sampler=list\n':
            if verbose:
                print("Loading list chain at")
                print(filename)
            nsample = 0
        elif s == '#sampler=listppd\n':
            if verbose:
                print("Loading list chain at")
                print(filename)
            nsample = 0
        elif s == "#sampler=maxlike\n":
            print("Loading maxlike chain at")
            print(filename)
            nsample = 0
        elif s == "#sampler=metropolis\n":
            print("Loading metropolis chain at")
            print(filename)
            nsample = 0
        elif s == "#sampler=importance\n":
            print("Loading importance chain at")
            print(filename)
            nsample = 0
        elif s == "#sampler=apriori\n":
            print("Loading apriori chain at")
            print(filename)
            nsample = 0
        elif s == "#sampler=pmc\n":
            print("Loading pmc chain at")
            print(filename)
            list_s = file.read().splitlines()
            nsample = int(list_s[-1].replace('#nsample=',''))
        else:
            print("Loading chain at")
            print(filename)
            print("Sampler not found -- non-standard cosmosis output")
            nsample = 0
            # raise NotImplementedError

    # Load the chain
    chain = np.atleast_2d(np.loadtxt(filename))   

    dico = OrderedDict()
    keys = []
    for i, s in enumerate(s_a):
        if params_lambda(s):
            keys.append(s)
            dico[s] = chain[-nsample:,i]
            
    if 'weight' in s_a:
        weights = chain[-nsample:,i]
    else:
        weights = np.ones_like(dico[keys[0]])
    
    if verbose:
        print("- using params")
        print(keys)
        print("- using nsample = ", len(weights))
        
    return dico, weights




# #Giving a chain file and a data file, get the indices in the data file that correspond to 
# #scale cuts used in chain
# def get_scale_cuts(chain_file, data_file, print_out=False, return_zbins=False, file_mode='rb', like_module='2pt_like', data_sets_realizations=None, num_observables=None):

#     fits_data = fits.open(data_file)

#     #Get observables
#     if num_observables is None:
#         num_observables = len(fits_data)-4
#     data_sets = []
#     start_indices = []
#     header = fits_data['COVMAT'].header
#     len_data = fits_data['COVMAT'].data.shape[0]
#     for oi in range(0,num_observables):
#         data_sets.append(header['NAME_' + str(oi)])
#         start_indices.append(header['strt_' + str(oi)])
    
#     #Current hardcoded
#     min_theta = 2.5
#     max_theta = 250.0

#     #First we use chain to get list of indices in angular array that get included
#     with open(chain_file, file_mode) as file:
#         # at_last_scalecut_line = False
#         observable_list = []
#         previous_observable = ''
        
#         #Labels  for each angular scale
#         all_observable_good_indices = []
#         all_observable_zbini = []
#         all_observable_zbinj = []
#         #label for each correlation function
#         all_bini = []
#         all_binj = []
        
#         this_observable_good_indices = []
#         #this_observable_zbini = []
#         #this_observable_zbinj = []
#         this_bini = []
#         this_binj = []
#         #number of redshift bins
#         num_redshift_bins = 0
 
#         likelihood_module_reached = False
#         cut_bins = {}
#         for idx, line in enumerate(file):
#             if not likelihood_module_reached and not line.startswith("## ["+like_module+"]"):
#                 continue
#             else:
#                 if not likelihood_module_reached:
#                     if print_out:
#                         print("Reading options for likelihood module: ", like_module)
#                 likelihood_module_reached = True
                
#             #If line is a bin cut line
#             if "## cut_" in line:
#                 # print(line)
#                 line_sp = line.split(' ')
#                 observable = line_sp[1][4:]
#                 cut_bins[observable] = []
#                 for _x in line_sp[3:]:
#                     xi, xj = _x.split(',')
#                     if print_out:
#                         print("Cutting",observable,int(xi),int(xj))
#                     cut_bins[observable].append((int(xi),int(xj)))
                
#             #If line is a scale cut line
#             if ('## angle_range_' in line):
#                 line_sp = line.split()#' ')
#                 observable = line_sp[1][12:-4]
                
#                 nth = fits_data[observable].header['N_ANG']
#                 theta_bins = np.exp(np.linspace(np.log(min_theta), np.log(max_theta), num = nth+1))
#                 log_bins = np.linspace(np.log(min_theta), np.log(max_theta), num = 21)
#                 theta_centers = np.exp(0.5*(log_bins[1:] + log_bins[:-1]))
           
#                 #If we're at a new observable, then save the old one
#                 if (observable != previous_observable and previous_observable != ''):
#                     #all_observable_redshiftbins.append(num_redshift_bins)
#                     observable_list.append(previous_observable)
#                     all_observable_good_indices.append(this_observable_good_indices)
#                     all_bini.append(this_bini)
#                     all_binj.append(this_binj)
#                     #all_observable_zbini.append(this_observable_zbini)
#                     #all_observable_zbinj.append(this_observable_zbinj)
                    
#                     #reset stuff
#                     this_observable_good_indices = []
#                     #this_observable_zbini = []
#                     #this_observable_zbinj = []
#                     this_bini = []
#                     this_binj = []
#                     num_redshift_bins = 0
                
#                 bin_string_sp = line_sp[1].split('_')
#                 bini = int(bin_string_sp[-2])
#                 binj = int(bin_string_sp[-1])
#                 this_bini.append(bini)
#                 this_binj.append(binj)
                
#                 # print(line_sp)
#                 min_theta_cut = float(line_sp[3])
#                 max_theta_cut = float(line_sp[4])

#                 print("gerrgregwererg")                
#                 if observable in cut_bins.keys():
#                     print("gererg")
#                     if (bini,binj) in cut_bins[observable]:
#                         min_theta_cut = np.inf
#                         max_theta_cut = -np.inf
#                         print("Applying cuts")
                
#                 #good = (np.where((theta_bins[:-1] >= min_theta_cut) & (theta_bins[1:] <= max_theta_cut))[0]).tolist()
#                 good = np.where((theta_centers >= min_theta_cut) & (theta_centers <= max_theta_cut))[0]
#                 this_observable_good_indices.append(good)
                
#                 #this_observable_zbini.append(np.array(bini + np.zeros(len(good))))
#                 #this_observable_zbinj.append(np.array(binj + np.zeros(len(good))))
                
#                 previous_observable =  observable
                
#             #line is not a scale cut line, but previous line was then we know we're at the end
#             elif (previous_observable != ''):
#                 observable_list.append(previous_observable)
#                 all_bini.append(this_bini)
#                 all_binj.append(this_binj)
#                 all_observable_good_indices.append(this_observable_good_indices)
#                 #all_observable_zbini.append(this_observable_zbini)
#                 #all_observable_zbinj.append(this_observable_zbinj)
#                 break
    
#     print("observable_list", observable_list)
#     print("all_bini", all_bini)
#     print("all_binj", all_binj)
#     print("all_observable_good_indices", all_observable_good_indices    )
#     print("cut_bins", cut_bins    )

#     #Now we match up the indices from above to indicies in the full length data vector or covariance
#     all_include_indices = []
#     all_zbini_indices = []
#     all_zbinj_indices = []
    
#     #Loop over observables in data vector
#     for di in range(0,len(data_sets)):
#         data_set_name = data_sets[di]
        
#         include_indices = []
#         try:
#             match_index = observable_list.index(data_sets[di])
#         except ValueError:
#             match_index = -1
            
#         #If there are no scale cuts specified, then include all scales
#         if (match_index == -1):
#             if (di == len(data_sets)-1):
#                 end_index = len_data
#             else:
#                 end_index = start_indices[di+1]
#             include_indices = np.arange(start_indices[di], end_index).tolist()
#             if (print_out):
#                 print (data_set_name)
#                 print ("   including indices = ", include_indices)
#             all_include_indices.append(include_indices)

#             zbini_indices =  fits_data[data_sets[di]].data['bin1'].tolist()
#             zbinj_indices =  fits_data[data_sets[di]].data['bin2'].tolist()
#             all_zbini_indices.append(zbini_indices)
#             all_zbinj_indices.append(zbinj_indices)

#         #if there are scale cuts specified, then use those
#         if (match_index != -1):
#             start_index = start_indices[di]
#             this_bini = np.array(all_bini[match_index])
#             this_binj = np.array(all_binj[match_index])

#             fits_i = fits_data[data_set_name]
#             nth = fits_i.header['N_ANG']
#             nz1 = fits_i.header['N_ZBIN_1']
#             nz2 = fits_i.header['N_ZBIN_2']     
#             #Ordering of redshift bins
#             cov_bini_all = fits_i.data['bin1'][nth*np.arange(len(fits_i.data['bin1'])//nth)]
#             cov_binj_all = fits_i.data['bin2'][nth*np.arange(len(fits_i.data['bin2'])//nth)]

#             #will store info for this observable
#             include_indices_thisobservable = []
#             zbini_indices_thisobservable = []
#             zbinj_indices_thisobservable = []
                        
#             #match bins
#             redshift_ij = 0
#             for pair_index in range(0,len(cov_bini_all)):
#                 #print "zbini_index = ", zbini_index, "zbinj_index = ", zbinj_index
#                 cov_bini = cov_bini_all[pair_index]
#                 cov_binj = cov_binj_all[pair_index]
#                 bin_match = (np.where((cov_bini == this_bini) & (cov_binj == this_binj))[0])
#                 if (print_out):
#                     print (data_set_name + " zi = ",cov_bini, " zj = ", cov_binj)

#                 if (len(bin_match) == 1):
#                     include_indices = (all_observable_good_indices[match_index])[bin_match[0]]
#                     for ii in range(0,len(include_indices)):
#                         include_indices[ii] = include_indices[ii] + start_index + nth*redshift_ij 
#                     redshift_ij += 1
#                     if (print_out):
#                         print ("   including indices = ", include_indices)
#                     include_indices_thisobservable.append(include_indices)
                    
#                     zbini_indices = (np.zeros(len(include_indices)).astype('int') + cov_bini).tolist()
#                     zbinj_indices = (np.zeros(len(include_indices)).astype('int') + cov_binj).tolist()
#                     zbini_indices_thisobservable.append(zbini_indices)
#                     zbinj_indices_thisobservable.append(zbinj_indices)
#                 else:
#                     if (print_out):
#                         print ("   redshift bin combination excluded")                           
                            
#             all_include_indices.append(include_indices_thisobservable)
#             all_zbini_indices.append(zbini_indices_thisobservable)
#             all_zbinj_indices.append(zbinj_indices_thisobservable)

#     if (not return_zbins):
#         return data_sets, all_include_indices
#     else:
#         assert len(data_sets) == len(all_include_indices) == len(all_zbini_indices) == len(all_zbinj_indices)
#         dico_out = OrderedDict()
#         count = 0
#         if data_sets_realizations is None:
#             iiiiiii = range(len(data_sets))
#         else:
#             iiiiiii = [data_sets.index(x) for x in data_sets_realizations]
#         for i in iiiiiii:
# #             print(data_sets[i])
#             dico_out[data_sets[i]] = OrderedDict()
#             assert len(all_zbini_indices[i]) == len(all_zbinj_indices[i])
#             if print_out:
#                 print(data_sets[i], all_zbini_indices[i])
#             for j in range(len(all_zbini_indices[i])):
#                 if len(as_list(all_zbini_indices[i][j]))>0:
#                     bini = as_list(all_zbini_indices[i][j])[0]
#                     binj = as_list(all_zbinj_indices[i][j])[0]
#                     assert np.all([x==bini for x in as_list(all_zbini_indices[i][j])])
#                     assert np.all([x==binj for x in as_list(all_zbinj_indices[i][j])])
#                     dico_out[data_sets[i]][bini,binj] = {}
#                     dico_out[data_sets[i]][bini,binj]['idx_full'] = as_list(all_include_indices[i][j])
#                     _n = len(dico_out[data_sets[i]][bini,binj]['idx_full'])
#                     dico_out[data_sets[i]][bini,binj]['idx_sub'] = np.arange(_n) + count
#                     count += _n

                
#         return dico_out
# #         return data_sets, all_include_indices, all_zbini_indices, all_zbinj_indices

# if (__name__ == '__main__'):
#     data_file = 'internal_6x2pt_1225_fiducial_rebin_wtheorycov_cov_5x2pt_v12.18_3x2pt_v111417_GK_v050818mcal_NK_v050818_dvecunblinded_noisecorrectedtheorycov_CORRECTED_FIXTYPE.fits'
#     chain_file = 'd_3x2_chain.txt'
#     data_vec = fits.open(data_file)
#     data_sets, include_indices, zbin_indices = get_scale_cuts(chain_file, data_file, print_out = False, return_zbins = True)
#     pdb.set_trace()



