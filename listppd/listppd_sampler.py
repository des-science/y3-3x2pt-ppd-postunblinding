from __future__ import print_function
from builtins import zip
from builtins import map
from builtins import str
from builtins import range
import itertools
import numpy as np
from cosmosis.output.text_output import TextColumnOutput
from cosmosis.samplers import ParallelSampler
import pickle

keys = ['dprime_true', 'dprime_realization', 'dprime_theory', 'd_true', 'd_theory']

def task(p):
    i,p = p
    results = listppd_sampler.pipeline.run_results(p, all_params=True)

    if listppd_sampler.dprime_real_output_file:
        if i==0:
            # inv_cov_dprime_chi2 = results.block['data_vector', '2pt_dprime_inverse_covariance'] 
            # inv_cov_d_chi2 = results.block['data_vector', '2pt_d_inverse_covariance']

            # cov_pm_dprime = np.linalg.inv(inv_cov_dprime_chi2)
            # cov_pm_d = np.linalg.inv(inv_cov_d_chi2)

            # np.savetxt(listppd_sampler.ppd_output_file_basename + '_dprime_cov_pm.txt', cov_pm_dprime)
            # np.savetxt(listppd_sampler.ppd_output_file_basename + '_d_cov_pm.txt', cov_pm_d)

            inv_cov_dprime = results.block['data_vector', '2pt_dprime_inverse_covariance'] 
            inv_cov_d = results.block['data_vector', '2pt_d_inverse_covariance']
            np.save(listppd_sampler.ppd_output_file_basename + '_dprime_inv_cov_pm.npy', inv_cov_dprime)
            np.save(listppd_sampler.ppd_output_file_basename + '_d_inv_cov_pm.npy', inv_cov_d)

            cov_dprime = results.block['data_vector', '2pt_dprime_covariance'] 
            cov_d = results.block['data_vector', '2pt_d_covariance'] 
            np.save(listppd_sampler.ppd_output_file_basename + '_dprime_cov.npy', cov_dprime)
            np.save(listppd_sampler.ppd_output_file_basename + '_d_cov.npy', cov_d)

            for like_name in ['xip', 'xim', '1x2', 'gammat', 'wtheta', '2x2']:
                try:
                    _icov_pm = results.block['data_vector', '2pt_{}_inverse_covariance'.format(like_name)]
                    np.save(listppd_sampler.ppd_output_file_basename + '_{}_inv_cov_pm.npy'.format(like_name), _icov_pm)
                except:
                    pass

        a = {}
        try:
            a['omega_m'] = results.block['cosmological_parameters','omega_m']
            a['A_s'] = results.block['cosmological_parameters','A_s']
            for k in keys:
                a[k] = results.block['ppd', k]
        except:
            a['omega_m'] = -1.
            a['A_s'] = -1.
            for k in keys:
                a[k] = None
        return results.post, (results.prior, results.extra), a
    else:
        return results.post, (results.prior, results.extra)




class ListPPDSampler(ParallelSampler):
    parallel_output = False
    sampler_outputs = [("prior", float), ("post", float)]

    def config(self):
        global listppd_sampler
        listppd_sampler = self

        self.converged = False
        self.filename = self.read_ini("filename", str)
        self.ppd_output_file_basename = self.read_ini("ppd_output_file_basename", str, "")
        self.burn = self.read_ini("burn", int, 0)
        self.thin = self.read_ini("thin", int, 1)
        limits = self.read_ini("limits", bool, False)
        self.random = self.read_ini("random", bool, False)
        self.nsamples = self.read_ini("nsamples", int, 0)
        if self.random:
            assert self.nsamples > 0
        
        self.size_chunk = self.read_ini("size_chunk", int, -1)
        self.i_chunk = self.read_ini("i_chunk", int, -999)

        if self.size_chunk>0:
            assert self.i_chunk>=0
            self.ppd_output_file_basename += '_'+str(self.i_chunk)

        self.d_theory_output_file = self.ppd_output_file_basename + '_d_theory.npy' #self.read_ini("d_theory_output_file", str, "")
        self.dprime_theory_output_file = self.ppd_output_file_basename + '_dprime_theory.npy' #self.read_ini("dprime_theory_output_file", str, "")
        self.dprime_real_output_file = self.ppd_output_file_basename + '_dprime_real.npy' #self.read_ini("dprime_real_output_file", str, "")


        #overwrite the parameter limits
        if not limits:
            if self.output is not None:
                self.output.columns = []
            for p in self.pipeline.parameters:
                p.limits = (-np.inf, np.inf)
                if self.output is not None:
                    self.output.add_column(str(p), float)
            if self.output is not None:
                for p in self.pipeline.extra_saves:
                    self.output.add_column('{}--{}'.format(*p), float)
                for p,ptype in self.sampler_outputs:
                    self.output.add_column(p, ptype)

        if self.dprime_real_output_file:
            assert len(self.dprime_theory_output_file)>0
            self.dv_ppd_dict = {k:{} for k in keys}
            


    def execute(self):
        print("### List sampler PPD")
        print("# Starting execute")

        #Load in the filename that was requested
        file_options = {"filename":self.filename}
        column_names, samples, _, _, _ = TextColumnOutput.load_from_options(file_options)
        # samples = samples[0]

        # Apply burn
        samples = samples[0][self.burn:]

        # Pick samples
        if self.random:
            which = np.random.choice(len(samples), size=self.nsamples, replace=False)
            samples = [samples[w] for w in which]
        else:
            samples = [samples[i] for i in range(0, len(samples), self.thin)] 
        
        # find where in the parameter vector of the pipeline
        # each of the table parameters can be found
        replaced_params = []
        for i,column_name in enumerate(column_names):
            # ignore additional columns like e.g. "like", "weight"
            try:
                section,name = column_name.split('--')
            except ValueError:
                print("Not including column %s as not a cosmosis name" % column_name)
                continue
            section = section.lower()
            name = name.lower()
            # find the parameter in the pipeline parameter vector
            # may not be in there - warn about this
            try:
                j = self.pipeline.parameters.index((section,name))
                replaced_params.append((i,j))
            except ValueError:
                print("Not including column %s as not in values file" % column_name)

        #Create a collection of sample vectors at the start position.
        #This has to be a list, not an array, as it can contain integer parameters,
        #unlike most samplers
        v0 = self.pipeline.start_vector(all_params=True, as_array=False)
        sample_vectors = [v0[:] for i in range(len(samples))]
        #Fill in the varied parameters. We are not using the
        #standard parameter vector in the pipeline with its 
        #split according to the ini file
        for s, v in zip(samples, sample_vectors):
            for i,j in replaced_params:
                v[j] = s[i]

        #Turn this into a list of jobs to be run 
        #by the function above
        if self.size_chunk>0:
            assert self.i_chunk>=0
            sample_index = list(range(self.size_chunk*self.i_chunk, min(len(sample_vectors), self.size_chunk*(self.i_chunk+1)), self.thin))
            sample_vectors = [sample_vectors[i] for i in sample_index]
            print("### List sampler")
            print("# Running on nsamples = [{}-{}]".format(sample_index[0], sample_index[-1]))

        else:
            sample_index = list(range(len(sample_vectors)))
        jobs = list(zip(sample_index, sample_vectors))

        print("### List sampler PPD")
        print("# Running on nsamples =", len(jobs))

        #Run all the parameters
        #This only outputs them all at the end
        #which is a bit problematic, though you 
        #can't use MPI and retain the output ordering.
        #Have a few options depending on whether
        #you care about this we should think about
        #(also true for grid sampler).
        if self.pool:
            results = self.pool.map(task, jobs)
        else:
            results = list(map(task, jobs))

        print("### List sampler PPD")
        print("# Pipeline finished, gathering outputs")
        #Save the results of the sampling
        #We now need to abuse the output code a little.
        for idx, sample, result in zip(sample_index, sample_vectors, results):
            #Optionally save all the results calculated by each
            #pipeline run to files
            if self.dprime_real_output_file:
                (prob, (prior,extra), a) = result
                if a is not None:
                    self.dv_ppd_dict[idx] = a
            else:
                (prob, (prior,extra)) = result
            #always save the usual text output
            self.output.parameters(sample, extra, prior, prob)
        #We only ever run this once, though that could 
        #change if we decide to split up the runs
        self.converged = True

        print("### List sampler PPD")
        print("# Saving results to files")
        if self.ppd_output_file_basename:
            # Get true data DV
            dprime_true = None
            # idx = 0
            idx = sample_index[0]
            # while dprime_true is None:
            #     dprime_true = self.dv_ppd_dict[idx]['dprime_true']
            #     d_true = self.dv_ppd_dict[idx]['d_true']
            #     idx +=1
            # with open(self.dprime_real_output_file, "w+") as thefile:
            #     output_string = np.array2string(dprime_true, max_line_width = 1000000)[1:-1] + '\n'
            #     thefile.write(output_string)
            # with open(self.dprime_theory_output_file, "w+") as thefile:
            #     output_string = np.array2string(dprime_true, max_line_width = 1000000)[1:-1] + '\n'
            #     thefile.write(output_string)
            # with open(self.d_theory_output_file, "w+") as thefile:
            #     output_string = np.array2string(d_true, max_line_width = 1000000)[1:-1] + '\n'
            #     thefile.write(output_string)
            dprime_true = self.dv_ppd_dict[idx]['dprime_true']
            d_true = self.dv_ppd_dict[idx]['d_true']
            np.save(self.ppd_output_file_basename + '_dprime_true.npy', dprime_true)
            np.save(self.ppd_output_file_basename + '_d_true.npy', d_true)

            # for idx in sample_index:
            #     if self.dv_ppd_dict[idx]['dprime_true'] is not None:
            #         with open(self.dprime_real_output_file, "a") as thefile:
            #             output_array = np.append(self.dv_ppd_dict[idx]['omega_m'], np.append(self.dv_ppd_dict[idx]['A_s'], self.dv_ppd_dict[idx]['dprime_realization']))
            #             output_string = np.array2string(output_array, max_line_width = 1000000)[1:-1] + '\n'
            #             thefile.write(output_string)
            #         with open(self.dprime_theory_output_file, "a") as thefile:
            #             output_array = np.append(self.dv_ppd_dict[idx]['omega_m'], np.append(self.dv_ppd_dict[idx]['A_s'], self.dv_ppd_dict[idx]['dprime_theory']))
            #             output_string = np.array2string(output_array, max_line_width = 1000000)[1:-1] + '\n'
            #             thefile.write(output_string)
            #         with open(self.d_theory_output_file, "a") as thefile:
            #             output_array = np.append(self.dv_ppd_dict[idx]['omega_m'], np.append(self.dv_ppd_dict[idx]['A_s'], self.dv_ppd_dict[idx]['d_theory']))
            #             output_string = np.array2string(output_array, max_line_width = 1000000)[1:-1] + '\n'
            #             thefile.write(output_string)

            files = [self.dprime_real_output_file, self.dprime_theory_output_file, self.d_theory_output_file]
            data_keys = ['dprime_realization', 'dprime_theory', 'd_theory']

            # for file, data_key in zip(files, data_keys):
            #     with open(file, "a") as thefile:
            #         for idx in sample_index:
            #             if self.dv_ppd_dict[idx]['dprime_true'] is not None:
            #                 output_array = np.append(self.dv_ppd_dict[idx]['omega_m'], np.append(self.dv_ppd_dict[idx]['A_s'], self.dv_ppd_dict[idx][data_key]))
            #                 output_string = np.array2string(output_array, max_line_width = 1000000)[1:-1] + '\n'
            #                 thefile.write(output_string)

            for file, data_key in zip(files, data_keys):
                output_array = []
                for idx in sample_index:
                    temp = np.append(self.dv_ppd_dict[idx]['omega_m'], np.append(self.dv_ppd_dict[idx]['A_s'], self.dv_ppd_dict[idx][data_key]))
                    output_array.append(temp)
                output_array = np.array(output_array)
                np.save(file, output_array)

            

    def is_converged(self):
        return self.converged
