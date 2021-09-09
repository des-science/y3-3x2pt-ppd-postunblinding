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

keys = ['dprime_true', 'dprime_realization', 'dprime_theory']

def task(p):
    i,p = p
    results = list_sampler_ppd.pipeline.run_results(p, all_params=True)

    # We use omega_m and A_s just to double check the order of things at the end
    if list_sampler_ppd.ppd_output_file:
        a = {}
        a['omega_m'] = block['cosmological_parameters','omega_m']
        a['A_s'] = block['cosmological_parameters','A_s']
        try:
            for k in keys:
                a[k] = results.block['ppd', k]
        except:
            for k in keys:
                a[k] = None
        return results.post, (results.prior, results.extra), a
    else:
        return results.post, (results.prior, results.extra)




class ListSamplerPPD(ParallelSampler):
    parallel_output = False
    sampler_outputs = [("prior", float), ("post", float)]

    def config(self):
        global list_sampler_ppd
        list_sampler_ppd = self

        self.converged = False
        self.filename = self.read_ini("filename", str)
        self.ppd_output_file = self.read_ini("ppd_output_file", str, "")
        self.theory_output_file = self.read_ini("theory_output_file", str, "")
        self.burn = self.read_ini("burn", int, 0)
        self.thin = self.read_ini("thin", int, 1)
        limits = self.read_ini("limits", bool, False)
        self.random = self.read_ini("random", bool, False)
        self.nsamples = self.read_ini("nsamples", int, 0)
        if self.random:
            assert self.nsamples > 0
        
        self.size_chunk = self.read_ini("size_chunk", int, -1)
        self.i_chunk = self.read_ini("i_chunk", int, -999)

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

        if self.ppd_output_file:
            assert len(self.theory_output_file)>0
            self.dv_ppd_dict = {k:{} for k in keys}
            


    def execute(self):

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
            sample_index = list(range(self.size_chunk*self.i_chunk, self.size_chunk*(self.i_chunk+1), self.thin))
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

        #Save the results of the sampling
        #We now need to abuse the output code a little.
        for idx, sample, result in zip(sample_index, sample_vectors, results):
            #Optionally save all the results calculated by each
            #pipeline run to files
            if self.ppd_output_file:
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

        if self.ppd_output_file:
            # Get true data DV
            dprime_true = None
            idx = 0

            # First add the observed dprime in the file
            while dprime_true is None:
                dprime_true = self.dv_ppd_dict[idx]['dprime_true']
                idx +=1
            with open(self.ppd_output_file, "w+") as thefile:
                output_string = np.array2string(dprime_true, max_line_width = 1000000)[1:-1] + '\n'
                thefile.write(output_string)
            with open(self.theory_output_file, "w+") as thefile:
                output_string = np.array2string(dprime_true, max_line_width = 1000000)[1:-1] + '\n'
                thefile.write(output_string)

            # Then add for each sample, Om, As, realization, theory
            for idx in sample_index:
                if self.dv_ppd_dict[idx]['dprime_true'] is not None:
                    with open(self.ppd_output_file, "a") as thefile:
                        output_array = np.append(self.dv_ppd_dict[idx]['omega_m'], np.append(self.dv_ppd_dict[idx][,'A_s'], self.dv_ppd_dict[idx]['dprime_realization']))
                        output_string = np.array2string(output_array, max_line_width = 1000000)[1:-1] + '\n'
                        thefile.write(output_string)
                    with open(self.theory_output_file, "a") as thefile:
                        output_array = np.append(self.dv_ppd_dict[idx]['omega_m'], np.append(self.dv_ppd_dict[idx][,'A_s'], self.dv_ppd_dict[idx]['dprime_theory']))
                        output_string = np.array2string(output_array, max_line_width = 1000000)[1:-1] + '\n'
                        thefile.write(output_string)

    def is_converged(self):
        return self.converged
