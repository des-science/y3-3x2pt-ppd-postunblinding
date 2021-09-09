# How to do PPD for Y3?
This README first gives an overview of the process for internal consistency tests using the PPD and then dives into details about the PPD module and sampler.

## 3 steps of the PPD tests
1. getting a posterior sample from some data d, ie θ~P(θ|d),
2. using that sample to generate data d' (two cases here),
3. comparing simulated d' to observed d' (to get, eg, a p-value),

## How to do that?
First, determine what kind of test you're running:
- Goodness of fit test: in this case d' and d denote uncorrelated realizations of the same measurements.
- Consistency test: in this case, d' and d denote disjoint subsets of the full set of measurements and they are correlated, eg d'=shear based on d=2x2.

Then, proceed as follows for the steps above.
1. Get a posterior sample from d alone from cosmosis by applying appropriate cuts for d in the likelihood module.
2. Generate samples of d'. To do so, we run a modified version the cosmosis list sampler with the normal pipeline and an additional ppd module, which takes posterior samples θ from step 1. See an example in eg https://github.com/des-science/y3-3x2pt-ppd/blob/master/3x2_goodnessoffit/ppd.ini
	- For each θ, this computes the theory for d', denoted d'(θ) here, and then draws d' (and saves it to disk). Two cases:
		- For a goodness of fit test, the module simply samples the likelihood for d' (which is the same as d).
		- For a consistency test, the module samples the likelihood for d' conditioned on observed d (that's the big difference).
	- For a nested sample, it is much faster than step 1 since it requires a number of pipeline evaluations equal to the number of samples (so of order 10k instead of 10^5 to 10^6).
3. Analyze/postprocess generated d' samples (aka PPD samples) vs observed d'.
	- For this, we proposed comparing chi2's computed between a) observed d' and d'(θ), and b) sampled d' at θ and d'(θ). The raw p-value is simply the fraction of chi2_b > chi2_a, ie the fraction of posterior samples (with their weights) for which the generated data is further from predictions than the observed data.
	- Additionally, we "calibrate" this p-value, which tends can be biased low for consistency tests, because of discrepant constraints on parameters between d and d'. The most extreme case of this (which we don't use), is if you were to split data into d and d' such that d' depends on some parameter that is fully unconstrained by d, eg galaxy bias if d'=w(θ) and d=shear. The most up-to-date way to do this is illustrated in https://github.com/des-science/y3-3x2pt-ppd/blob/master/postprocessing/ppd_y1_data.ipynb. The full "analysis/postprocessing" part consists in two function calls.
		- Note this is related to the choice of metric we use to analyze PPD samples, not a shortcoming of how we sampled it. The basic reason is that we're trying to compare distributions in ~500-dimensional space, which requires us to make simplifications.
		- To calibrate the p-value, we generate consistent d and d' by sampling the likelihood at some fiducial cosmology and apply the same PPD test to them, to get a raw p-value for each. The calibrated p-value is then given by the fraction of those simulated data vectors with a raw p-value smaller than the observed ones.
		- Since we can't get posterior samples from each of those, we apply importance sampling to some posterior, for instance the one used above. The thing to watch for is the effective number of samples: if they get too low, p-values for the simulated data vectors are unreliable, so comparing them to the p-value for observed data can be misleading.
        
        
# PPD sampler
In order to save the PPD output files correctly, we use a new sampler called `listppd`, which is simply the `list` sampler with additional functionality. Its options are the same as the list sampler, plus the following ones
| Parameters | Values |
| -- | -- |
| filename |  path/to/the/chain.txt|
| ppd_output_file | Where to save PPD realizations, usually `%(OUTDIR)s/ppd_realizations_%(RUN)s.txt`|
| theory_output_file | Where to save PPD theory DVs, usually  `%(OUTDIR)s/ppd_theory_%(RUN)s.txt`|

To use it:
1. Copy the folder `listppd` into the `samplers` folder of cosmosis.
2. Append `from .listppd import listppd_sampler` to `samplers/__init__.py` so that it is registered.

3. In the params.ini file, change the sampler to `listppd` and add
    ```
    filename = path/to/the/chain.txt
    ppd_output_file=%(OUTDIR)s/ppd_realizations_%(RUN)s.txt
    theory_output_file=%(OUTDIR)s/ppd_theory_%(RUN)s.txt
    ```



 
# PPD module

Define the following additional variables to determine observables in d an dprime.
```
[DEFAULT]
2PT_DATA_SETS_D = xip xim wtheta gammat
2PT_DATA_SETS_DPRIME = xip xim wtheta gammat
```

The module can work in several modes. Common parameters are:

| Parameters | Values |
| -- | -- |
| statistic | Only accepts `chi2` so far |
| ppd_d_names | Names of the observables for d, eg `xip xim wtheta gammat`|
| ppd_dprime_names | Names of the observables for dprime, eg `xip xim wtheta gammat`| 
| condition_on_d | Boolean, whether to condition dprime on d|
| use_like_cuts | Determines scale cuts for d and dprime. If `use_like_cuts=0`, scale cuts are those of `2pt_like` and observanles are determined by `ppd_d_names` and `ppd_dprime_names`. If `use_like_cuts=1`, uses cuts from `2pt_dprime` (see below) for dprime and sets d to be the complementary of dprime in `2pt`. If `use_like_cuts=2`, uses cuts from `2pt_d` for d and `2pt_dprime` for dprime (don't use it).|


Then, several options.

## Goodness-of-fit tests

Here d and dprime are the same observables and are independent.

1. Likelihood options (to ensure the data vector is saved in the block)
    ```
    [like_2pt]
    likelihood_only=F
    data_file = %(INPUT_2PT_FILE)s
    data_sets = %(2PT_DATA_SETS)s
    ```
1. PPD module options
    ```
    [ppd]
    statistic=chi2
    ppd_d_names = %(2PT_DATA_SETS_D)s
    ppd_dprime_names = %(2PT_DATA_SETS_DPRIME)s
    condition_on_d=F
    use_like_cuts=0
    ```

1. Pipeline modifications
    - Add `ppd` to the list of modules.
    - To save chi2 statistics, add `data_vector/2pt_chi2 ppd/chi2_data ppd/chi2_realization` to `extra_output`.

## Consistency tests

Here dprime is conditoned on d, eg when applying consistency tests between observables and/or scales. In this case, use `condition_on_d=T` and `use_like_cuts=1` or `use_like_cuts=2`.

1. You need to run two or three versions of the 2pt likelihood module (it's fast and determine scale cuts without errors).
    - `like_2pt` is the likelihood with fiducial cuts, only add
        ```
        [like_2pt]
        likelihood_only=F
        like_name = 2pt
        data_file = %(INPUT_2PT_FILE)s
        data_sets = %(2PT_DATA_SETS)s
        ```
    - `like_2pt_dprime` is the likelihood with cuts applied to dprime, defined using standard `angle_range_` and `cut_` options.
        ```
        [like_2pt_d]
        likelihood_only=F
        like_name = 2pt_dprime
        data_file = %(INPUT_2PT_FILE)s
        data_sets = %(2PT_DATA_SETS_DPRIME)s
        ; Optional scale cuts, eg for large vs small scales, or bin cuts, eg for bin i vs no bin i.
        ```
    - (optional )`like_2pt_d` is the likelihood with cuts applied to d, defined using standard `angle_range` options.
        ```
        [like_2pt_d]
        likelihood_only=F
        like_name = 2pt_d
        data_file = %(INPUT_2PT_FILE)s
        data_sets = %(2PT_DATA_SETS_D)s
        ; Optional scale cuts, eg for large vs small scales, or bin cuts, eg for bin i vs no bin i. If use_like_cuts=1, then cuts will not be used and d is the complementary of dprime.
        ```
   
1. PPD module options
    ```
    [ppd]
    statistic=chi2
    ppd_d_names = %(2PT_DATA_SETS_D)s
    ppd_dprime_names = %(2PT_DATA_SETS_DPRIME)s
    condition_on_d=T
    use_like_cuts=1
    ```
1. Pipeline modifications
    - Add `2pt_like 2pt_like_d 2pt_like_dprime ppd` to the list of modules.
    - To save chi2 statistics, add `data_vector/2pt_chi2 ppd/chi2_data ppd/chi2_realization` to `extra_output`.





