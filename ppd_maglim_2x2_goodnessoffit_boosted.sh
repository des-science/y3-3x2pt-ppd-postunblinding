#!/bin/bash -l
#SBATCH -A des
#SBATCH -N 1
#SBATCH --tasks-per-node=32
#SBATCH -C haswell
#SBATCH -q regular
#SBATCH -J ppd
#SBATCH --mail-user=cdoux@sas.upenn.edu
#SBATCH --mail-type=ALL
#SBATCH -t 00:30:00
#SBATCH --array=1-221

echo $SHELL
echo ${SLURM_NTASKS}

conda deactivate

source ~/codes/cosmosis/config/setup-cosmosis-nersc-zsh

# These should be the same as the corresponding maglim chain script
export RUNNAME=2x2pt_lcdm_SR_maglim_boosted
export DATAFILE=des-y3/2pt_NG_final_2ptunblind_02_26_21_wnz_maglim_covupdate.fits
export DATAFILE_SR=des-y3/2pt_NG_final_2ptunblind_02_26_21_wnz_maglim_covupdate_sr.npy
export SCALEFILE=maglim/scales_3x2pt_8_6_0.5_v0.40.ini
export INCLUDEFILE=maglim/params_SR.ini
export VALUESINCLUDE=maglim/values.ini
export PRIORSINCLUDE=maglim/priors.ini

export OMP_NUM_THREADS=2

cd ..
srun -n ${SLURM_NTASKS} cosmosis --mpi ppd/maglim_2x2_goodnessoffit/ppd.ini -p listppd.size_chunk=1000 listppd.i_chunk=${SLURM_ARRAY_TASK_ID} output.filename=ppd/maglim_2x2_goodnessoffit/ppd_chain_${RUNNAME}_maglim_2x2_goodnessoffit_${SLURM_ARRAY_TASK_ID}.txt
cd ppd
