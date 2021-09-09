#!/bin/bash -l
#SBATCH -A des
#SBATCH -N 12
#SBATCH --tasks-per-node=32
#SBATCH -C haswell
#SBATCH -q debug
#SBATCH -J ppd
#SBATCH --mail-user=cdoux@sas.upenn.edu
#SBATCH --mail-type=ALL
#SBATCH -t 00:30:00

echo $SHELL
echo ${SLURM_NTASKS}

conda deactivate

source ~/codes/cosmosis/config/setup-cosmosis-nersc-zsh

# These should be the same as the corresponding fiducial chain script
export RUNNAME=1x2_nobin2_ML
export DATAFILE=des-y3/2pt_NG_final_2ptunblind_02_26_21_wnz_maglim_covupdate.fits
export DATAFILE_SR=des-y3/2pt_NG_final_2ptunblind_02_26_21_wnz_maglim_covupdate_sr.npy
export SCALEFILE=maglim/scales_3x2pt_8_6_0.5_v0.40.ini
export INCLUDEFILE=maglim/params_SR.ini
export VALUESINCLUDE=maglim/values.ini
export PRIORSINCLUDE=maglim/priors.ini

export OMP_NUM_THREADS=2

cd ..
srun -n ${SLURM_NTASKS} cosmosis --mpi ppd/1x2bin2_vs_1x2nobin2/ppd.ini
cd ppd