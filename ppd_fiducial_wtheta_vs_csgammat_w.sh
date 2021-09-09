#!/bin/bash -l
#SBATCH -A des
#SBATCH -N 12
#SBATCH --tasks-per-node=32
#SBATCH -C haswell
#SBATCH -q regular
#SBATCH -J ppd
#SBATCH --mail-user=cdoux@sas.upenn.edu
#SBATCH --mail-type=ALL
#SBATCH -t 00:30:00

echo $SHELL
echo ${SLURM_NTASKS}

conda deactivate

source ~/codes/cosmosis/config/setup-cosmosis-nersc-zsh

# These should be the same as the corresponding fiducial chain script
export RUNNAME=cs_gt_fiducial_lcdm_unblind_02_24_21_covupdatev2_wnz_sr
export DATAFILE=des-y3/2pt_NG_final_2ptunblind_02_24_21_wnz_covupdate.v2.fits
export DATAFILE_SR=des-y3/2pt_NG_final_2ptunblind_02_24_21_wnz_covupdate_sr.npy
export SCALEFILE=fiducial/scales_3x2pt_0.5_8_6_v0.4.ini
export INCLUDEFILE=blank_include.ini
export VALUESINCLUDE=fiducial/values_w.ini
export PRIORSINCLUDE=blank_include.ini

export OMP_NUM_THREADS=2

cd ..
srun -n ${SLURM_NTASKS} cosmosis --mpi ppd/fiducial_wtheta_vs_csgammat/ppd.ini
cd ppd
