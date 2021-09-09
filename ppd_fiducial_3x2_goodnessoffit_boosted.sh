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
#SBATCH --array=0-200

echo $SHELL
echo ${SLURM_NTASKS}

conda deactivate

source ~/codes/cosmosis/config/setup-cosmosis-nersc-zsh

# These should be the same as the corresponding fiducial chain script
export RUNNAME=3x2lcdm_0321_boosted
export DATAFILE=des-y3/2pt_NG_final_2ptunblind_02_24_21_wnz_covupdate.v2.fits
export DATAFILE_SR=des-y3/2pt_NG_final_2ptunblind_02_24_21_wnz_covupdate_sr.npy
export SCALEFILE=fiducial/scales_3x2pt_0.5_8_6_v0.4.ini
export INCLUDEFILE=blank_include.ini
export VALUESINCLUDE=blank_include.ini
export PRIORSINCLUDE=blank_include.ini

export OMP_NUM_THREADS=2

cd ..
srun -n ${SLURM_NTASKS} cosmosis --mpi ppd/fiducial_3x2_goodnessoffit/ppd.ini -p listppd.size_chunk=1000 listppd.i_chunk=${SLURM_ARRAY_TASK_ID} output.filename=ppd/fiducial_3x2_goodnessoffit/ppd_chain_${RUNNAME}_fiducial_3x2_goodnessoffit_${SLURM_ARRAY_TASK_ID}.txt
cd ppd
