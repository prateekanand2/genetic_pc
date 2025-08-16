#!/bin/sh
#$ -cwd
#$ -l h_data=4G,h_rt=16:00:00
#$ -o logs/
#$ -j y
#$ -N bootstrap
#$ -pe shared 8
#$ -t 1-4

# Define your methods
METHODS=("10K_real_8020" "10K_hclt_8020" "10K_rbm_8020" "10K_wgan_8020")

# Select the method based on SGE_TASK_ID
INDEX=$(($SGE_TASK_ID - 1))
METHOD=${METHODS[$INDEX]}

. /u/local/Modules/default/init/modules.sh
. /u/home/p/panand2/.bashrc
ulimit -c 0
module load anaconda3
conda activate pyjuice_env
python --version

threads=8

export MKL_NUM_THREADS=${threads}
export NUMEXPR_NUM_THREADS=${threads}
export OMP_NUM_THREADS=${threads}
export PYTHONDONTWRITEBYTECODE=1

python bootstrap_info.py --method $METHOD