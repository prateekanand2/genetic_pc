#!/bin/sh
#$ -cwd
#$ -l h_data=8G,h_rt=3:00:00
#$ -o /dev/null
#$ -j y
#$ -N impute_8020

. /u/local/Modules/default/init/modules.sh
. /u/home/p/panand2/.bashrc
ulimit -c 0
module load anaconda3
conda activate pyjuice_env
python --version

threads=1
log=impute_${METHOD_FULL}
train=results/1KG/8020/${METHOD}/${METHOD_FULL}_samples
test=../results/1KG/8020/data/8020_test
chrnum=15

export MKL_NUM_THREADS=${threads}
export NUMEXPR_NUM_THREADS=${threads}
export OMP_NUM_THREADS=${threads}
export PYTHONDONTWRITEBYTECODE=1

output_dir="out/${log}/"
mkdir -p "${output_dir}"

for i in $(seq 0 99); do
    my_task_id=$((SGE_TASK_ID + i))
    logfile="${output_dir}/${log}.o${JOB_ID}.${my_task_id}"

    {
        echo "Starting task ${my_task_id}"
        echo "Using mapfile: ${MAPFILE}"

        if [ -f "${MAPFILE}" ]; then
            index=$(sed -n "${my_task_id}p" "${MAPFILE}")
        else
            echo "Mapfile not found at ${MAPFILE}"
            index=${my_task_id}
        fi

        if [ -z "$index" ]; then
            echo "Index not found or empty at line ${my_task_id}, skipping."
            exit 0
        fi

        echo "Running with index: ${index}"
        which python3
        python3 --version
        python3 impute5.py ${index} ${threads} ${train} ${test} ${chrnum} ${METHOD_FULL}
        echo "Task ${my_task_id} finished with exit code $?"

    } > "${logfile}" 2>&1
done

wait