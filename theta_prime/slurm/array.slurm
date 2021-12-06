#!/bin/bash -l

#SBATCH --account sscheid1_deep_replication
#SBATCH --mail-type ALL
#SBATCH --mail-user antoine.didisheim@unil.ch
#SBATCH --array=0-3

#SBATCH --chdir /scratch/adidishe/deep_recovery
#SBATCH --job-name dr_default
#SBATCH --output=/scratch/adidishe/theta_project/theta_prime/out/my_code_%a.out
#SBATCH --error=/scratch/adidishe/theta_project/theta_prime/out/my_code_%a.err
#SBATCH --chdir=/scratch/adidishe/theta_project/theta_prime

#SBATCH --partition cpu


#SBATCH --nodes 1
#SBATCH --ntasks 1
#SBATCH --cpus-per-task 12
#SBATCH --mem 128G
#SBATCH --time 01:00:00


module load gcc/9.3.0 python/3.8.8

source /scratch/adidishe/theta_project/theta_prime/venv/bin/activate


python3 /scratch/adidishe/theta_project/theta_prime/create_client_types_1.py ${SLURM_ARRAY_TASK_ID}
python3 /scratch/adidishe/theta_project/theta_prime/create_client_types_2.py ${SLURM_ARRAY_TASK_ID}
