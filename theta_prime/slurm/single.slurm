#!/bin/bash -l

#SBATCH --account sscheid1_deep_replication
#SBATCH --mail-type ALL
#SBATCH --mail-user antoine.didisheim@unil.ch

#SBATCH --job-name dr_default
#SBATCH --output=/scratch/adidishe/theta_project/theta_prime/single.out
#SBATCH --error=/scratch/adidishe/theta_project/theta_prime/single.err
#SBATCH --chdir=/scratch/adidishe/theta_project/theta_prime

#SBATCH --partition cpu


#SBATCH --nodes 1
#SBATCH --ntasks 1
#SBATCH --cpus-per-task 12
#SBATCH --mem 512G
#SBATCH --time 2:00:00


module load gcc/9.3.0 python/3.8.8

source /scratch/adidishe/theta_project/theta_prime/venv/bin/activate
python3 /scratch/adidishe/theta_project/theta_prime/data.py
