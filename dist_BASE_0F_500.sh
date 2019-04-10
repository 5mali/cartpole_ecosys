#!/bin/bash

#SBATCH --job-name=dist_BASE_0F_500
#SBATCH --output=dist_BASE_0F_500.out
#SBATCH -p p
#SBATCH -N 1
#SBATCH -n 1
#SBATCH --cpus-per-task=11
#SBATCH --mem 1GB

srun python ./dist_BASE_0F_500.py $@ >> ./dist_BASE_0F_500.data


