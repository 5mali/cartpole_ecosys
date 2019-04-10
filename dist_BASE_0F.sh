#!/bin/bash

#SBATCH --job-name=dist_BASE_0F
#SBATCH --output=dist_BASE_0F.out
#SBATCH -p knm
#SBATCH -N 1
#SBATCH -n 1
#SBATCH --cpus-per-task=11
#SBATCH --mem 1GB

srun python ./dist_BASE_0F.py $@ >> ./dist_BASE_0F.data


