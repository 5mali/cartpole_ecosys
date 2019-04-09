#!/bin/bash

#SBATCH --job-name=D32
#SBATCH --output=D32.out
#SBATCH -p big 
#SBATCH -N 1
#SBATCH -n 1
#SBATCH --cpus-per-task=11
#SBATCH --mem 800GB

srun python ./D_32.py $@ >> ./D_32.data


