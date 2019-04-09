#!/bin/bash

#SBATCH --job-name=D1
#SBATCH --output=D1.out
#SBATCH -p big 
#SBATCH -N 1
#SBATCH -n 1
#SBATCH --cpus-per-task=11
#SBATCH --mem 150GB

srun python ./D1.py $@ >> ./D1.data


