#!/bin/bash

#SBATCH --job-name=D
#SBATCH --output=D.out
#SBATCH -p knm 
#SBATCH -N 1
#SBATCH -n 1
#SBATCH --cpus-per-task=11
#SBATCH --mem 150GB

srun python ./D.py $@ >> ./D.data


