#!/bin/bash

#SBATCH --job-name=Dlo4
#SBATCH --output=Dlo4.out
#SBATCH -p big
#SBATCH -N 1
#SBATCH -n 1
#SBATCH --cpus-per-task=11
#SBATCH --mem 70GB

srun python ./Dlo4.py $@ >> ./Dlo4.data


