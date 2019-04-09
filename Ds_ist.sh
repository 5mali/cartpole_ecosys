#!/bin/bash

#SBATCH --job-name=Ds
#SBATCH --output=Ds.out
#SBATCH -p big
#SBATCH -N 1
#SBATCH -n 1
#SBATCH --cpus-per-task=11
#SBATCH --mem 100GB

srun python ./Ds.py $@ >> ./Ds.data


