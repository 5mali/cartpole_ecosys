#!/bin/bash

#SBATCH --job-name=Deq0
#SBATCH --output=Deq0.out
#SBATCH -p p 
#SBATCH -N 1
#SBATCH -n 1
#SBATCH --cpus-per-task=11
#SBATCH --mem 20GB

srun python ./Deq0.py $@ >> ./Deq0.data


