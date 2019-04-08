#!/bin/bash

#SBATCH --job-name=ECO_A
#SBATCH --output=ECO_A.out
#SBATCH -p p
#SBATCH -N 1
#SBATCH -n 20

srun python ./ECO_A.py $@ >> ECO_A.data

