#!/bin/bash

#SBATCH --job-name=Dbase
#SBATCH --output=Dbase.out
#SBATCH -p knm
#SBATCH -N 1
#SBATCH -n 1
#SBATCH --cpus-per-task=11
#SBATCH --mem 20GB

srun python ./D_base.py $@ >> ./D_base.data


