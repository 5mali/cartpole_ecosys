#!/bin/bash

#SBATCH --job-name=BASE_G
#SBATCH --output=BASE_G.out
#SBATCH -p knm 
#SBATCH -N 1
#SBATCH -n 1
#SBATCH --cpus-per-task=10
#SBATCH --mem 20GB

srun parallel -j10  python ./BASE_G.py ::: 741 9086 4507 7551 5987 862 5955 5095 3119 3664 >> BASE_G.data 

