#!/bin/bash
#SBATCH -p gpu4
#SBATCH -N 1-1
#SBATCH -n 1
#SBATCH -c 2
#SBATCH --mem 4G
#SBATCH --gres gpu:1
#SBATCH -o log/job-%j.out

echo "job begin"
date
srun singularity exec --nv /data-x/g12/chenkj/gpu3090.sif /public/dingjinyang/anaconda3/envs/pt_preview/bin/python -u example.py
echo "job end"
date