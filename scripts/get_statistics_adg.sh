#!/bin/bash
#SBATCH -p gpu4
#SBATCH -N 1-1
#SBATCH -n 1
#SBATCH -c 6 # 申请 CPU 核心
#SBATCH -J djy_adg
#SBATCH --gres gpu:1 # 分配一个GPU
#SBATCH --mem 16G # 申请内存
#SBATCH -o log/job-%j.out

echo "job begin"
date
srun singularity exec --nv /data-x/g12/chenkj/gpu3090.sif /public/dingjinyang/anaconda3/envs/pt_preview/bin/python -u src/example_adg.py
echo "job end"
date
