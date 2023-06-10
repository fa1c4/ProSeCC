#!/bin/bash
#SBATCH --job-name djy_ckp
#SBATCH -p gpu3
#SBATCH -N 1-1
#SBATCH -n 1
#SBATCH -c 8 # 申请 CPU 核心
#SBATCH --gres gpu:10 # 分配一个GPU
#SBATCH --mem 20G # 申请内存
#SBATCH -o log/job-%j.out

echo "job begin"
date
srun /public/dingjinyang/anaconda3/envs/nlp/bin/python -u src/analysis/test_checkpoints.py
echo "job end"
date