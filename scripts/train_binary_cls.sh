#!/bin/bash
#SBATCH --job-name djy_train_binary_cls
#SBATCH -p gpu3
#SBATCH -N 1-1
#SBATCH -n 1
#SBATCH -c 10 # 申请 CPU 核心
#SBATCH --gres gpu:2 # 分配 GPU 数量
#SBATCH --mem 20G # 申请内存
#SBATCH -o log/job-%j.out

echo "job begin"
date
srun /public/dingjinyang/anaconda3/envs/nlp/bin/python -u src/analysis/train_binary_cls.py
echo "job end"
date

