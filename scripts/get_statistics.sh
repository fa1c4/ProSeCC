#!/bin/bash
#SBATCH -N 1-1
#SBATCH -n 1
#SBATCH -p cpu5
#SBATCH -c 8 # 申请 CPU 核心
#SBATCH --mem 20G # 申请内存
#SBATCH -J djy_statistics_ss
#SBATCH -o log/job-%j_ss.out

echo "job begin"
date
srun /public/dingjinyang/anaconda3/envs/nlp/bin/python -u src/example.py
echo "job end"
date
