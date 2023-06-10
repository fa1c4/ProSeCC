#!/bin/bash
#SBATCH -p gpu3
#SBATCH -N 1-1
#SBATCH -n 1
#SBATCH -c 2 # 申请 CPU 核心
#SBATCH --gres gpu:1 # 分配一个GPU
#SBATCH --mem 8G # 申请内存
#SBATCH -o log/job-%j_gen_data.out

echo "job begin"
date
srun /public/dingjinyang/anaconda3/envs/nlp/bin/python -u src/analysis/generate_dataset_meteor.py --index=$1
echo "job end"
date

