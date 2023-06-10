#!/bin/bash
#SBATCH -N 1-1
#SBATCH -n 1
#SBATCH -p cpu5
#SBATCH -c 16 # 申请 CPU 核心
#SBATCH --mem 32G # 申请内存
#SBATCH -J djy_statistics_0429_meteor
#SBATCH -o log/job-%j_all_0429_meteor.out

echo "job begin"
date
srun /public/dingjinyang/anaconda3/envs/nlp/bin/python -u src/example.py
# srun /public/dingjinyang/anaconda3/envs/nlp/bin/python -u src/example_adg.py
# srun /public/dingjinyang/anaconda3/envs/nlp/bin/python -u src/example_arithmetic.py
srun /public/dingjinyang/anaconda3/envs/nlp/bin/python -u src/example_meteor.py
# srun /public/dingjinyang/anaconda3/envs/nlp/bin/python -u src/example_sample.py
srun /public/dingjinyang/anaconda3/envs/nlp/bin/python -u src/example_dc.py

echo "job end"
date
