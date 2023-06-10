#!/bin/bash

for i in {1..5}
do
    sbatch scripts/gen_data_meteor.sh $i --job-name=djy_gen_data_$i
done