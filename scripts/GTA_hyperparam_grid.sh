#!/usr/bin/env bash
mkdir output

alphas=(0.0 0.05 0.1 0.15 0.2)
betas=(0.0 0.01 0.02 0.03 0.04 0.05)
seeds=(0 1 2 3 4)

for a in ${alphas[@]}
do
  for b in ${betas[@]}
  do
    for s in ${seeds[@]}
    do
      sbatch scripts/launch_GTA_single.sh $s $a $b
      sleep 2
    done
  done
done
