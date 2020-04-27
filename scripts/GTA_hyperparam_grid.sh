#!/usr/bin/env bash
mkdir output

alphas=(0.04 0.07 0.1 0.13 0.16)
betas=(0.01 0.02 0.03 0.04 0.05)
seeds=(6 7 8)

for a in ${alphas[@]}
do
  for b in ${betas[@]}
  do
    for s in ${seeds[@]}
    do
      sbatch scripts/launch_GTA_tune_wgan.sh $s $a $b
    done
  done
done
