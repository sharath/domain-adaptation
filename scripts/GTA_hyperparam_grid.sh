#!/bin/sh

alphas=(0.0 0.05 0.1 0.15 0.2)
betas=(0.0 0.01 0.02 0.03 0.04 0.05)
seed=1

for a in ${alphas[@]}
do
  for b in ${betas[@]}
  do
    sbatch --output=/dev/null --error=/dev/null -p 1080ti-short --gres=gpu:1 --mem=6144 scripts/launch_GTA_single.sh $seed $a $b
    sleep 2
  done
done
