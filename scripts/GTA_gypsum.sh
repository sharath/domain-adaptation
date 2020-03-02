#!/bin/sh

vals=(0.05 0.15 0.25 0.35 0.45)
seed=1

for a in ${vals[@]}
do
  for b in ${vals[@]}
  do
    sbatch --output=/dev/null --error=/dev/null -p 2080ti-long --gres=gpu:1 --mem=6144 scripts/launch_GTA.sh $seed $a $b
    sleep 2
  done
done
