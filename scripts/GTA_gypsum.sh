#!/bin/sh

alpha=(0.05 0.1 0.15 0.20 0.25 0.30 0.35 0.40 0.45 0.50)
adv_weight=(0.05 0.1 0.15 0.20 0.25 0.30 0.35 0.40 0.45 0.50)
seed=1

for i in {1..20}
do
  for a in ${alpha[@]}
  do
    for b in ${adv_weight[@]}
    do
      sbatch --output=/dev/null --error=/dev/null -p 2080ti-long --gres=gpu:1 scripts/launch_GTA.sh $seed $a $b
      sleep 1
    done
  done
done
