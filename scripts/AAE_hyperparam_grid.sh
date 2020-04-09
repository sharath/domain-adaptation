#!/usr/bin/env bash
mkdir output

decs=(0.05 0.1 0.15 0.2 0.25 0.3 0.35 0.4 0.45 0.5 0.55 0.6 0.65 0.7 0.75 0.8 0.85 0.9 0.95)
seeds=(0 1 2 3 4)

for d in ${decs[@]}
do
  for s in ${seeds[@]}
  do
    sbatch scripts/launch_AAE.sh $s $d
  done
done
