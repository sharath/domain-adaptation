#!/bin/sh

for i in {1..20}
do
  sbatch -p 1080ti-long --gres=gpu:1 scripts/launch_GTA.sh
  sleep 1
done