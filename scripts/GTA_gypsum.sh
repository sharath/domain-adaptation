#!/bin/sh

for i in {1..20}
do
  sbatch --output=/dev/null --error=/dev/null -p 1080ti-long --gres=gpu:1 scripts/launch_GTA.sh
  sleep 1
done