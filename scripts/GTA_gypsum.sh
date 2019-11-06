#!/bin/sh

sleep 2
sbatch -p 1080ti-long --gres=gpu:1 scripts/launch_GTA_basline.sh
sleep 2
sbatch -p 1080ti-long --gres=gpu:1 scripts/launch_GTA_basline.sh
sleep 2
sbatch -p 1080ti-long --gres=gpu:1 scripts/launch_GTA_basline.sh
sleep 2
sbatch -p 1080ti-long --gres=gpu:1 scripts/launch_GTA_basline.sh
sleep 2
sbatch -p 1080ti-long --gres=gpu:1 scripts/launch_GTA_basline.sh

sleep 2
sbatch -p 1080ti-long --gres=gpu:1 scripts/launch_GTA_adapt.sh
sleep 2
sbatch -p 1080ti-long --gres=gpu:1 scripts/launch_GTA_adapt.sh
sleep 2
sbatch -p 1080ti-long --gres=gpu:1 scripts/launch_GTA_adapt.sh
sleep 2
sbatch -p 1080ti-long --gres=gpu:1 scripts/launch_GTA_adapt.sh
sleep 2
sbatch -p 1080ti-long --gres=gpu:1 scripts/launch_GTA_adapt.sh
