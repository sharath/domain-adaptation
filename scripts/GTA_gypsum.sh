#!/bin/sh


sbatch -p 1080ti-long --gres=gpu:1 scripts/launch_GTA_basline.sh
sbatch -p 1080ti-long --gres=gpu:1 scripts/launch_GTA_basline.sh
sbatch -p 1080ti-long --gres=gpu:1 scripts/launch_GTA_basline.sh
sbatch -p 1080ti-long --gres=gpu:1 scripts/launch_GTA_basline.sh
sbatch -p 1080ti-long --gres=gpu:1 scripts/launch_GTA_basline.sh
sbatch -p 1080ti-long --gres=gpu:1 scripts/launch_GTA_basline.sh
sbatch -p 1080ti-long --gres=gpu:1 scripts/launch_GTA_basline.sh
sbatch -p 1080ti-long --gres=gpu:1 scripts/launch_GTA_basline.sh
sbatch -p 1080ti-long --gres=gpu:1 scripts/launch_GTA_basline.sh
sbatch -p 1080ti-long --gres=gpu:1 scripts/launch_GTA_basline.sh

sbatch -p 1080ti-long --gres=gpu:1 scripts/launch_GTA_adapt.sh
sbatch -p 1080ti-long --gres=gpu:1 scripts/launch_GTA_adapt.sh
sbatch -p 1080ti-long --gres=gpu:1 scripts/launch_GTA_adapt.sh
sbatch -p 1080ti-long --gres=gpu:1 scripts/launch_GTA_adapt.sh
sbatch -p 1080ti-long --gres=gpu:1 scripts/launch_GTA_adapt.sh
sbatch -p 1080ti-long --gres=gpu:1 scripts/launch_GTA_adapt.sh
sbatch -p 1080ti-long --gres=gpu:1 scripts/launch_GTA_adapt.sh
sbatch -p 1080ti-long --gres=gpu:1 scripts/launch_GTA_adapt.sh
sbatch -p 1080ti-long --gres=gpu:1 scripts/launch_GTA_adapt.sh
sbatch -p 1080ti-long --gres=gpu:1 scripts/launch_GTA_adapt.sh


