#!/usr/bin/env bash
#
#SBATCH --partition=1080ti-long
#SBATCH --gres=gpu:1
#SBATCH --mem=8192
#SBATCH --account=rkozma
#SBATCH --output=output/ann_%j.log	

seed=$1
alpha=$2
beta=$3

echo python3 -m GTA --seed $seed --alpha $alpha --beta $beta
python3 -m GTA --seed $seed --alpha $alpha --beta $beta
