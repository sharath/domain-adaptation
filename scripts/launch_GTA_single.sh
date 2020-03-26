#!/usr/bin/env bash
#
#SBATCH --partition=1080ti-short
#SBATCH --gres=gpu:1
#SBATCH --mem=8192
#SBATCH --account=rkozma
#SBATCH --output=output/ann_%j.out

seed=$1
alpha=$2
beta=$3

python3 -m GTA --seed $seed --alpha $alpha --beta $beta
