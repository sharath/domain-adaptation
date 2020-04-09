#!/usr/bin/env bash
#
#SBATCH --partition=1080ti-long
#SBATCH --gres=gpu:1
#SBATCH --mem=8192
#SBATCH --account=rkozma
#SBATCH --output=output/aae_%j.out

seed=$1
dec=$2

python3 -m AAE --seed $seed --dec_weight $dec