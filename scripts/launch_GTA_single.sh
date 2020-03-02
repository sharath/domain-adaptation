#!/bin/sh
seed=$1
alpha=$2
beta=$3

python3 -m GTA --seed $1 --alpha $alpha --beta $beta
