#!/bin/sh
seed=$1
alpha1=$2
beta1=$3

alpha2=`echo "print(${alpha1} + 1/20)" | python3`
beta2=`echo "print(${beta1} + 1/20)" | python3`

python3 -m GTA --seed $1 --alpha $alpha1 --beta $beta1
python3 -m GTA --seed $1 --alpha $alpha1 --beta $beta2
python3 -m GTA --seed $1 --alpha $alpha2 --beta $beta1
python3 -m GTA --seed $1 --alpha $alpha2 --beta $beta2