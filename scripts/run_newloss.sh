#!/bin/bash

echo "start training with nohup"
shopt -s nullglob
numfiles=(./nohup_outputs/*)
numfiles=${#numfiles[@]}
numfiles=$((numfiles+2))
echo "create log > nohup_newloss_"$numfiles".out"

nohup \
 python -W ignore crack.py \
 train \
 --dataset=./../../dataset/new-crack500/ \
 --weights=coco \
 --logs=./../../logs/crack_logs_newloss \
&> nohup_outputs/nohup_newloss_$numfiles.out \
&
