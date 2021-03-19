#!/usr/bin/env bash
#1 gpu id, #2 dataset, #3 model save folder name

echo "[GPU：" $1 "] Training on" $2 "dataset"

if [ $2 == "multipie" ];
then
    CUDA_VISIBLE_DEVICES=$1 python main.py --mode='train' --dataset=$2 --save_folder=$3 --epochs=200 --epoch_save=10
elif [ $2 == "kitti" ];
then
    CUDA_VISIBLE_DEVICES=$1 python main.py --mode='train' --dataset=$2 --save_folder=$3 --epochs=10 --epoch_save=9
else
    echo "Invalid dataset!"
fi

echo "Training done."