#!/usr/bin/env bash
#1 gpu id , #2 dataset, #3 model name, #4 output folder name

echo "[GPU： " $1 "] Testing on " $2 " dataset"

CUDA_VISIBLE_DEVICES=$1 python main.py --mode='test' --dataset=$2 --model=$3 --output_folder=$4

echo "Testing done."