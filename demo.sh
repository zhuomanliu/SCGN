#!/usr/bin/env bash
#1 gpu id , #2 dataset, #3 model name, #4 path of inputs #5 input name (L) #6 input name (R) #7 output name

echo "[GPU： " $1 "] Inference on " $2 " dataset"

CUDA_VISIBLE_DEVICES=$1 python main.py --mode='demo' --dataset=$2 --model=$3 --data_path=$4 \
--input_l_name=$5 --input_r_name=$6 --output_name=$7

echo "Inference done."