#!/usr/bin/env bash
set -e
set -x

# Use specified checkpoint path, otherwise, default value
finetune_ckpt=${1:-"output/train_weather_finetune_adapter_only/checkpoint/latest"}
output_child_dir=${2:-"adapter_only"}

# python script/depth/infer.py \
#     --base_checkpoint checkpoint/marigold-depth-v1-1 \
#     --seed 2026 \
#     --base_data_dir data/cadcd \
#     --denoise_steps 2 \
#     --ensemble_size 1 \
#     --processing_res 0 \
#     --dataset_config config/dataset_depth/dataset_cadcd.yaml \
#     --output_dir output/eval/cadcd_test/${output_child_dir} \
#     --version restore \
#     --finetune_checkpoint ${finetune_ckpt} \


python script/depth/eval.py \
    --base_data_dir data/cadcd \
    --dataset_config config/dataset_depth/dataset_cadcd.yaml \
    --alignment least_square \
    --prediction_dir output/eval/cadcd_test/${output_child_dir} \
    --output_dir output/eval/cadcd_test/eval_metric
