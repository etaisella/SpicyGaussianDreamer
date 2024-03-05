#!/bin/bash
echo "Starting Run!"

# Reading arguments:
gpu_num=0
while getopts g:d: flag
do
    case "${flag}" in
        g) gpu_num=${OPTARG};;
    esac
done

# Setting GPU:
echo "Running on GPU: $gpu_num";
export CUDA_VISIBLE_DEVICES=$gpu_num

# Rendering function template:
train_default() {
	# Train:
	python launch.py \
	--config="$2" \
	--train \
	--random \
	--gpu=0 \
	system.prompt_processor.prompt="$1"
}

# STARTING RUN:

prompt="a red velvet chair"
config="configs/gaussiandreamer-sd-chair-deabs-2.yaml"

train_default "$prompt" "$config"

prompt="a red velvet chair"
config="configs/gaussiandreamer-sd-chair-deabs-2.yaml"

train_default "$prompt" "$config"

prompt="a red velvet chair"
config="configs/gaussiandreamer-sd-chair-deabs-2.yaml"

train_default "$prompt" "$config"

prompt="a red velvet chair"
config="configs/gaussiandreamer-sd-chair-deabs-2.yaml"

train_default "$prompt" "$config"