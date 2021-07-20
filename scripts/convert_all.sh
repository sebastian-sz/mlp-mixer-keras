#!/bin/bash

# Script to convert all downloaded weights to .h5 format.

### B16 Imagenet 1k ###
python scripts/mlp_mixer_weight_update_util.py \
  -i weights/original_weights/b16-imagenet1k.npz \
  -m b16 \
  -o weights/mlp-mixer-b16.h5

### B16 SAM ###
python scripts/mlp_mixer_weight_update_util.py \
  -i weights/original_weights/b16-sam.npz \
  -m b16 \
  -o weights/mlp-mixer-b16-sam.h5

### B32 SAM ###
python scripts/mlp_mixer_weight_update_util.py \
  -i weights/original_weights/b32-sam.npz \
  -m b32 \
  -o weights/mlp-mixer-b32-sam.h5

### L16 Imagenet 1k ###
python scripts/mlp_mixer_weight_update_util.py \
  -i weights/original_weights/l16-imagenet1k.npz \
  -m l16 \
  -o weights/mlp-mixer-l16.h5
