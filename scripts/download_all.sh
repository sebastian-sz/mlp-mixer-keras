#!/bin/bash

# Script to download all (released) JAX checkpoints, regarding MLP-Mixer models.

curl -o weights/original_weights/b16-imagenet1k.npz https://storage.googleapis.com/mixer_models/imagenet1k/Mixer-B_16.npz
curl -o weights/original_weights/l16-imagenet1k.npz https://storage.googleapis.com/mixer_models/imagenet1k/Mixer-L_16.npz
curl -o weights/original_weights/b16-sam.npz  https://storage.googleapis.com/mixer_models/sam/Mixer-B_16.npz
curl -o weights/original_weights/b32-sam.npz https://storage.googleapis.com/mixer_models/sam/Mixer-B_32.npz
