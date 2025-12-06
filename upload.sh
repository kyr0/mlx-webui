#!/bin/bash

hf upload \
    kyr0/aidana-slm-mlx \
    ./quantized_model \
    --commit-message "Initial commit"