#!/bin/bash

SCENE="images_710_780_single_ts"

maps=(
  "chapel_day_4k_32x16_rot0"
  "golden_bay_4k_32x16_rot330"
  "test_env_j6_i24"
)

for MAP_NAME in "${maps[@]}"; do
  MAP_PATH="/home/pwojcik/IRGS/data/${SCENE}/${MAP_NAME}.hdr"

  echo "Running with MAP_PATH=${MAP_PATH} and SCENE=${SCENE}"
  export MAP_PATH
  export MAP_NAME
  export SCENE

  CUDA_VISIBLE_DEVICES=0 python eval_relighting_enerf.py -m "outputs/irgs_${SCENE}" \
    --diffuse_sample_num 1024 \
    --light_sample_num 256 \
    --resolution 4 \
    --albedo_rescale 0 \
    -e light

done
