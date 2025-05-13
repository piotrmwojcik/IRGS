SCENE=images_710_780_single_ts

CUDA_VISIBLE_DEVICES=0 python train_refgaussian.py \
  -s data/$SCENE \
  -m outputs/$SCENE \
  --eval -w --lambda_mask_entropy 0.05 --resolution 2
