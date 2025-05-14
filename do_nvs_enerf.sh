SCENE="images_710_780_single_ts"

CUDA_VISIBLE_DEVICES=0 python render.py \
  -m "outputs/irgs_${SCENE}_masks" --eval --diffuse_sample_num 1024
