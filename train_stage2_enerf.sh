SCENE=images_710_780_single_ts

CUDA_VISIBLE_DEVICES=0 python train.py \
    -s "data/$SCENE" \
    --iterations 20000 \
    --start_checkpoint_refgs "outputs/${SCENE}/chkpnt50000.pth" \
    --envmap_resolution 128 \
    --lambda_base_color_smooth 2 \
    --lambda_roughness_smooth 2 \
    --diffuse_sample_num 256 \
    --envmap_cubemap_lr 0.01 \
    --lambda_light_smooth 0.0005 \
    --init_roughness_value 0.6 \
    --lambda_light 0.1 \
    -m "outputs/irgs_${SCENE}_new" \
    --train_ray \
    --resolution 2

