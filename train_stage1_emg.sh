for SCENE in \
    hook150_v3_transl_statictimestep1 \
    jumpingjacks150_v3_tex_statictimestep75 \
    mouse150_v2_transl_statictimestep1 \
    spheres_cube_dataset_v5_statictimestep1 \
    standup150_v3_statictimestep75
do
    for DATA_SUBDIR in \
        small_harbour_sunset_4k_32x16_rot270 \
        aaa
    do
        export DATA_SUBDIR
        echo "Processing SCENE: $SCENE with DATA_SUBDIR: $DATA_SUBDIR"

        CUDA_VISIBLE_DEVICES=0 python train_refgaussian.py \
            -s data/$SCENE \
            -m outputs/$SCENE/$DATA_SUBDIR \
            --eval -w --lambda_mask_entropy 0.05 --resolution 2
    done
done
