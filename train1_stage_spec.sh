for SCENE in \
    hook150_v4_spec_statictimestep1 \
    jumpingjacks150_v34_spec_statictimestep75 \
    mouse150_v3_spec_statictimestep1 \
    spheres_cube_dataset_v6_spec_statictimestep1 \
    standup150_v4_spec_statictimestep75
do
    for DATA_SUBDIR in \
      chapel_day_4k_1024x512_rot0 \
      dam_wall_4k_1024x512_rot90 \
      golden_bay_4k_1024x512_rot330
    do
        export DATA_SUBDIR
        echo "Processing SCENE: $SCENE with DATA_SUBDIR: $DATA_SUBDIR"

        CUDA_VISIBLE_DEVICES=0 python train_refgaussian.py \
            -s data_specular/$SCENE \
            -m outputs_specular/$SCENE/$DATA_SUBDIR \
            --eval -w --lambda_mask_entropy 0.05 --resolution 2
    done
done
