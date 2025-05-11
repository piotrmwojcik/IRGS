for SCENE in \
    spheres_cube_dataset_v8_spec32_statictimestep1 \
    aaa
do
    for DATA_SUBDIR in \
      chapel_day_4k_32x16_rot0 \
      dam_wall_4k_32x16_rot90 \
      golden_bay_4k_32x16_rot330 \
      small_harbour_sunset_4k_32x16_rot270
    do
        export DATA_SUBDIR
        echo "Processing SCENE: $SCENE with DATA_SUBDIR: $DATA_SUBDIR"

        CUDA_VISIBLE_DEVICES=0 python train_refgaussian.py \
            -s data_specular/$SCENE \
            -m outputs_specular/$SCENE/$DATA_SUBDIR \
            --eval -w --lambda_mask_entropy 0.05 --resolution 2
    done
done
