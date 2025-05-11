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

        CUDA_VISIBLE_DEVICES=0 python render.py \
            -m outputs/s2_${DATA_SUBDIR}/irgs_$SCENE --eval --diffuse_sample_num 1024
    done
done
