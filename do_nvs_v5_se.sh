for SCENE in \
    hook150_v5_spec32_statictimestep1 \
    jumpingjacks150_v5_spec32_statictimestep75 \
    mouse150_v5_spec32_statictimestep1 \
    spheres_v5_spec32_statictimestep1 \
    standup150_v5_spec32_statictimestep75
do
    for DATA_SUBDIR in \
      chapel_day_4k_32x16_rot0 \
      dam_wall_4k_32x16_rot90 \
      golden_bay_4k_32x16_rot330
    do
        export DATA_SUBDIR
        echo "Processing SCENE: $SCENE with DATA_SUBDIR: $DATA_SUBDIR"

        CUDA_VISIBLE_DEVICES=0 python render.py \
            -m outputs_specular_nem/s2_${DATA_SUBDIR}/irgs_$SCENE --eval --diffuse_sample_num 1024
    done
done
