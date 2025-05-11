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

        CUDA_VISIBLE_DEVICES=0 python train.py \
            -s data_specular/$SCENE/ \
            --iterations 20000 \
            --start_checkpoint_refgs outputs_specular/$SCENE/$DATA_SUBDIR/chkpnt50000.pth \
            --envmap_resolution 128 \
            --lambda_base_color_smooth 2 \
            --lambda_roughness_smooth 2 \
            --diffuse_sample_num 256 \
            --envmap_cubemap_lr 0.01 \
            --lambda_light_smooth 0.0005 \
            --init_roughness_value 0.6 \
            --lambda_light 0.1 \
            -m outputs_specular/s2_${DATA_SUBDIR}/irgs_$SCENE \
            --train_ray \
            --resolution 2
    done
done
