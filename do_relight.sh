for SCENE in \
    hook150_v3_transl_statictimestep1 \
    jumpingjacks150_v3_tex_statictimestep75 \
    mouse150_v2_transl_statictimestep1 \
    spheres_cube_dataset_v5_statictimestep1 \
    standup150_v3_statictimestep75
do

    pairs=(
      "chapel_day_4k_32x16_rot0 golden_bay_4k_32x16_rot330"
      "golden_bay_4k_32x16_rot330 dam_wall_4k_32x16_rot90"
    )

    for pair in "${pairs[@]}"; do
      read DATA_SUBDIR MAP_PATH <<< "$pair"
      echo "Running with DATA_SUBDIR=$DATA_SUBDIR and SCENE=$SCENE"

       #export DATA_SUBDIR
      echo "Processing SCENE: $SCENE with DATA_SUBDIR: $DATA_SUBDIR and with $MAP_PATH"

      #CUDA_VISIBLE_DEVICES=0 CUDA_VISIBLE_DEVICES=0 python eval_material_syn4.py -m \
      #  outputs/s2_${DATA_SUBDIR}/irgs_$SCENE --albedo_rescale 2 --resolution 2

    done
done
