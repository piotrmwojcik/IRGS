for SCENE in \
    spheres_cube_dataset_v8_spec32_statictimestep1 \
    aa
    #standup150_v3_statictimestep75
do

    pairs=(
      "chapel_day_4k_32x16_rot0 golden_bay_4k_32x16_rot330"
      #"dam_wall_4k_32x16_rot90 small_harbour_sunset_4k_32x16_rot270"
      #"golden_bay_4k_32x16_rot330 dam_wall_4k_32x16_rot90"
      "golden_bay_4k_32x16_rot330 chapel_day_4k_32x16_rot0"
    )

    for pair in "${pairs[@]}"; do
      read DATA_SUBDIR MAP_NAME <<< "$pair"
      MAP_PATH="/home/pwojcik/IRGS/data_specular/$SCENE/$MAP_NAME.hdr"
      echo "Running with DATA_SUBDIR=$DATA_SUBDIR and SCENE=$SCENE"
      export MAP_PATH
      export MAP_NAME
      export SCENE
      export DATA_SUBDIR
       #export DATA_SUBDIR
      echo "Processing SCENE: $SCENE with DATA_SUBDIR: $DATA_SUBDIR and with $MAP_PATH"

      CUDA_VISIBLE_DEVICES=0 python eval_relighting_syn4.py -m outputs_specular/s2_${DATA_SUBDIR}/irgs_$SCENE \
        --diffuse_sample_num 1024 --light_sample_num 0 --albedo_rescale 2 -e light

    done
done
