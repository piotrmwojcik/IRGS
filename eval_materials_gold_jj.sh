for SCENE in \
    jumpingjacks150_v3_tex_statictimestep75 \
    aaa
do
    for DATA_SUBDIR in \
        golden_bay_4k_32x16_rot330 \
        aaa
    do
        export DATA_SUBDIR
        echo "Processing SCENE: $SCENE with DATA_SUBDIR: $DATA_SUBDIR"

     CUDA_VISIBLE_DEVICES=0 python eval_material_syn4.py -m \
        outputs/s2_${DATA_SUBDIR}/irgs_$SCENE --albedo_rescale 2 --resolution 2

    done
done
