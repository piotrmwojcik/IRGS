SCENE="images_710_780_single_ts"

for SCENE in \
    hook150_v3_transl_statictimestep1 \
    jumpingjacks150_v3_tex_statictimestep75 \
    mouse150_v2_transl_statictimestep1 \
    spheres_cube_dataset_v5_statictimestep1 \
    standup150_v3_statictimestep75
do
    CUDA_VISIBLE_DEVICES=0 python render.py \
      -m "outputs/irgs_${SCENE}_masks" --eval --diffuse_sample_num 1024
done
