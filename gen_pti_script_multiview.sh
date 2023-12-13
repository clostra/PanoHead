#!/usr/bin/env bash

models=("easy-khair-180-gpc0.8-trans10-025000.pkl")
# models=("easy-khair-180-gpc0.8-trans10-025000.pkl"\
  # "ablation-trigridD-1-025000.pkl")

in="models"
out="pti_out"
num_steps="1000"

for model in ${models[@]}

do

  # perform the pti and save w
  # python projector_withseg.py --outdir=${out} --target_img=dataset/testdata_img --network ${in}/${model} --num-steps-pti=${num_steps}
  # generate .mp4 after finetune
  python gen_videos_proj_withseg.py --output=${out}/${model}/PTI_render/post.mp4 --latent=${out}/${model}/projected_w.npz --trunc 0.7 --network ${out}/${model}/fintuned_generator.pkl --cfg Head --shapes
  # generate .mp4 before finetune
  # python gen_videos_proj_withseg.py --output=${out}/${model}/PTI_render/pre.mp4 --latent=${out}/${model}/projected_w.npz --trunc 0.7 --network ${in}/${model} --cfg Head --shapes

done