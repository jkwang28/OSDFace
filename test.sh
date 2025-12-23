export HF_ENDPOINT="https://hf-mirror.com"
ckpt_path="pretrained"
python infer.py \
    --input_image data/WebPhoto-Test \
    --output_dir results/WebPhoto-Test \
    --pretrained_model_name_or_path "stabilityai/stable-diffusion-2-1-base" \
    --img_encoder_weight "$ckpt_path/associate_2.ckpt" \
    --gpu_id 0 \
    --ckpt_path $ckpt_path \
    --merge_lora