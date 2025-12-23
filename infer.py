import copy
import os
import sys
sys.path.append(os.getcwd())

import glob
import argparse
import copy
import torch
import torch.multiprocessing as mp
import torch.nn as nn
from torchvision import transforms
import torchvision.transforms.functional as F
import torch.nn.functional as Fun
import numpy as np
from PIL import Image
from tqdm import tqdm
import random
from safetensors import safe_open
from diffusers import DDIMScheduler, AutoencoderKL, UNet2DConditionModel, StableDiffusionPipeline

from utils.vaehook import perfcount
from utils.others import get_x0_from_noise
from lq_embed import vqvae_encoder, TwoLayerConv1x1

tensor_transforms = transforms.Compose([
    transforms.ToTensor(),
])

class OSDFace_test(nn.Module):
    def __init__(self, args, gpu_id, Unet):
        super().__init__()

        self.args = args
        self.device = torch.device(f"cuda:{gpu_id}")

        self.noise_scheduler = DDIMScheduler.from_pretrained(args.pretrained_model_name_or_path, subfolder="scheduler")
        self.alphas_cumprod = self.noise_scheduler.alphas_cumprod.to(self.device)
        self.vae = AutoencoderKL.from_pretrained(self.args.pretrained_model_name_or_path, subfolder="vae")
        if args.merge_lora:
            self.unet = copy.deepcopy(Unet)
        else:
            self.unet = UNet2DConditionModel.from_pretrained(self.args.pretrained_model_name_or_path, subfolder="unet")

        self.weight_dtype = torch.float32
        if args.mixed_precision == "fp16":
            self.weight_dtype = torch.float16

        self.load_ckpt(args.ckpt_path)

        self.img_encoder = vqvae_encoder(args).to(self.device, dtype=self.weight_dtype)
        self.unet.to(self.device, dtype=self.weight_dtype)
        self.vae.to(self.device, dtype=self.weight_dtype)
        self.img_encoder.to(self.device, dtype=self.weight_dtype)

        self.timesteps = 399

    def load_ckpt(self, ckpt_path):
        if not self.args.cat_prompt_embedding:
            self.embedding_change = TwoLayerConv1x1(512, 1024)
            self.embedding_change.load_state_dict(
                torch.load(os.path.join(ckpt_path, "embedding_change_weights.pth"), weights_only=False)
            )
            self.embedding_change.to(self.device, dtype=self.weight_dtype)
        if not self.args.merge_lora:
            pipe = StableDiffusionPipeline(
                self.vae,
                None,
                None,
                self.unet,
                self.noise_scheduler,
                None, None
            )
            pipe.load_lora_weights(ckpt_path)
            self.unet = pipe.unet

    @perfcount
    @torch.no_grad()
    def forward(self, lq):
        stream1 = torch.cuda.Stream()
        stream2 = torch.cuda.Stream()

        with torch.cuda.stream(stream1):
            prompt_embeds = self.img_encoder(lq).reshape(lq.shape[0], 77, -1)
            if not self.args.cat_prompt_embedding:
                prompt_embeds = self.embedding_change(prompt_embeds)

        with torch.cuda.stream(stream2):
            lq_latent = self.vae.encode(
                lq.to(self.weight_dtype)
            ).latent_dist.sample() * self.vae.config.scaling_factor

        torch.cuda.synchronize()

        model_pred = self.unet(lq_latent, self.timesteps, encoder_hidden_states=prompt_embeds).sample

        x_0 = get_x0_from_noise(
            lq_latent.double(),
            model_pred.double(),
            self.alphas_cumprod.double(),
            self.timesteps
        ).float()

        output_image = (self.vae.decode(x_0.to(self.weight_dtype) / self.vae.config.scaling_factor).sample).clamp(
            -1, 1)
        output_image = output_image * 0.5 + 0.5

        return output_image.clamp(0.0, 1.0)

def merge_Unet(args):
    unet = UNet2DConditionModel.from_pretrained(args.pretrained_model_name_or_path, subfolder="unet")
    lora_alpha = args.lora_alpha
    lora_rank = args.lora_rank
    alpha = float(lora_alpha/lora_rank)
    processed_keys = set()
    with safe_open(
            os.path.join(args.ckpt_path, "pytorch_lora_weights.safetensors"),
            framework="pt") as f:
        state_dict = {key: f.get_tensor(key) for key in f.keys()}

    state_dict_unet = unet.state_dict()

    for key in state_dict.keys():
        if "lora_A" in key:
            lora_a_key = key
            lora_b_key = key.replace("lora_A", "lora_B")
            unet_key = key.replace(".lora_A.weight", ".weight").replace("unet.","")

            assert lora_b_key in state_dict and unet_key in state_dict_unet
            W_A = state_dict[lora_a_key]
            W_B = state_dict[lora_b_key]
            original_weight = state_dict_unet[unet_key]
            processed_keys.update([lora_a_key, lora_b_key])
            if len(original_weight.shape) == 4 and len(W_A.shape) == 4 and len(W_B.shape) == 4:
                out_channels, in_channels, kH, kW = original_weight.shape
                rank = W_A.shape[0]
                # print(rank)
                assert rank == lora_rank, f"lora rank should be {rank}, but {lora_alpha}"
                assert W_A.shape == (rank, in_channels, kH, kW), "W_A shape not matching! "
                assert W_B.shape == (out_channels, rank, 1, 1), "W_B shape not matching! "
                W_A_flat = W_A.view(rank, -1)  # (rank, in_channels * kH * kW)
                W_B_flat = W_B.view(out_channels, rank)  # (out_channels, rank)
                delta_W_flat = torch.matmul(W_B_flat, W_A_flat)  # (out_channels, in_channels * kH * kW)
                delta_W = delta_W_flat.view(out_channels, in_channels, kH, kW)
                merged_weight = original_weight + alpha * delta_W
            else:
                merged_weight = original_weight + alpha * torch.mm(W_B, W_A)
            state_dict_unet[unet_key] = merged_weight
        elif 'lora.up.weight' in key:
            lora_up_key = key
            lora_down_key = key.replace('lora.up.weight', 'lora.down.weight')

            original_weight_key = key.replace('.lora.up.weight', '.weight').replace("unet.", "")
            assert lora_down_key in state_dict and original_weight_key in state_dict_unet
            W_up = state_dict[lora_up_key]
            W_down = state_dict[lora_down_key]
            W_orig = state_dict_unet[original_weight_key]
            processed_keys.update([lora_up_key, lora_down_key])

            if W_orig.ndim == 2:
                delta_W = torch.matmul(W_up, W_down)
                W_merged = W_orig + alpha * delta_W

            else:
                print(f"Warning: Unhandled weight shape for {original_weight_key}")
                continue
            state_dict_unet[original_weight_key] = W_merged
    remaining_lora_keys = [k for k in state_dict.keys() if k not in processed_keys]
    if remaining_lora_keys:
        print("Warning: There are unprocessed LoRA weights:")
        for key in remaining_lora_keys:
            print(f" - {key}")
    print("Merge Done!")
    unet.load_state_dict(state_dict_unet)
    return unet

def main_worker(Unet, rank, gpu_id, image_names, weight_dtype, args):
    torch.cuda.set_device(gpu_id)

    model = OSDFace_test(args, gpu_id, Unet).to(gpu_id)

    for image_name in tqdm(image_names):
        output_file_path = os.path.join(args.output_dir, os.path.basename(image_name))
        if os.path.exists(output_file_path):
            print(f"Skipping {os.path.basename(image_name)} as it already exists in the output directory.")
            continue

        input_image = Image.open(image_name).convert('RGB')
        with torch.no_grad():
            lq = F.to_tensor(input_image).unsqueeze(0).to(gpu_id, dtype=weight_dtype) * 2 - 1
            if lq.shape[2] == lq.shape[3] :
                lq = Fun.interpolate(lq, (512, 512), mode='bilinear', align_corners=True)
            output_image = model(lq)
            output_pil = transforms.ToPILImage()(output_image[0].cpu())
            output_pil.save(output_file_path)

def run_inference(args, Unet):
    if os.path.isdir(args.input_image):
        image_names = sorted(glob.glob(f'{args.input_image}/*.[jpJP][pnPN]*[gG]'))
    else:
        image_names = [args.input_image]

    exist_images = sorted(glob.glob(f'{args.output_dir}/*.[jpJP][pnPN]*[gG]'))
    exist_image_names = [os.path.basename(img) for img in exist_images]

    image_names = [img for img in image_names if os.path.basename(img) not in exist_image_names]

    print("Exist image names: ", '\n', exist_image_names)

    random.shuffle(image_names)

    weight_dtype = torch.float32
    if args.mixed_precision == "fp16":
        weight_dtype = torch.float16

    num_gpus = len(args.gpu_ids)
    images_per_gpu = len(image_names) // num_gpus
    processes = []

    for rank, gpu_id in enumerate(args.gpu_ids):
        start_idx = rank * images_per_gpu
        end_idx = start_idx + images_per_gpu if rank != num_gpus - 1 else len(image_names)
        image_subset = image_names[start_idx:end_idx]

        p = mp.Process(target=main_worker, args=(Unet, rank, gpu_id, image_subset, weight_dtype, args))
        p.start()
        processes.append(p)

    for p in processes:
        p.join()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_image', '-i', type=str, required=True,
                        help='path to the input image')
    parser.add_argument('--output_dir', '-o', type=str, required=True,
                        help='the directory to save the output')
    parser.add_argument('--pretrained_model_name_or_path', type=str, default='preset/models/stable-diffusion-2-1-base', help='sd model path')
    parser.add_argument('--seed', type=int, default=114, help='Random seed to be used')
    parser.add_argument("--process_size", type=int, default=512)
    parser.add_argument("--ckpt_path", type=str, default=None)
    parser.add_argument("--mixed_precision", type=str, choices=['fp16', 'fp32'], default="fp32")
    parser.add_argument("--img_encoder_weight", type=str, default="preset/pretrained/associate_2.ckpt")
    parser.add_argument("--gpu_ids", nargs='+', type=int, default=[0], help="List of GPU IDs to use")
    parser.add_argument("--cat_prompt_embedding", action="store_true", help="use cat_prompt_embedding to exchange embedding change")
    parser.add_argument("--use_pos_embedding", action="store_true", help="use 2D pos embedding")
    parser.add_argument("--use_att_pool", action="store_true", help="use attention pool layer")
    parser.add_argument("--merge_lora", action="store_true", help="merge LoRA weights")
    parser.add_argument("--lora_rank", type=int, default=16)
    parser.add_argument("--lora_alpha", type=float, default=16)

    args = parser.parse_args()
    mp.set_start_method('spawn', force=True)

    seed = args.seed
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    Unet = None
    if args.merge_lora:
        Unet = merge_Unet(args)
    os.makedirs(args.output_dir, exist_ok=True)
    print(f'There are {len(glob.glob(f"{args.input_image}/*"))} images to process.')

    run_inference(args, Unet)