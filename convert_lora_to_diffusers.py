import os
import argparse
import torch
from safetensors.torch import load_file, save_file

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input_lora",
        type=str,
        required=True,
        help="Path to comfyui LoRA .safetensors",
    )
    parser.add_argument(
        "--dtype",
        type=str,
        default=None,
        help="Optional dtype (bfloat16, float16, float32), defaults to input dtype",
    )
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()

    comfy_lora_sd = load_file(args.input_lora)
    diffusers_lora_sd = {}

    # 遍历 comfyui 格式的 key，如果带有 "diffusion_model." 前缀，则去掉
    for key, value in comfy_lora_sd.items():
        if key.startswith("diffusion_model."):
            new_key = key[len("diffusion_model."):]
        else:
            new_key = key
        diffusers_lora_sd[new_key] = value

    # 根据需要进行数据类型转换
    dtype = None
    if args.dtype == "bfloat16":
        dtype = torch.bfloat16
    elif args.dtype == "float16":
        dtype = torch.float16
    elif args.dtype == "float32":
        dtype = torch.float32

    if dtype is not None:
        dtype_min = torch.finfo(dtype).min
        dtype_max = torch.finfo(dtype).max
        for key in diffusers_lora_sd.keys():
            if diffusers_lora_sd[key].min() < dtype_min or diffusers_lora_sd[key].max() > dtype_max:
                print(f"warning: {key} has values outside of {dtype} {dtype_min} {dtype_max} range")
            diffusers_lora_sd[key] = diffusers_lora_sd[key].to(dtype)

    output_path = os.path.splitext(args.input_lora)[0] + "_diffusers.safetensors"
    save_file(diffusers_lora_sd, output_path)
    print(f"saved to {output_path}")
