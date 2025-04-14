import os, sys

sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), "../"))

import torch
import torch.nn.functional as F

if torch.cuda.is_available():
    # device = torch.cuda.current_device()
    torch.cuda.init()
    torch.backends.cuda.matmul.allow_tf32 = True
else:
    raise Exception("unable to initialize CUDA")

import os
import gc
import math
import datetime
import argparse
from tqdm import tqdm
from safetensors.torch import load_file
from torchvision.transforms import v2

import glob
import cv2
import time
import argparse
import concurrent.futures

import numpy as np
import decord

use_torch_bridge = False


from wan.utils.utils import cache_video


def get_depth_model(device="cuda:0"):
    if not os.path.exists(
        "./models/Depth-Anything-V2-Small/depth_anything_v2_vits.pth"
    ):
        print("depth model not found, downloading to ./models/Depth-Anything-V2-Small")
        from huggingface_hub import snapshot_download

        snapshot_download(
            repo_type="model",
            repo_id="depth-anything/Depth-Anything-V2-Small",
            local_dir="./models/Depth-Anything-V2-Small",
            allow_patterns="*.pth",
        )

    from utils.depth_anything_v2.dpt import DepthAnythingV2

    depth_model = DepthAnythingV2(
        encoder="vits", features=64, out_channels=[48, 96, 192, 384],device=device
    )
    depth_model.load_state_dict(
        torch.load(
            "./models/Depth-Anything-V2-Small/depth_anything_v2_vits.pth",
            map_location="cpu",
            weights_only=True,
        )
    )
    depth_model = depth_model.to(device)
    depth_model.requires_grad_(False)
    depth_model.eval()

    return depth_model


def resize512(height, width):
    r = 512 / min(height, width)

    new_height = int(height * r)
    new_width = int(width * r)

    new_height = (new_height // 2) * 2
    new_width = (new_width // 2) * 2

    return new_height, new_width

def handle_single(depth_model, video_path, output_path,skip_existing=False):
    try:
        if use_torch_bridge:
            decord.bridge.set_bridge("torch")

        device = next(depth_model.parameters()).device
        st = time.time()
        print("handle==", video_path, device)
        if os.path.exists(output_path):
            print('exists!')
            if skip_existing:
                return

        vr = decord.VideoReader(video_path)
        control_pixels = vr[:]
        
        f, height, width, _ = control_pixels.shape
        height, width = resize512(height, width)

        if not use_torch_bridge:
            control_pixels = torch.from_numpy(control_pixels.asnumpy())

        # (F, H, W, C) -> (F, C, H, W) -> (1, F, C, H, W)
        control_pixels = control_pixels.movedim(3, 1).unsqueeze(0)

        transform = v2.Compose([
            v2.ToDtype(torch.float32, scale=True),
            v2.Resize(size=(height, width))
        ])
        control_pixels = transform(control_pixels).squeeze(0)  # 变成 (F, C, H, W)

        depth_frames = []
        # 遍历每一帧进行模型推理
        for i in range(control_pixels.shape[0]):
            d_input = control_pixels[i].movedim(0, -1).cpu().float().numpy() * 0.5 + 0.5
            depth = depth_model.infer_image(d_input)
            depth = (depth - depth.min()) / (depth.max() - depth.min())
            # depth_frames.append(depth.cpu().numpy() * 2 - 1)
            depth_frames.append(depth * 2 - 1)

        # depth_frames_np = np.stack(depth_frames, axis=0)
        # depth_frames_tensor = torch.from_numpy(depth_frames_np).float()
        depth_frames_tensor = torch.stack(depth_frames, dim=0) 

        depth_frames_tensor = depth_frames_tensor.unsqueeze(0).repeat(3, 1, 1, 1)

        depth_frames_tensor = depth_frames_tensor.to(device=device, dtype=torch.bfloat16)

        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        cache_video(
            tensor=depth_frames_tensor[None],
            save_file=output_path,
            fps=16,
            nrow=1,
            normalize=True,
            value_range=(-1, 1),
        )

        gc.collect()
        torch.cuda.empty_cache()

        print("cost==", time.time() - st, f)
    except:
        import traceback
        traceback.print_exc()


def process_batch_videos(depth_model, video_paths,skip_existing):
    for video_path in video_paths:
        output_path = video_path.replace("/train/", "/train_depth/")
        handle_single(depth_model, video_path, output_path,skip_existing)
        
def test():
    depth_model = get_depth_model('cuda:0')
    
    handle_single(depth_model,sys.argv[1],sys.argv[2])


if __name__ == "__main__":

    # test()
    # exit()
    # handle_single(sys.argv[1], sys.argv[2])

    parser = argparse.ArgumentParser()
    parser.add_argument("--input-folder", type=str)
    parser.add_argument("-j", type=int, default=3, help="Num workers")
    parser.add_argument('--skip-existing', action='store_true', help='Skip generating videos that already exist.')
    args = parser.parse_args()
    # num_workers = args.j
    num_gpus = torch.cuda.device_count()
    num_workers = num_gpus
    #TODO: 4卡有问题目前
    num_gpus = 2

    gpu_ids = [i for i in range(num_gpus) if torch.cuda.is_available()]
    print(f"avaliable gpu ids: {gpu_ids}")

    os.makedirs(os.path.join(args.input_folder, "train_depth"),exist_ok = True)
    video_paths = glob.glob(os.path.join(args.input_folder, "train/*/*.mp4"))

    batch_size = (len(video_paths) + num_workers - 1) // num_workers
    print(f"Num videos: {len(video_paths)} {batch_size = }")
    video_chunks = [
        video_paths[i : i + batch_size] for i in range(0, len(video_paths), batch_size)
    ]

    depth_models = []
    for gpu_id in gpu_ids:
        depth_models.append(get_depth_model(f"cuda:{gpu_id}"))

    with concurrent.futures.ThreadPoolExecutor() as executor:
        futures = []
        for i, chunk in enumerate(video_chunks):
            # init detector
            depth_model = depth_models[i % len(gpu_ids)]

            futures.append(executor.submit(process_batch_videos, depth_model, chunk,args.skip_existing))
        for future in concurrent.futures.as_completed(futures):
            future.result()
