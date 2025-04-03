import os,sys
from huggingface_hub import hf_hub_download
from huggingface_hub import snapshot_download

dirname = os.path.dirname(os.path.abspath(__file__))


def prepare_wan_control():
    print(f"Preparing sky weights...")
    local_dir = os.path.join(dirname, "../models/wan-control")
    os.makedirs(local_dir, exist_ok=True)
    snapshot_download(
        repo_id="alibaba-pai/Wan2.1-Fun-1.3B-Control", local_dir=local_dir
    )
if __name__ == '__main__':
    prepare_wan_control()