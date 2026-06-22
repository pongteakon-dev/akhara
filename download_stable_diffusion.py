
from huggingface_hub import snapshot_download
snapshot_download(
    repo_id='runwayml/stable-diffusion-v1-5',
    local_dir='./stable-diffusion-v1-5',
    ignore_patterns=['*.safetensors', 'flax_model*']
)