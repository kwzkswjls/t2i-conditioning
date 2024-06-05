import requests
from huggingface_hub import snapshot_download, configure_http_backend

proxy = None


def backend_factory() -> requests.Session:
    session = requests.Session()
    session.proxies = proxy
    return session


configure_http_backend(backend_factory=backend_factory)

models = [
    "runwayml/stable-diffusion-v1-5",
    "TencentARC/t2iadapter_canny_sd15v2",
    "lllyasviel/sd-controlnet-canny",
    "openai/clip-vit-large-patch14"
]

for model in models:
    snapshot_download(
        repo_id=model,
        allow_patterns=["*.json", "*pytorch_model.bin", "merges.txt"],
        local_dir=f"../ptms/{model}",
        local_dir_use_symlinks=False,
    )
