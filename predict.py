import os
import requests

def download_if_needed(url, path):
    if not os.path.exists(path):
        print(f"Downloading {os.path.basename(path)} ...")
        with requests.get(url, stream=True) as r:
            r.raise_for_status()
            with open(path, "wb") as f:
                for chunk in r.iter_content(chunk_size=8192):
                    f.write(chunk)

# 모델 다운로드 경로 및 URL 설정
UNET_PATH = "model/flux1-dev.safetensors"
VAE_PATH = "model/vae_diffusion_pytorch_model.safetensors"

UNET_URL = "https://huggingface.co/black-forest-labs/FLUX.1-dev/resolve/main/flux1-dev.safetensors"
VAE_URL = "https://huggingface.co/black-forest-labs/FLUX.1-dev/resolve/main/vae/diffusion_pytorch_model.safetensors"

os.makedirs("model", exist_ok=True)
download_if_needed(UNET_URL, UNET_PATH)
download_if_needed(VAE_URL, VAE_PATH)

def predict(input_image):
    # 여기에 comfyui 또는 custom loader 코드로 모델 로딩
    unet = load_flux_unet(UNET_PATH)
    vae = load_vae(VAE_PATH)
    lora = load_lora("model/lora.safetensors")  # 레포에 이 파일만 포함
    # 아래는 환경 맞게 커스텀(ComfyUI, Diffusers 등)
    result = my_inference_function(unet, vae, lora, input_image)
    return result