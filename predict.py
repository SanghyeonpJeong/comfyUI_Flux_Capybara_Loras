import torch
import os
from cog import BasePredictor, Input, Path
# 🌟 (수정) Diffusers의 FluxPipeline을 import
from diffusers import FluxPipeline
# 🌟 (신규) 런타임에 snapshot_download를 사용하기 위해 import
from huggingface_hub import snapshot_download

MODEL_ID = "black-forest-labs/FLUX.1-dev"
LORA_PATH = "/src/loras/Flux_Capybara_v1.safetensors"

class Predictor(BasePredictor):
    def setup(self):
        """🌟 (수정) 런타임에 FluxPipeline을 다운로드하고 로드합니다 🌟"""
        print("Booting... Attempting to download FLUX.1-dev pipeline (this may take a while)...")
        
        # 1. 'push.yml'의 env: 에서 전달된 HF_TOKEN을 읽습니다.
        huggingface_token = os.environ.get("HF_TOKEN")
        
        if not huggingface_token:
            print("WARNING: HF_TOKEN environment variable not set. Download may fail.")
        
        # 2. 런타임에 Gated Model 다운로드 (22GB)
        downloaded_model_path = snapshot_download(
            repo_id=MODEL_ID,
            token=huggingface_token,
            cache_dir="/root/.cache/huggingface"
        )
        print("Model download complete.")

        # 3. 로컬 캐시 경로에서 모델 로드
        self.pipe = FluxPipeline.from_pretrained(
            downloaded_model_path,
            torch_dtype=torch.bfloat16
        )
        
        # 4. VRAM 절약을 위해 CPU 오프로드
        self.pipe.enable_model_cpu_offload()
        
        # 5. LoRA 로드
        # (이 부분의 주석을 해제하여 LoRA를 적용합니다)
        self.pipe.load_lora_weights(LORA_PATH)
        print(f"LoRA loaded from {LORA_PATH}")

        print("FluxPipeline loaded successfully. Booting complete.")

    def predict(
        self,
        prompt: str = Input(description="Prompt for the model."),
        height: int = Input(description="Height of the image.", default=1024),
        width: int = Input(description="Width of the image.", default=1024),
        num_inference_steps: int = Input(description="Number of inference steps.", default=50),
        guidance_scale: float = Input(description="Guidance scale.", default=3.5)
    ) -> Path: # 🌟 (수정) 반환 타입이 Path(파일)로 변경되었습니다.
        """프롬프트를 사용하여 이미지를 생성합니다."""
        
        image = self.pipe(
            prompt=prompt,
            height=height,
            width=width,
            guidance_scale=guidance_scale,
            num_inference_steps=num_inference_steps,
            generator=torch.Generator("cpu").manual_seed(0)
        ).images[0]
        
        output_path = "/tmp/output.png"
        image.save(output_path)
        return Path(output_path)