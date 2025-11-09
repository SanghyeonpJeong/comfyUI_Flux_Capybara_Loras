import torch
import os
from cog import BasePredictor, Input, Path
from diffusers import FluxPipeline

# 🌟 (수정) cog.yaml이 다운로드한 로컬 경로를 지정
LOCAL_MODEL_PATH = "/src/models"
LORA_PATH = "/src/loras/Flux_Capybara_v1.safetensors"

class Predictor(BasePredictor):
    def setup(self):
        """🌟 (수정) 런타임 다운로드 '삭제'. 로컬 캐시에서 로드합니다. 🌟"""
        print("Booting... Loading FLUX.1-dev pipeline from local cache...")
        
        # 1. 런타임에 Gated Model 다운로드 (삭제됨)
        
        # 2. 로컬 캐시 경로에서 모델 로드
        self.pipe = FluxPipeline.from_pretrained(
            LOCAL_MODEL_PATH, # 🌟 로컬 경로
            torch_dtype=torch.bfloat16
            # 🌟 토큰(token=)이 더 이상 필요 없음
        )
        
        # 3. VRAM 절약을 위해 CPU 오프로드
        self.pipe.enable_model_cpu_offload()
        
        # 4. LoRA 로드 (필요한 경우)
        # self.pipe.load_lora_weights(LORA_PATH)
        # print(f"LoRA loaded from {LORA_PATH}")

        print("FluxPipeline loaded successfully. Booting complete.")

    def predict(
        self,
        prompt: str = Input(description="Prompt for the model."),
        height: int = Input(description="Height of the image.", default=1024),
        width: int = Input(description="Width of the image.", default=1024),
        num_inference_steps: int = Input(description="Number of inference steps.", default=50),
        guidance_scale: float = Input(description="Guidance scale.", default=3.5)
    ) -> Path: 
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