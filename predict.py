import torch
import os
from cog import BasePredictor, Input, Path
from diffusers import FluxPipeline
# 🌟 (삭제) snapshot_download import 삭제

# 🌟 (수정) 기반 이미지(fofr/flux-dev)가 모델을 저장한 경로
# (이 경로는 fofr/flux-dev의 cog.yaml을 참조하여 확인했습니다)
MODEL_ID = "black-forest-labs/FLUX.1-dev"
LORA_PATH = "/src/loras/Flux_Capybara_v1.safetensors"

class Predictor(BasePredictor):
    def setup(self):
        """🌟 (수정) 런타임 다운로드 '삭제'. 기반 이미지의 모델을 로드합니다. 🌟"""
        print("Booting... Loading FLUX.1-dev pipeline from base image...")
        
        # 1. 런타임에 Gated Model 다운로드 (삭제됨)
        # (기반 이미지에 이미 포함되어 있음)
        
        # 2. 로컬 캐시 경로에서 모델 로드
        # (Diffusers는 MODEL_ID를 보고, 이미 캐시된 것을 확인하고 즉시 로드합니다)
        self.pipe = FluxPipeline.from_pretrained(
            MODEL_ID, # 🌟 토큰 없이 ID만 전달
            torch_dtype=torch.bfloat16
        )
        
        # 3. VRAM 절약을 위해 CPU 오프로드
        self.pipe.enable_model_cpu_offload()
        
        # 4. LoRA 로드 (필수!)
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
    ) -> Path: 
        """프롬프트를 사용하여 이미지를 생성합니다."""
        
        # 🌟 (수정) LoRA를 적용했으므로, predict 시 lora_scale을 전달할 수 있습니다.
        # (단, LoRA 로드 방식에 따라 이 코드는 달라질 수 있습니다.)
        image = self.pipe(
            prompt=prompt,
            height=height,
            width=width,
            guidance_scale=guidance_scale,
            num_inference_steps=num_inference_steps,
            generator=torch.Generator("cpu").manual_seed(0)
            # cross_attention_kwargs={"scale": 0.93} # LoRA 스케일 예시
        ).images[0]
        
        output_path = "/tmp/output.png"
        image.save(output_path)
        return Path(output_path)