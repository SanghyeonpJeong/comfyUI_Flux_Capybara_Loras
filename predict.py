import torch
import os
from cog import BasePredictor, Input, Path
from diffusers import FluxPipeline

MODEL_ID = "black-forest-labs/FLUX.1-dev"
LORA_PATH = "/src/loras/Flux_Capybara_v1.safetensors"

class Predictor(BasePredictor):
    def setup(self):
        # 기반 이미지에 모델 포함, 런타임 다운로드 삭제
        print("Booting... Loading FLUX.1-dev pipeline...")
        self.pipe = FluxPipeline.from_pretrained(
            MODEL_ID,
            torch_dtype=torch.bfloat16
        )
        self.pipe.enable_model_cpu_offload()
        self.pipe.load_lora_weights(LORA_PATH)
        print(f"LoRA loaded from {LORA_PATH}")
        print("Pipeline loaded. Booting complete.")

    def predict(
        self,
        prompt: str = Input(description="Prompt"),
        height: int = Input(description="Height", default=1024),
        width: int = Input(description="Width", default=1024),
        num_inference_steps: int = Input(description="Num steps", default=50),
        guidance_scale: float = Input(description="Guidance scale", default=3.5)
    ) -> Path:
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
