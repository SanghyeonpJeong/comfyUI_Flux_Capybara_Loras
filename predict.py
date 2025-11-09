import torch
import os
from cog import BasePredictor, Input, Path
from diffusers import FluxPipeline

MODEL_ID = "black-forest-labs/FLUX.1-dev"
LORA_PATH = "/src/loras/Flux_Capybara_v1.safetensors"

class Predictor(BasePredictor):
    def setup(self):
        print("Booting... Loading FLUX.1-dev pipeline from base image...")

        # 모델은 기반 이미지에 이미 포함되어 있으니 새로 다운로드하지 않음
        self.pipe = FluxPipeline.from_pretrained(
            MODEL_ID,
            torch_dtype=torch.bfloat16
        )
        self.pipe.enable_model_cpu_offload()

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
