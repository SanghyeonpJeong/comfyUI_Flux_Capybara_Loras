import torch
import os
from cog import BasePredictor, Input, Path
from transformers import AutoModelForCausalLM, AutoTokenizer
# 🌟🌟🌟 (신규) 런타임에 snapshot_download를 사용하기 위해 import 🌟🌟🌟
from huggingface_hub import snapshot_download

# (주의) cog.yaml에서 캐시하지 않으므로, 이 경로는 이제 로컬 경로가 아닙니다.
MODEL_ID = "black-forest-labs/FLUX.1-dev"
LORA_PATH = "/src/loras/Flux_Capybara_v1.safetensors"

class Predictor(BasePredictor):
    def setup(self):
        """🌟🌟🌟 (수정) 런타임에 모델을 다운로드하고 로드합니다 🌟🌟🌟"""
        print("Booting... Attempting to download model (this may take a while)...")
        
        # 1. 'push.yml'의 env: 에서 전달된 HF_TOKEN을 읽습니다.
        huggingface_token = os.environ.get("HF_TOKEN")
        
        if not huggingface_token:
            print("WARNING: HF_TOKEN environment variable not set. Download may fail.")
        
        # 2. 런타임에 모델 다운로드 (22GB)
        # (이것이 타임아웃될 수 있지만, 유일한 방법입니다.)
        downloaded_model_path = snapshot_download(
            repo_id=MODEL_ID,
            token=huggingface_token,
            cache_dir="/root/.cache/huggingface"
            # local_dir="/src/models" # 캐시를 사용하도록 local_dir 주석 처리
        )
        print("Model download complete.")

        # 3. 로컬 캐시 경로에서 모델 로드
        self.tokenizer = AutoTokenizer.from_pretrained(
            downloaded_model_path
        )
        self.model = AutoModelForCausalLM.from_pretrained(
            downloaded_model_path,
            torch_dtype=torch.float16,
            device_map="auto"
        )
        
        # 4. LoRA 로드 (필요한 경우)
        # (이 부분은 사용자님의 실제 LoRA 로드 코드로 대체해야 합니다)
        # self.model.load_adapter(LORA_PATH) 
        # print(f"LoRA loaded from {LORA_PATH}")

        print("Model loaded successfully. Booting complete.")

    def predict(
        self,
        prompt: str = Input(description="Inquiry for the model."),
        max_new_tokens: int = Input(description="Maximum number of tokens to generate.", default=128, ge=1, le=2048),
        temperature: float = Input(description="Creativity of the generation.", default=0.9, ge=0.01, le=1.0),
        top_p: float = Input(description="Probability mass of tokens to consider.", default=0.9, ge=0.0, le=1.0),
    ) -> str:
        """프롬프트를 사용하여 텍스트를 생성합니다."""
        
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
        
        with torch.no_grad():
            output_tokens = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
                do_sample=True,
            )
        
        output_text = self.tokenizer.decode(
            output_tokens[0], 
            skip_special_tokens=True
        )
        
        return output_text.replace(prompt, "", 1).strip()