import torch
import os
from cog import BasePredictor, Input, Path
from transformers import AutoModelForCausalLM, AutoTokenizer

# Hugging Face 모델 ID를 정의합니다. (사용자님의 실제 모델 ID로 교체하세요)
MODEL_ID = "Your_HuggingFace_Model/Name" # 예시: "meta-llama/Llama-2-7b-chat-hf"

class Predictor(BasePredictor):
    def setup(self):
        """모델을 로드하는 동안 초기화합니다."""
        print("Loading model and tokenizer...")

        # 1. 환경 변수에서 Hugging Face 토큰을 로드합니다.
        # Replicate Secret에 설정된 이름과 일치해야 합니다 (예: HF_TOKEN)
        huggingface_token = os.environ.get("HF_TOKEN")
        
        if not huggingface_token:
            print("WARNING: HF_TOKEN environment variable not set. This may cause issues if the model requires authentication.")
        
        # 2. 토크나이저와 모델을 로드합니다.
        # 'use_auth_token' 인수에 로드된 토큰을 전달합니다.
        self.tokenizer = AutoTokenizer.from_pretrained(
            MODEL_ID,
            use_auth_token=huggingface_token
        )
        
        self.model = AutoModelForCausalLM.from_pretrained(
            MODEL_ID,
            use_auth_token=huggingface_token,
            torch_dtype=torch.float16,  # 필요에 따라 정밀도 설정
            device_map="auto"          # 적절한 디바이스 맵핑
        )
        
        print("Model loaded successfully.")

    def predict(
        self,
        prompt: str = Input(description="Inquiry for the model."),
        max_new_tokens: int = Input(description="Maximum number of tokens to generate.", default=128, ge=1, le=2048),
        temperature: float = Input(description="Creativity of the generation.", default=0.9, ge=0.01, le=1.0),
        top_p: float = Input(description="Probability mass of tokens to consider.", default=0.9, ge=0.0, le=1.0),
    ) -> str:
        """프롬프트를 사용하여 텍스트를 생성합니다."""
        
        # 3. 입력 프롬프트를 토큰화합니다.
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
        
        # 4. 텍스트를 생성합니다.
        with torch.no_grad():
            output_tokens = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
                do_sample=True, # 샘플링을 활성화하여 temperature 및 top_p 적용
            )
        
        # 5. 생성된 토큰을 디코딩하고 불필요한 부분(프롬프트)을 제거합니다.
        output_text = self.tokenizer.decode(
            output_tokens[0], 
            skip_special_tokens=True
        )
        
        # 생성된 텍스트에서 입력 프롬프트를 제외하고 순수한 응답만 반환합니다.
        # 이는 모델의 특성에 따라 다를 수 있으므로 조정이 필요할 수 있습니다.
        return output_text.replace(prompt, "", 1).strip()