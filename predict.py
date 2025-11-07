# predict.py (최종 버전: 로컬 모델 로드)
import torch
import os
from cog import BasePredictor, Input, Path
from transformers import AutoModelForCausalLM, AutoTokenizer

# cog.yaml에서 모델을 다운로드한 로컬 경로를 지정합니다.
LOCAL_MODEL_PATH = "/src/models" 

class Predictor(BasePredictor):
    def setup(self):
        """모델을 로컬 캐시에서 로드합니다. (부팅 시간 단축)"""
        print("Loading model and tokenizer from local cache...")
        
        # 1. 토크나이저와 모델을 로컬 경로에서 로드합니다.
        self.tokenizer = AutoTokenizer.from_pretrained(
            LOCAL_MODEL_PATH
        )
        
        # 2. 모델 로드 시 HF 토큰은 더 이상 필요 없으며, 로컬 경로만 사용합니다.
        self.model = AutoModelForCausalLM.from_pretrained(
            LOCAL_MODEL_PATH,
            torch_dtype=torch.float16,  
            device_map="auto"          
        )
        
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