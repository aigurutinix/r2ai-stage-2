"""Host auto_planner.py as a Modal Web API Endpoint using FastAPI."""

from __future__ import annotations

import modal
from fastapi import FastAPI, Request
import json

app = modal.App("vifinqa-llm-api")
cache = modal.Volume.from_name("vifinqa-huggingface-cache", create_if_missing=True)

image = modal.Image.from_registry("nvidia/cuda:12.1.1-devel-ubuntu22.04", add_python="3.11").pip_install(
    "vllm",
    "pyairports",
    "fastapi"
)

QWEN_MODEL = "Qwen/Qwen2.5-14B-Instruct"

# Using FastAPI app for the endpoint
web_app = FastAPI()

@app.cls(image=image, gpu="H200", timeout=7200, scaledown_window=300, volumes={"/cache": cache})
class QwenAPI:
    @modal.enter()
    def load(self) -> None:
        from vllm import LLM
        self.llm = LLM(
            model=QWEN_MODEL,
            download_dir="/cache",
            max_model_len=32768,
            trust_remote_code=True,
            gpu_memory_utilization=0.9,
            enforce_eager=True
        )
        self.tokenizer = self.llm.get_tokenizer()

    @modal.web_endpoint(method="POST")
    def generate(self, data: dict):
        from vllm import SamplingParams
        
        prompts = data.get("prompts", [])
        max_new_tokens = data.get("max_tokens", 2400)
        
        if not prompts:
            return {"responses": []}
            
        formatted_prompts = []
        for prompt in prompts:
            sliced_prompt = prompt[:50000]
            messages = [
                {
                    "role": "system", 
                    "content": (
                        "Bạn là semantic planner cho truy vấn tài chính tiếng Việt. "
                        "Bạn chỉ sinh FinancialPlan JSON; không tính và không bịa số liệu. "
                        "Không viết Python, SQL, markdown hoặc văn bản ngoài JSON."
                    )
                },
                {"role": "user", "content": sliced_prompt},
            ]
            text = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            formatted_prompts.append(text)
            
        sampling_params = SamplingParams(
            temperature=0.0,
            max_tokens=max_new_tokens,
            skip_special_tokens=True
        )
        
        outputs = self.llm.generate(formatted_prompts, sampling_params)
        responses = [output.outputs[0].text for output in outputs]
        
        return {"responses": responses}
