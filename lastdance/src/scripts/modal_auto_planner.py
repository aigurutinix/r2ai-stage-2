"""Run auto_planner.py on Modal GPUs using vLLM for high throughput.

This script wraps the auto_planner logic to run on a Modal H200/A100.
It reads the questions locally, sends them to Modal, and writes the JSONL output.
"""

from __future__ import annotations

import json
from pathlib import Path
import modal

app = modal.App("vifinqa-auto-planner-vllm")
cache = modal.Volume.from_name("vifinqa-huggingface-cache", create_if_missing=True)

# Use official NVIDIA CUDA devel image to provide nvcc for flashinfer JIT compilation
image = modal.Image.from_registry("nvidia/cuda:12.1.1-devel-ubuntu22.04", add_python="3.11").pip_install(
    "vllm",
    "pyairports"
)

QWEN_MODEL = "Qwen/Qwen2.5-14B-Instruct"
DEEPSEEK_MODEL = "deepseek-ai/DeepSeek-R1-Distill-Qwen-14B"

@app.cls(image=image, gpu="H200", timeout=7200, scaledown_window=300, volumes={"/cache": cache})
class QwenPlanner:
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

    @modal.method()
    def generate(self, prompts: list[str], max_new_tokens: int) -> list[str]:
        from vllm import SamplingParams
        
        tokenizer = self.llm.get_tokenizer()
        formatted_prompts = []
        for prompt in prompts:
            # Force truncation of ultra-long tables to avoid >32768 tokens crash
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
            text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            formatted_prompts.append(text)
            
        sampling_params = SamplingParams(
            temperature=0.0,
            max_tokens=max_new_tokens,
            skip_special_tokens=True
        )
        
        outputs = self.llm.generate(formatted_prompts, sampling_params)
        return [output.outputs[0].text for output in outputs]

@app.cls(image=image, gpu="H200", timeout=7200, scaledown_window=300, volumes={"/cache": cache})
class DeepSeekPlanner:
    @modal.enter()
    def load(self) -> None:
        from vllm import LLM
        self.llm = LLM(
            model=DEEPSEEK_MODEL,
            download_dir="/cache",
            max_model_len=32768,
            trust_remote_code=True,
            gpu_memory_utilization=0.9,
            enforce_eager=True
        )

    @modal.method()
    def generate(self, prompts: list[str], max_new_tokens: int) -> list[str]:
        from vllm import SamplingParams
        
        tokenizer = self.llm.get_tokenizer()
        formatted_prompts = []
        for prompt in prompts:
            # Force truncation of ultra-long tables to avoid >32768 tokens crash
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
            text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            formatted_prompts.append(text)
            
        sampling_params = SamplingParams(
            temperature=0.0,
            max_tokens=max_new_tokens,
            skip_special_tokens=True
        )
        
        outputs = self.llm.generate(formatted_prompts, sampling_params)
        return [output.outputs[0].text for output in outputs]

@app.local_entrypoint()
def main(
    prompts_file: str,
    output: str,
    max_new_tokens: int = 2400,
) -> None:
    prompts = json.loads(Path(prompts_file).read_text(encoding="utf-8"))
    print(f"Loaded {len(prompts)} prompts. Sending to Modal vLLM engine on H200...")
    
    qwen = QwenPlanner()
    deepseek = DeepSeekPlanner()
    
    batch_texts = [p["prompt"] for p in prompts]
    
    # Send all prompts to the vLLM engine at once. vLLM handles batching optimally inside the GPU.
    print(f"Generating Qwen responses for {len(prompts)} prompts...")
    qwen_responses = qwen.generate.remote(batch_texts, max_new_tokens)
    
    print(f"Generating DeepSeek responses for {len(prompts)} prompts...")
    deepseek_responses = deepseek.generate.remote(batch_texts, max_new_tokens)
        
    responses = []
    for p, raw in zip(prompts, qwen_responses):
        responses.append({
            "question_id": p["question_id"],
            "model": QWEN_MODEL,
            "raw_response": raw
        })
        
    for p, raw in zip(prompts, deepseek_responses):
        responses.append({
            "question_id": p["question_id"],
            "model": DEEPSEEK_MODEL,
            "raw_response": raw
        })
            
    Path(output).write_text(json.dumps(responses, ensure_ascii=False), encoding="utf-8")
    print(f"Saved {len(responses)} raw responses to {output}")
