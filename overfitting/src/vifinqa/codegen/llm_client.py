"""LLM clients.

- HfBatchClient:   plain transformers batched generation. SLOWER than vLLM but
                   works everywhere, including Kaggle T4 (SM75) where recent
                   vLLM V1 engines fail to start. RECOMMENDED on T4.
- VllmBatchClient: offline batched inference via vLLM (fast, but needs a version
                   whose engine supports the GPU: on T4 pin vllm==0.7.3).
- OpenAIClient:    any OpenAI-compatible server (LM Studio / Ollama / vLLM serve)
                   for small-scale local testing.
- NoLLM:           placeholder that returns nothing (rule-based-only runs).

All clients share one interface:
    chat_batch(conversations, n, temperature, max_tokens) -> list[list[str]]
"""
from __future__ import annotations


class NoLLM:
    name = "none"

    def chat_batch(self, conversations, n=1, temperature=0.0, max_tokens=768):
        return [[] for _ in conversations]


class HfBatchClient:
    """transformers + batched generate. No vLLM, no custom CUDA engine.

    model_id options (all <=14B, open, pre-2026-06):
      Qwen/Qwen2.5-Coder-14B-Instruct-AWQ   ~9.5GB, needs autoawq, best quality
      Qwen/Qwen2.5-Coder-7B-Instruct        fp16 15GB / 4bit ~5GB via load_4bit
    """

    name = "hf"

    def __init__(self, model: str, load_4bit: bool = False, batch_size: int = 8,
                 max_input_tokens: int = 5000, dtype: str = "float16"):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.batch_size = batch_size
        self.max_input = max_input_tokens
        self.tok = AutoTokenizer.from_pretrained(model, trust_remote_code=True)
        if self.tok.pad_token_id is None:
            self.tok.pad_token = self.tok.eos_token
        self.tok.padding_side = "left"   # required for decoder-only batching
        # Preserve the generation marker and the final task anchor when an OOM
        # retry shortens a long evidence prompt. Right truncation leaves the
        # model mid-CSV, which makes it continue the table instead of writing code.
        self.tok.truncation_side = "left"

        kwargs = dict(device_map="auto", trust_remote_code=True,
                      torch_dtype=getattr(torch, dtype),
                      attn_implementation="sdpa")   # T4-safe (no FlashAttention)
        if load_4bit and "awq" not in model.lower() and "gptq" not in model.lower():
            from transformers import BitsAndBytesConfig
            kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True, bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_use_double_quant=True)
        self.model = AutoModelForCausalLM.from_pretrained(model, **kwargs)
        self.model.eval()
        self.device = next(self.model.parameters()).device
        print(f"[hf] loaded {model} on {self.device} "
              f"(4bit={load_4bit}, batch={batch_size})")

    def chat_batch(self, conversations, n=1, temperature=0.0, max_tokens=768):
        import gc
        import torch
        if n > 1 and not temperature:
            raise ValueError("HF self-consistency with n>1 requires temperature>0")
        texts = [self.tok.apply_chat_template(c, tokenize=False,
                                              add_generation_prompt=True)
                 for c in conversations]
        # group similar lengths together -> far less padding waste
        order = sorted(range(len(texts)), key=lambda i: len(texts[i]))
        results: list[list[str]] = [[] for _ in texts]

        try:
            from tqdm import tqdm
            bar = tqdm(total=len(order), desc=f"hf-gen(n={n})")
        except ImportError:
            bar = None

        cursor = 0
        adaptive_batch = max(1, self.batch_size)
        # ``num_return_sequences`` multiplies the KV cache just like another
        # prompt in the batch.  On two T4s a 14B model can therefore OOM at
        # batch=1 when n=5.  Adapt the two dimensions independently instead of
        # giving up once the prompt batch reaches one.
        adaptive_returns = max(1, n)
        adaptive_input = max(1, self.max_input)
        while cursor < len(order):
            idx = order[cursor:cursor + adaptive_batch]
            grouped: list[list[str]] = [[] for _ in idx]
            sample_cursor = 0
            retry_with_smaller_prompt_batch = False
            while sample_cursor < n:
                n_return = min(adaptive_returns, n - sample_cursor)
                enc = out = new = dec = None
                try:
                    enc = self.tok([texts[i] for i in idx], return_tensors="pt",
                                   padding=True, truncation=True,
                                   max_length=adaptive_input).to(self.device)
                    gen_kwargs = dict(
                        max_new_tokens=max_tokens,
                        num_return_sequences=n_return,
                        pad_token_id=self.tok.pad_token_id,
                    )
                    if temperature and temperature > 0:
                        gen_kwargs.update(do_sample=True, temperature=temperature,
                                          top_p=0.95)
                    else:
                        gen_kwargs.update(do_sample=False)
                    with torch.inference_mode():
                        out = self.model.generate(**enc, **gen_kwargs)
                    new = out[:, enc["input_ids"].shape[1]:]
                    dec = self.tok.batch_decode(new, skip_special_tokens=True)
                except RuntimeError as e:
                    if "out of memory" not in str(e).lower():
                        if bar:
                            bar.close()
                        raise
                    del enc, out, new, dec
                    gc.collect()
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                    if len(idx) > 1:
                        adaptive_batch = max(1, len(idx) // 2)
                        retry_with_smaller_prompt_batch = True
                        print(
                            f"[hf-oom] retrying item {cursor} with prompt_batch="
                            f"{adaptive_batch}, return_batch={adaptive_returns}",
                            flush=True,
                        )
                        break
                    if n_return > 1:
                        adaptive_returns = max(1, n_return // 2)
                        print(
                            f"[hf-oom] retrying item {cursor} with prompt_batch=1, "
                            f"return_batch={adaptive_returns}", flush=True,
                        )
                        continue
                    if adaptive_input > 2048:
                        adaptive_input = max(2048, int(adaptive_input * 0.75))
                        print(
                            f"[hf-oom] retrying item {cursor} with prompt_batch=1, "
                            f"return_batch=1, max_input={adaptive_input}", flush=True,
                        )
                        continue
                    if bar:
                        bar.close()
                    raise RuntimeError(
                        "CUDA OOM at HF prompt_batch=1, return_batch=1 and "
                        "max_input=2048; reduce --k or use a smaller model"
                    ) from e

                for j in range(len(idx)):
                    grouped[j].extend(dec[j * n_return:(j + 1) * n_return])
                sample_cursor += n_return
                del enc, out, new, dec

            if retry_with_smaller_prompt_batch:
                continue
            for j, i in enumerate(idx):
                results[i] = grouped[j]
            cursor += len(idx)
            if bar:
                bar.update(len(idx))
        if bar:
            bar.close()
        return results


class VllmBatchClient:
    name = "vllm"

    def __init__(self, model: str, tensor_parallel: int = 2, max_model_len: int = 6144,
                 gpu_mem: float = 0.92, dtype: str = "half", quantization: str | None = None,
                 enforce_eager: bool = False, download_dir: str | None = None,
                 seed: int = 13):
        import re
        from vllm import LLM  # heavy import, Kaggle only
        self.seed = seed
        kwargs = dict(model=model, tensor_parallel_size=tensor_parallel,
                      max_model_len=max_model_len, gpu_memory_utilization=gpu_mem,
                      dtype=dtype, trust_remote_code=True)
        if enforce_eager:
            kwargs["enforce_eager"] = True
        if quantization:
            kwargs["quantization"] = quantization
        if download_dir:
            kwargs["download_dir"] = download_dir
        # vLLM renames/removes engine kwargs across versions (e.g. swap_space);
        # drop whatever the installed version does not accept and retry.
        for _ in range(len(kwargs)):
            try:
                self.llm = LLM(**kwargs)
                break
            except TypeError as e:
                m = re.search(r"unexpected keyword argument '([A-Za-z_]+)'", str(e))
                if not m or m.group(1) == "model" or m.group(1) not in kwargs:
                    raise
                dropped = kwargs.pop(m.group(1))
                print(f"[vllm-compat] dropped kwarg {m.group(1)}={dropped!r}")
        else:
            raise RuntimeError("could not construct vllm.LLM")

    def chat_batch(self, conversations, n=1, temperature=0.0, max_tokens=768):
        from vllm import SamplingParams
        sp = SamplingParams(n=n, temperature=temperature,
                            top_p=0.95 if temperature > 0 else 1.0,
                            max_tokens=max_tokens, seed=self.seed)
        outs = self.llm.chat(conversations, sp)   # order preserved
        return [[o.text for o in out.outputs] for out in outs]


class OpenAIClient:
    name = "openai"

    def __init__(self, base_url: str, model: str, api_key: str = "not-needed"):
        from openai import OpenAI
        self.client = OpenAI(base_url=base_url, api_key=api_key)
        self.model = model

    def chat_batch(self, conversations, n=1, temperature=0.0, max_tokens=768):
        out = []
        for msgs in conversations:
            samples = []
            for _ in range(max(1, n)):
                r = self.client.chat.completions.create(
                    model=self.model, messages=msgs,
                    temperature=temperature, max_tokens=max_tokens)
                samples.append(r.choices[0].message.content or "")
            out.append(samples)
        return out
