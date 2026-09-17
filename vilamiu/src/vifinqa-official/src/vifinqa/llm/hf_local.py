"""Portable local ChatLLM using Transformers across CUDA, MPS, and CPU."""

from __future__ import annotations

from transformers import AutoModelForCausalLM, AutoTokenizer

from vifinqa.llm.base import LLMError
from vifinqa.models.device import pick_attn_implementation, pick_device, pick_dtype

_MAX_NEW_TOKENS = 8192


class HFLocalLLM:
    def __init__(
        self,
        model_name: str,
        *,
        hf_token: str | None = None,
        device: str | None = None,
        dtype: str | None = None,
        max_completion_tokens: int = _MAX_NEW_TOKENS,
        temperature: float = 0,
    ) -> None:
        device = device if device not in (None, "auto") else pick_device()
        self._max_completion_tokens = max_completion_tokens
        self._temperature = temperature
        self._tokenizer = AutoTokenizer.from_pretrained(model_name, token=hf_token)
        self._model = AutoModelForCausalLM.from_pretrained(
            model_name,
            dtype=pick_dtype(device) if dtype in (None, "auto") else dtype,
            device_map=device,
            attn_implementation=pick_attn_implementation(device),
            token=hf_token,
        )

    def complete(self, *, system: str, user: str) -> str:
        messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ]
        prompt = self._tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = self._tokenizer(prompt, return_tensors="pt").to(self._model.device)
        try:
            generation: dict[str, object] = {
                "max_new_tokens": self._max_completion_tokens,
                "do_sample": self._temperature > 0,
            }
            if self._temperature > 0:
                generation["temperature"] = self._temperature
            output = self._model.generate(**inputs, **generation)
        except RuntimeError as exc:
            # CUDA/MPS OOM or kernel failures can leave the device unsafe; stop the run.
            raise LLMError(f"Local Hugging Face LLM generation failed: {exc}", fatal=True) from exc
        generated = output[0][inputs["input_ids"].shape[1] :]
        return self._tokenizer.decode(generated, skip_special_tokens=True)
