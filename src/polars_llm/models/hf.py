"""Hugging Face / Unsloth inference wrapper (GPU).

Loads a model (optionally with a LoRA adapter) via Unsloth in bf16 with no 4-bit
quantization, and generates completions with greedy decoding and batching.

The heavy imports (``unsloth``, ``torch``) happen lazily inside ``__init__`` /
``generate`` so this module can be imported on a CPU-only machine (for tests and
for the mock pipeline) without pulling in the GPU stack.

Thinking/reasoning is disabled by passing ``enable_thinking=False`` to the
tokenizer chat template (Qwen3.5 is a hybrid reasoning model, thinking-on by
default, and dropped the `/no_think` soft switch).
"""

from __future__ import annotations

import time

from .base import Generation

DEFAULT_MAX_SEQ_LENGTH = 4096


class HFModel:
    """A batched, greedy-decoding wrapper around an Unsloth-loaded model."""

    def __init__(
        self,
        model_id: str,
        adapter: str | None = None,
        *,
        max_seq_length: int = DEFAULT_MAX_SEQ_LENGTH,
        enable_thinking: bool = False,
        seed: int = 0,
    ):
        import torch
        from unsloth import FastLanguageModel

        self.name = model_id
        self.adapter = adapter
        self.enable_thinking = enable_thinking
        self.max_seq_length = max_seq_length

        FastLanguageModel.seed = seed  # best-effort determinism hook

        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name=model_id,
            max_seq_length=max_seq_length,
            dtype=torch.bfloat16,  # bf16, no quantization
            load_in_4bit=False,
        )

        if adapter:
            from peft import PeftModel

            model = PeftModel.from_pretrained(model, adapter)

        FastLanguageModel.for_inference(model)

        # Left-pad for decoder-only batched generation.
        tokenizer.padding_side = "left"
        if tokenizer.pad_token_id is None:
            tokenizer.pad_token = tokenizer.eos_token

        self.model = model
        self.tokenizer = tokenizer
        self._torch = torch

    def _render(self, messages: list[dict]) -> str:
        return self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=self.enable_thinking,
        )

    def generate(
        self,
        messages_list: list[list[dict]],
        max_new_tokens: int = 768,
        batch_size: int = 8,
    ) -> list[Generation]:
        torch = self._torch
        results: list[Generation] = []

        for start in range(0, len(messages_list), batch_size):
            batch = messages_list[start : start + batch_size]
            prompts = [self._render(m) for m in batch]

            enc = self.tokenizer(
                prompts,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=self.max_seq_length,
            ).to(self.model.device)
            prompt_len = enc["input_ids"].shape[1]
            prompt_tokens = enc["attention_mask"].sum(dim=1).tolist()

            t0 = time.perf_counter()
            with torch.no_grad():
                out = self.model.generate(
                    **enc,
                    max_new_tokens=max_new_tokens,
                    do_sample=False,  # greedy / temperature 0
                    num_beams=1,
                    use_cache=True,
                    pad_token_id=self.tokenizer.pad_token_id,
                )
            elapsed = time.perf_counter() - t0
            per_item_latency = elapsed / max(len(batch), 1)

            new_tokens = out[:, prompt_len:]
            texts = self.tokenizer.batch_decode(new_tokens, skip_special_tokens=True)
            eos = self.tokenizer.eos_token_id
            for j, text in enumerate(texts):
                row = new_tokens[j]
                completion_tokens = int((row != self.tokenizer.pad_token_id).sum())
                if eos is not None:
                    # don't count anything after the first eos
                    eos_pos = (row == eos).nonzero()
                    if eos_pos.numel() > 0:
                        completion_tokens = int(eos_pos[0].item())
                results.append(
                    Generation(
                        text=text,
                        prompt_tokens=int(prompt_tokens[j]),
                        completion_tokens=completion_tokens,
                        latency_s=round(per_item_latency, 4),
                    )
                )
        return results
