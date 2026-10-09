"""Verify HFModel configures batched generation correctly, without a GPU.

We inject fake ``torch`` / ``unsloth`` modules so ``HFModel.__init__`` runs on a
CPU-only machine, then assert it sets **left** padding. Left padding is what
guarantees batched greedy decoding produces the same tokens as single-prompt
decoding (and makes the ``out[:, prompt_len:]`` new-token slice valid).
"""

import sys
import types


def _install_fakes(monkeypatch):
    torch_mod = types.ModuleType("torch")
    torch_mod.bfloat16 = "bfloat16"
    monkeypatch.setitem(sys.modules, "torch", torch_mod)

    class FakeTokenizer:
        def __init__(self):
            self.padding_side = "right"  # HFModel must flip this to "left"
            self.pad_token_id = None
            self.eos_token = "<eos>"
            self.pad_token = None

    tok = FakeTokenizer()

    class FastLanguageModel:
        seed = 0

        @staticmethod
        def from_pretrained(**kwargs):
            FastLanguageModel.last_kwargs = kwargs
            return ("FAKE_MODEL", tok)

        @staticmethod
        def for_inference(model):
            FastLanguageModel.inferenced = model

    unsloth_mod = types.ModuleType("unsloth")
    unsloth_mod.FastLanguageModel = FastLanguageModel
    monkeypatch.setitem(sys.modules, "unsloth", unsloth_mod)
    return FastLanguageModel, tok


def test_hfmodel_sets_left_padding_and_bf16_no_4bit(monkeypatch):
    FastLanguageModel, tok = _install_fakes(monkeypatch)

    from polars_llm.models.hf import HFModel

    model = HFModel("Qwen/Qwen3.5-4B", seed=7)

    # Left padding is the key invariant for batched greedy == unbatched greedy.
    assert tok.padding_side == "left"
    # pad token falls back to eos when missing.
    assert tok.pad_token == "<eos>"

    # Loaded in bf16 with no 4-bit quantization.
    kw = FastLanguageModel.last_kwargs
    assert kw["dtype"] == "bfloat16"
    assert kw["load_in_4bit"] is False
    assert kw["model_name"] == "Qwen/Qwen3.5-4B"

    assert model.name == "Qwen/Qwen3.5-4B"
    assert model.enable_thinking is False
