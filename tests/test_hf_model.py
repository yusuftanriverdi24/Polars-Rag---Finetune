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


# --- multimodal processor (Qwen3.5 via Unsloth) ------------------------------


class _T:
    """Just enough of a torch tensor (1-D or 2-D, list-backed) for HFModel."""

    def __init__(self, data):
        self.data = data

    @property
    def shape(self):
        if self.data and isinstance(self.data[0], list):
            return (len(self.data), len(self.data[0]))
        return (len(self.data),)

    def __getitem__(self, idx):
        if isinstance(idx, tuple):  # out[:, k:]
            rows, cols = idx
            return _T([r[cols] for r in self.data[rows]])
        return _T(self.data[idx]) if isinstance(self.data[idx], list) else self.data[idx]

    def __eq__(self, other):
        return _T([x == other for x in self.data])

    def __ne__(self, other):
        return _T([x != other for x in self.data])

    def sum(self, dim=None):
        if dim == 1:
            return _T([sum(r) for r in self.data])
        return sum(self.data)

    def tolist(self):
        return list(self.data)

    def nonzero(self):
        return _T([[i] for i, x in enumerate(self.data) if x])

    def numel(self):
        return len(self.data)

    def item(self):
        return self.data[0]


class _Enc(dict):
    def to(self, device):
        return self


def _install_processor_fakes(monkeypatch):
    torch_mod = types.ModuleType("torch")
    torch_mod.bfloat16 = "bfloat16"

    class _NoGrad:
        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return False

    torch_mod.no_grad = _NoGrad
    monkeypatch.setitem(sys.modules, "torch", torch_mod)

    PAD, EOS = 0, 2

    class FakeTokenizer:
        def __init__(self):
            self.padding_side = "right"
            self.pad_token_id = None
            self.pad_token = None
            self.eos_token = "<eos>"
            self.eos_token_id = EOS
            self.calls = []

        def apply_chat_template(self, messages, **kwargs):
            assert kwargs["tokenize"] is False
            return "PROMPT:" + messages[-1]["content"]

        def __call__(self, *args, **kwargs):
            assert not args, "text must be passed by keyword"
            self.calls.append(kwargs)
            texts = kwargs["text"]
            assert all(isinstance(t, str) for t in texts)
            # left-padded batch of width 3
            ids = [[PAD, 5, 6], [7, 8, 9]][: len(texts)]
            mask = [[0, 1, 1], [1, 1, 1]][: len(texts)]
            return _Enc(input_ids=_T(ids), attention_mask=_T(mask))

        def batch_decode(self, rows, skip_special_tokens=True):
            return [f"decoded{r}" for r in rows.data]

    class FakeProcessor:
        """Mimics a multimodal processor: first positional arg is `images`."""

        def __init__(self, tokenizer):
            self.tokenizer = tokenizer
            self.image_calls = []

        def __call__(self, images=None, text=None, **kwargs):
            self.image_calls.append(images)
            if images is not None:
                raise ValueError("Incorrect image source. Failed with Incorrect padding")
            return self.tokenizer(text=text, **kwargs)

        def apply_chat_template(self, *a, **k):
            raise AssertionError("chat template must go through the text tokenizer")

        def batch_decode(self, *a, **k):
            raise AssertionError("decoding must go through the text tokenizer")

    class FakeModel:
        device = "cpu"

        def generate(self, **kwargs):
            self.kwargs = kwargs
            ids = kwargs["input_ids"].data
            return _T([r + [11, EOS, PAD] for r in ids])

    tok = FakeTokenizer()
    proc = FakeProcessor(tok)
    fake_model = FakeModel()

    class FastLanguageModel:
        seed = 0

        @staticmethod
        def from_pretrained(**kwargs):
            return (fake_model, proc)

        @staticmethod
        def for_inference(model):
            pass

    unsloth_mod = types.ModuleType("unsloth")
    unsloth_mod.FastLanguageModel = FastLanguageModel
    monkeypatch.setitem(sys.modules, "unsloth", unsloth_mod)
    return proc, tok, fake_model


def test_hfmodel_uses_text_tokenizer_of_multimodal_processor(monkeypatch):
    proc, tok, fake_model = _install_processor_fakes(monkeypatch)

    from polars_llm.models.hf import HFModel

    model = HFModel("Qwen/Qwen3.5-4B")

    # Text work is configured on the underlying tokenizer, not the processor.
    assert model.tokenizer is tok
    assert model.processor is proc
    assert tok.padding_side == "left"
    assert tok.pad_token == "<eos>"

    msgs = [[{"role": "user", "content": "a"}], [{"role": "user", "content": "b"}]]
    gens = model.generate(msgs, batch_size=8)

    # Text was encoded via `text=`, and nothing ever reached `images`.
    assert tok.calls and tok.calls[0]["text"] == ["PROMPT:a", "PROMPT:b"]
    assert all(images is None for images in proc.image_calls)

    assert [g.text for g in gens] == ["decoded[11, 2, 0]", "decoded[11, 2, 0]"]
    assert [g.prompt_tokens for g in gens] == [2, 3]
    assert [g.completion_tokens for g in gens] == [1, 1]  # stops at first eos
    assert fake_model.kwargs["do_sample"] is False
