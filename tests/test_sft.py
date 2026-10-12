"""SFT example construction: same prompt as Arm A/C inference, loss on the
completion only (Arm C fine-tuning)."""

from polars_llm.bench.schema import Task
from polars_llm.prompts import build_messages
from polars_llm.train.sft import (
    IGNORE_INDEX,
    LORA,
    TRAIN,
    answer_text,
    render_prompt,
    tokenize_example,
    tokenize_split,
    total_steps,
    warmup_steps,
)

TASK = Task(
    id="t1", difficulty="easy", category="select_filter", instruction="Keep rows where `a` > 1.",
    setup_code='inputs = {"df": pl.DataFrame({"a": [1, 2]})}', signature="def solve(df: pl.DataFrame) -> pl.DataFrame:",
    reference_solution='def solve(df: pl.DataFrame) -> pl.DataFrame:\n    return df.filter(pl.col("a") > 1)\n',
)  # fmt: skip


class FakeChatTokenizer:
    """ChatML-style template (like Qwen) + a character-level tokenizer."""

    eos_token = "<|im_end|>"

    def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=False, enable_thinking=None):
        assert enable_thinking is False
        text = "".join(f"<|im_start|>{m['role']}\n{m['content']}<|im_end|>\n" for m in messages)
        if add_generation_prompt:
            text += "<|im_start|>assistant\n<think>\n\n</think>\n\n"
        elif messages[-1]["role"] == "assistant":
            # Render the final assistant turn the way the generation prompt starts it.
            head = text[: text.rindex("<|im_start|>assistant\n")]
            text = head + "<|im_start|>assistant\n<think>\n\n</think>\n\n" + messages[-1]["content"] + "<|im_end|>\n"
        return text

    def __call__(self, text, add_special_tokens=False):
        return {"input_ids": [ord(ch) for ch in text]}


def test_prompt_is_identical_to_inference_prompt():
    tok = FakeChatTokenizer()
    expected = tok.apply_chat_template(build_messages(TASK, context=None), tokenize=False,
                                       add_generation_prompt=True, enable_thinking=False)
    assert render_prompt(tok, TASK) == expected
    assert "documentation" not in expected  # no RAG context block


def test_labels_mask_the_prompt_and_cover_answer_plus_end_of_turn():
    tok = FakeChatTokenizer()
    ex = tokenize_example(tok, TASK)
    n_p = ex["n_prompt_tokens"]
    assert ex["labels"][:n_p] == [IGNORE_INDEX] * n_p
    completion = "".join(chr(i) for i in ex["labels"][n_p:])
    assert completion == answer_text(TASK) + "<|im_end|>"
    assert answer_text(TASK).startswith("```python\ndef solve(") and answer_text(TASK).endswith("\n```")
    assert ex["from_template"] and len(ex["input_ids"]) == len(ex["labels"]) == len(ex["attention_mask"])


def test_template_mismatch_falls_back_to_answer_plus_eos():
    class Odd(FakeChatTokenizer):
        def apply_chat_template(self, messages, **kw):
            if not kw.get("add_generation_prompt") and messages[-1]["role"] == "assistant":
                return "SOMETHING ELSE"
            return super().apply_chat_template(messages, **kw)

    ex = tokenize_example(Odd(), TASK)
    assert not ex["from_template"]
    assert "".join(chr(i) for i in ex["labels"][ex["n_prompt_tokens"]:]) == answer_text(TASK) + "<|im_end|>"


def test_too_long_examples_are_dropped_and_counted():
    rows, stats = tokenize_split(FakeChatTokenizer(), [TASK, TASK], max_seq_length=50)
    assert rows == [] and stats["dropped_too_long"] == 2


def test_hyperparameters_match_plan():
    assert (LORA["r"], LORA["lora_alpha"], LORA["lora_dropout"]) == (16, 16, 0.0)
    assert TRAIN["learning_rate"] == 2e-4 and TRAIN["lr_scheduler_type"] == "cosine"
    assert TRAIN["num_train_epochs"] == 2
    assert TRAIN["per_device_train_batch_size"] * TRAIN["gradient_accumulation_steps"] == 16
    assert total_steps(1600) == 200 and warmup_steps(1600) == 6
