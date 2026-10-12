"""Round-trip verification of training requests: sandbox verdicts, report, mock CLI."""

import json

from polars_llm.bench.executor import compute_expected
from polars_llm.datagen import roundtrip
from polars_llm.datagen.roundtrip import judge, report, solver_messages

EX = {
    "id": "c1__concise", "candidate_id": "c1", "style": "concise", "source": "docstring", "category": "group_by",
    "difficulty": "easy", "multi_step": True, "page": "DataFrame.group_by",
    "instruction": "Total `v` per `k`, columns `k` then `total`.",
    "setup_code": 'df = pl.DataFrame({"k": ["a", "b", "a"], "v": [1, 2, 3]})\ninputs = {"df": df}',
    "signature": "def solve(df: pl.DataFrame) -> pl.DataFrame:",
    "reference_solution": (
        "def solve(df: pl.DataFrame) -> pl.DataFrame:\n"
        '    return df.group_by("k", maintain_order=True).agg(total=pl.col("v").sum())'
    ),
    "check": {"ignore_row_order": False},
}  # fmt: skip
REF = compute_expected(EX["setup_code"], EX["reference_solution"])


def _out(body):
    return f"```python\ndef solve(df):\n    return {body}\n```"


def test_match_ignores_row_and_column_order():
    v = judge(EX, _out('df.group_by("k").agg(total=pl.col("v").sum()).sort("k", descending=True).select("total", "k")'), REF)
    assert v == {"kept": True, "reason": "match", "detail": ""}


def test_mismatches_are_classified():
    assert judge(EX, _out('df.group_by("k").agg(pl.col("v").sum())'), REF)["reason"] == "wrong_columns"
    assert judge(EX, _out('df.group_by("k").agg(total=pl.col("v").mean())'), REF)["reason"] == "wrong_values"
    assert judge(EX, _out('df.select("k", total=pl.col("v"))'), REF)["reason"] == "wrong_row_count"
    assert judge(EX, _out('df.group_by("nope").agg(total=pl.col("v").sum())'), REF)["reason"].startswith("error:")
    assert judge(EX, "Sorry, I can't.", REF)["reason"] == "no_code"
    # Numeric width is ignored, as in the benchmark: Int32 totals still match.
    assert judge(EX, _out('df.group_by("k").agg(total=pl.col("v").sum().cast(pl.Int32))'), REF)["kept"]


def test_list_columns_fall_back_to_row_order():
    ex = {**EX, "setup_code": 'df = pl.DataFrame({"k": ["a", "b"], "v": [[1], [2, 3]]})\ninputs = {"df": df}',
          "reference_solution": "def solve(df: pl.DataFrame) -> pl.DataFrame:\n    return df.select(\"k\", \"v\")"}
    ref = compute_expected(ex["setup_code"], ex["reference_solution"])
    assert judge(ex, _out('df.select("v", "k")'), ref)["kept"]


def test_solver_prompt_is_the_rag_template_with_oracle_context():
    msgs = solver_messages(EX, "[1] polars.DataFrame.group_by\nGroup by ...")
    assert "Here is relevant Polars 2.0 documentation" in msgs[1]["content"]
    assert "[1] polars.DataFrame.group_by" in msgs[1]["content"] and EX["instruction"] in msgs[1]["content"]
    assert "documentation" not in solver_messages(EX, "")[1]["content"]


def test_report_breakdowns_and_examples():
    exs = [{**EX, "id": f"c{i}__{s}", "candidate_id": f"c{i}", "style": s, "multi_step": i % 2 == 0}
           for i in range(8) for s in ("concise", "descriptive")]
    verdicts = {e["id"]: {"kept": e["style"] == "concise" or e["candidate_id"] == "c0", "reason": "x", "detail": ""}
                for e in exs}
    rep = report(exs, verdicts, {e["id"]: "```python\npass\n```" for e in exs})
    assert rep["n"] == 16 and rep["kept"] == 9 and rep["by_style"]["concise"]["retention"] == 1.0
    assert rep["both_styles_of_a_candidate"] == {"candidates_with_both": 8, "both_kept": 1, "one_kept": 7, "none_kept": 0}
    assert len(rep["random_kept"]) == 6 and len(rep["random_rejected"]) == 6
    assert set(rep["by_multi_step"]) == {"True", "False"}


def test_mock_cli_with_resume(tmp_path):
    exs = [{**EX, "id": f"c{i}__concise", "candidate_id": f"c{i}"} for i in range(6)]
    (tmp_path / "f.jsonl").write_text("".join(json.dumps(e) + "\n" for e in exs), encoding="utf-8")
    (tmp_path / "cands.jsonl").write_text(
        "".join(json.dumps({"id": f"c{i}", "doc_context": "[1] doc"}) + "\n" for i in range(6)), encoding="utf-8")
    common = ["--model", "mock", "--examples", str(tmp_path / "f.jsonl"), "--candidates", str(tmp_path / "cands.jsonl"),
              "--checkpoint", str(tmp_path / "ck.jsonl"), "--out", str(tmp_path / "v.jsonl"),
              "--results", str(tmp_path / "r.jsonl"), "--stats", str(tmp_path / "s.json"), "--batch-size", "4",
              "--workers", "2"]
    assert roundtrip.main(common + ["--limit", "4"]) == 0
    assert roundtrip.main(common) == 0
    stats = json.loads((tmp_path / "s.json").read_text(encoding="utf-8"))
    assert stats["this_session"]["n_resumed"] == 4 and stats["n"] == 6 and stats["n_without_doc_context"] == 0
    # The mock drops a column for ids whose hex value is divisible by 3: exactly those are rejected.
    bad = {e["id"] for e in exs if int(e["id"].encode().hex(), 16) % 3 == 0}
    kept = {json.loads(l)["id"] for l in (tmp_path / "v.jsonl").read_text(encoding="utf-8").splitlines()}
    assert kept == {e["id"] for e in exs} - bad and stats["reasons"].get("wrong_columns", 0) == len(bad)
