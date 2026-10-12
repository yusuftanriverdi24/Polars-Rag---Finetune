"""Round-trip verification of training requests: sandbox verdicts, canaries, report, mock CLI."""

import json

from polars_llm.bench.executor import compute_expected
from polars_llm.datagen import roundtrip
from polars_llm.datagen.roundtrip import canary_report, judge, load_canaries, report, solver_messages

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
    "check": {"ignore_row_order": False, "ignore_column_order": False, "ignore_numeric_width": True},
}  # fmt: skip
REF = compute_expected(EX["setup_code"], EX["reference_solution"])
GOOD = 'df.group_by("k", maintain_order=True).agg(total=pl.col("v").sum())'


def _out(body):
    return f"```python\ndef solve(df):\n    return {body}\n```"


def test_uses_the_candidates_own_check_flags():
    assert judge(EX, _out(GOOD), REF) == {"kept": True, "reason": "match", "detail": ""}
    # Row and column order are part of the request unless the candidate's flags say otherwise.
    assert judge(EX, _out(GOOD + '.sort("k", descending=True)'), REF)["reason"] == "wrong_row_order"
    assert judge(EX, _out(GOOD + '.select("total", "k")'), REF)["reason"] == "wrong_column_order"
    unordered = {**EX, "check": {**EX["check"], "ignore_row_order": True}}
    assert judge(unordered, _out(GOOD + '.sort("k", descending=True)'), REF)["kept"]
    any_cols = {**EX, "check": {**EX["check"], "ignore_column_order": True}}
    assert judge(any_cols, _out(GOOD + '.select("total", "k")'), REF)["kept"]


def test_mismatches_are_classified():
    assert judge(EX, _out('df.group_by("k").agg(pl.col("v").sum())'), REF)["reason"] == "wrong_columns"
    assert judge(EX, _out('df.group_by("k", maintain_order=True).agg(total=pl.col("v").mean())'), REF)["reason"] == "wrong_values"
    assert judge(EX, _out('df.select("k", total=pl.col("v"))'), REF)["reason"] == "wrong_row_count"
    assert judge(EX, _out('df.group_by("nope").agg(total=pl.col("v").sum())'), REF)["reason"].startswith("error:")
    assert judge(EX, "Sorry, I can't.", REF)["reason"] == "no_code"
    # Numeric width is ignored, as in the benchmark: Int32 totals still match.
    assert judge(EX, _out(GOOD.replace(".sum()", ".sum().cast(pl.Int32)")), REF)["kept"]


def test_unsortable_columns_fall_back_to_row_order():
    ex = {**EX, "check": {**EX["check"], "ignore_row_order": True},
          "setup_code": 'df = pl.DataFrame({"k": ["a", "b"], "v": [[1], [2, 3]]})\ninputs = {"df": df}',
          "reference_solution": "def solve(df: pl.DataFrame) -> pl.DataFrame:\n    return df.select(\"k\", \"v\")"}
    ref = compute_expected(ex["setup_code"], ex["reference_solution"])
    assert judge(ex, _out('df.select("k", "v")'), ref)["kept"]


def test_solver_prompt_is_the_rag_template_with_oracle_context():
    msgs = solver_messages(EX, "[1] polars.DataFrame.group_by\nGroup by ...")
    assert "Here is relevant Polars 2.0 documentation" in msgs[1]["content"]
    assert "[1] polars.DataFrame.group_by" in msgs[1]["content"] and EX["instruction"] in msgs[1]["content"]
    assert "documentation" not in solver_messages(EX, "")[1]["content"]


def test_report_breakdowns_audit_and_decision_rules():
    exs = [{**EX, "id": f"c{i}__{s}", "candidate_id": f"c{i}", "style": s, "multi_step": i % 2 == 0,
            "category": "reshape" if i < 4 else "string"} for i in range(30) for s in ("concise", "descriptive")]
    verdicts = {e["id"]: {"kept": e["style"] == "concise" and e["category"] == "string",
                          "reason": "match" if e["style"] == "concise" and e["category"] == "string" else "error:X",
                          "detail": ""} for e in exs}
    rep = report(exs, verdicts, {e["id"]: "```python\npass\n```" for e in exs})
    assert rep["n"] == 60 and rep["kept"] == 26 and rep["by_style"]["descriptive"]["retention"] == 0.0
    assert rep["both_styles_of_a_candidate"] == {"candidates_with_both": 30, "both_kept": 0, "one_kept": 26, "none_kept": 4}
    assert len(rep["audit_kept"]) == 20 and len(rep["audit_rejected"]) == 20
    assert rep["audit_kept"][0]["audit_label"] is None
    rules = rep["decision_rules"]
    assert rules["reshape_retention"] == 0.0 and rules["reshape_below_half_overall"]
    assert rules["error_share_of_rejections"] == 1.0 and rules["retry_with_error_message"]  # 26 kept < 1,000


def test_canaries_are_pinned_and_reported():
    cand = {k: EX[k] for k in ("difficulty", "category", "setup_code", "signature", "reference_solution", "check",
                               "source", "page", "multi_step")}
    cand.update(id="c1", api_names=[])
    canaries = [{"id": "canary:c1__concise", "candidate_id": "c1", "style": "concise", "instruction": "Wrong text.",
                 "label": "wrong", "expected": "reject", "note": ""}]

    import tempfile
    from pathlib import Path

    with tempfile.TemporaryDirectory() as d:
        p = Path(d) / "canaries.json"
        p.write_text(json.dumps(canaries), encoding="utf-8")
        (c,) = load_canaries(p, {"c1": cand})
    assert c["id"] == "canary:c1__concise" and c["instruction"] == "Wrong text." and c["canary"]["expected"] == "reject"
    rep = canary_report([c], {c["id"]: {"kept": True, "reason": "match"}})
    assert rep["passed"] == 0 and not rep["all_passed"] and rep["results"][0]["actual"] == "keep"


def test_mock_cli_with_resume_and_canaries(tmp_path):
    exs = [{**EX, "id": f"c{i}__concise", "candidate_id": f"c{i}"} for i in range(6)]
    (tmp_path / "f.jsonl").write_text("".join(json.dumps(e) + "\n" for e in exs), encoding="utf-8")
    cand = {k: EX[k] for k in ("difficulty", "category", "setup_code", "signature", "reference_solution", "check",
                               "source", "page", "multi_step")}
    (tmp_path / "cands.jsonl").write_text(
        "".join(json.dumps({**cand, "id": f"c{i}", "api_names": [], "doc_context": "[1] doc"}) + "\n" for i in range(6)),
        encoding="utf-8")
    (tmp_path / "canaries.json").write_text(json.dumps([
        {"id": "canary:c0__descriptive", "candidate_id": "c0", "style": "descriptive", "instruction": "x",
         "label": "correct", "expected": "keep", "note": ""}]), encoding="utf-8")
    common = ["--model", "mock", "--examples", str(tmp_path / "f.jsonl"), "--candidates", str(tmp_path / "cands.jsonl"),
              "--canaries", str(tmp_path / "canaries.json"), "--checkpoint", str(tmp_path / "ck.jsonl"),
              "--out", str(tmp_path / "v.jsonl"), "--results", str(tmp_path / "r.jsonl"),
              "--stats", str(tmp_path / "s.json"), "--batch-size", "4", "--workers", "2"]
    assert roundtrip.main(common + ["--limit", "4"]) == 0
    assert roundtrip.main(common) == 0
    stats = json.loads((tmp_path / "s.json").read_text(encoding="utf-8"))
    assert stats["this_session"]["n_resumed"] == 5 and stats["n"] == 6 and stats["n_without_doc_context"] == 0
    assert stats["answer_in_context"]["with_answer_in_context"] == 0 and stats["canaries"]["n"] == 1
    # The mock drops a column for ids whose hex value is divisible by 3: exactly those are rejected.
    bad = {e["id"] for e in exs if int(e["id"].encode().hex(), 16) % 3 == 0}
    kept = {json.loads(l)["id"] for l in (tmp_path / "v.jsonl").read_text(encoding="utf-8").splitlines()}
    assert kept == {e["id"] for e in exs} - bad and stats["reasons"].get("wrong_columns", 0) == len(bad)
    assert not any(k.startswith("canary:") for k in kept)  # canaries never enter the data


def test_answer_in_context_aborts(tmp_path):
    (tmp_path / "f.jsonl").write_text(json.dumps(EX) + "\n", encoding="utf-8")
    leaky = {**{k: EX[k] for k in ("reference_solution",)}, "id": "c1", "doc_context": ">>> " + GOOD}
    (tmp_path / "cands.jsonl").write_text(json.dumps(leaky) + "\n", encoding="utf-8")
    import pytest

    with pytest.raises(SystemExit, match="answer is in the oracle context"):
        roundtrip.main(["--model", "mock", "--examples", str(tmp_path / "f.jsonl"), "--candidates",
                        str(tmp_path / "cands.jsonl"), "--canaries", str(tmp_path / "none.json"),
                        "--checkpoint", str(tmp_path / "ck.jsonl")])
