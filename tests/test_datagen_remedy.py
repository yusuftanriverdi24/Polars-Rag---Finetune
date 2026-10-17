"""Post-audit remedy: drop api_mismatch kept requests, rebuild verified.jsonl, second audit."""

import json

from polars_llm.datagen import remedy
from polars_llm.io import read_jsonl, write_jsonl


def _data(n=40):
    exs = [{"id": f"c{i}__concise", "candidate_id": f"c{i}", "category": "group_by" if i % 2 else "string",
            "multi_step": i % 5 == 0, "instruction": f"request {i}", "reference_solution": "def solve(df): ..."}
           for i in range(n)]  # fmt: skip
    # c0-c9 rejected; of the kept ones every third is an api_mismatch; plus a canary row
    res = [{"id": e["id"], "kept": i >= 10, "reason": "match" if i >= 10 else "wrong_values", "detail": "",
            "solver_code": "def solve(df): ...", "api_mismatch": (i % 3 == 0) if i >= 10 else None}
           for i, e in enumerate(exs)]  # fmt: skip
    res.append({"id": "canary:x", "kept": True, "reason": "match", "detail": "", "solver_code": "", "api_mismatch": True})
    return exs, res


def test_drops_api_mismatch_and_reports_composition():
    exs, res = _data()
    out, rep = remedy.remedy(exs, res)
    kept_ids = {r["id"] for r in res if r["kept"] and not r["id"].startswith("canary")}
    mm_ids = {r["id"] for r in res if r["api_mismatch"] and not r["id"].startswith("canary")}
    assert {e["id"] for e in out} == kept_ids - mm_ids
    assert rep["n_dropped"] == len(mm_ids) == 10
    assert rep["before"]["n"] == 30 and rep["after"]["n"] == 20
    assert sum(rep["dropped_by_category"].values()) == 10
    assert [e["id"] for e in out] == [e["id"] for e in exs if e["id"] in kept_ids - mm_ids]  # input order kept


def test_second_audit_is_unstratified_seed_1_from_the_remedied_set():
    exs, res = _data(200)
    out, rep = remedy.remedy(exs, res)
    audit = rep["audit_kept"]
    assert len(audit) == 20 and rep["audit"]["seed"] == 1 and rep["audit"]["stratified"] is False
    assert not any(a["api_mismatch"] for a in audit) and {a["id"] for a in audit} <= {e["id"] for e in out}
    assert all(a["audit_label"] is None for a in audit)
    assert remedy.remedy(exs, res)[1]["audit_kept"] == audit  # deterministic


def test_cli_is_idempotent(tmp_path):
    exs, res = _data()
    write_jsonl(tmp_path / "filtered.jsonl", exs)
    write_jsonl(tmp_path / "results.jsonl", res)
    args = ["--examples", str(tmp_path / "filtered.jsonl"), "--results", str(tmp_path / "results.jsonl"),
            "--out", str(tmp_path / "verified.jsonl"), "--stats", str(tmp_path / "stats.json")]
    remedy.main(args)
    first = (tmp_path / "verified.jsonl").read_bytes(), json.loads((tmp_path / "stats.json").read_text())
    remedy.main(args)
    assert (tmp_path / "verified.jsonl").read_bytes() == first[0]
    assert json.loads((tmp_path / "stats.json").read_text()) == first[1]
    assert len(read_jsonl(tmp_path / "verified.jsonl")) == 20
