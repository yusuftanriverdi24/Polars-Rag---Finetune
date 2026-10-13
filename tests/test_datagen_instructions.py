"""Instruction writer parsing + automatic instruction filters (training data v1, PART 2)."""

import json

import pytest

from polars_llm.datagen import instructions as ins
from polars_llm.datagen.instructions import (
    MockWriter,
    initial_batch_size,
    instruction_problems,
    load_checkpoint,
    mentions_method,
    parse_writer_output,
    solution_api_identifiers,
    write_with_checkpoint,
    writer_messages,
)

REC = {
    "id": "x",
    "signature": "def solve(df: pl.DataFrame) -> pl.DataFrame:",
    "setup_code": 'df = pl.DataFrame({"region": ["a"], "amount": [1]})\ninputs = {"df": df}',
    "reference_solution": (
        "def solve(df: pl.DataFrame) -> pl.DataFrame:\n"
        '    return df.group_by("region", maintain_order=True).agg(pl.col("amount").sum().alias("total"))'
    ),
    "output_columns": ["region", "total"],
    "output_dtypes": ["String", "Int64"],
    "preview": "...",
    "n_rows": 1,
}


def test_parse_writer_output():
    good = 'Sure!\n{"concise": "Total  amount per `region`.", "descriptive": "For each region..."}'
    assert parse_writer_output(good) == {"concise": "Total amount per `region`.", "descriptive": "For each region..."}
    assert parse_writer_output("<think>x</think>" + good)["concise"].startswith("Total")
    assert parse_writer_output("no json here") is None
    assert parse_writer_output('{"concise": "x"}') is None  # a style is missing
    assert parse_writer_output('{"concise": "", "descriptive": "y"}') is None


def test_solution_api_identifiers():
    names = solution_api_identifiers(REC["reference_solution"])
    assert {"group_by", "agg", "col", "sum", "alias", "maintain_order"} <= names


def test_good_instruction_passes():
    text = "From `df`, compute the total `amount` for each `region` as `total`; return `region` then `total`."
    assert instruction_problems(text.replace("`amount`", "amount"), "concise", REC) == []


def test_method_names_are_flagged_but_plain_english_is_not():
    methods = solution_api_identifiers(REC["reference_solution"])
    allowed = {"region", "total", "df"}
    assert mentions_method("Use group_by on region and sum it.", methods, allowed) == "group_by"
    assert mentions_method("Keep the original order (maintain_order).", methods, allowed) == "maintain_order"
    assert mentions_method("Return the sum of amount per region as total.", methods, allowed) is None
    assert mentions_method("Call `.sum` on amount.", methods, allowed) == "sum"
    assert mentions_method("Apply sum() to amount.", methods, allowed) == "sum"
    # A method name that is also a column name is fine.
    assert mentions_method("Return the `sum` column.", methods | {"sum"}, allowed | {"sum"}) is None


def test_code_too_short_too_long_and_missing_columns():
    rec = REC
    assert "code" in instruction_problems("Compute df.group_by('region') with total and region columns.", "concise", rec)
    assert "code" in instruction_problems("```python\nx\n```\nregion total stuff here", "concise", rec)
    assert "too_short" in instruction_problems("Totals by `region`.", "concise", rec)
    assert "too_long" in instruction_problems("word " * 70 + "`region` `total`", "concise", rec)
    assert "missing_columns" in instruction_problems("Sum the amounts for every group in the table please.", "concise", rec)


def test_writer_prompt_and_mock_writer():
    msgs = writer_messages(REC)
    assert msgs[0]["role"] == "system" and "`region` (String), `total` (Int64)" in msgs[1]["content"]
    assert "Do not mention any function, method" in msgs[1]["content"]
    pair = parse_writer_output(MockWriter().write([REC])[0])
    assert instruction_problems(pair["concise"], "concise", REC) == []
    assert instruction_problems(pair["descriptive"], "descriptive", REC) == []
    json.dumps(pair)


# ---------------------------------------------------------------------------
# adaptive batching + checkpoint/resume
# ---------------------------------------------------------------------------

SETTINGS = {"writer_model": "m", "writer_revision": "r", "writer_prompt_sha256": "h", "max_new_tokens": 400}
RECS = [{"id": f"c{i}"} for i in range(10)]


class OutOfMemoryError(RuntimeError):  # same class name as torch.OutOfMemoryError
    pass


def _reply(r):
    return json.dumps({"concise": f"concise {r['id']}", "descriptive": f"descriptive {r['id']}"})


def test_initial_batch_size_from_free_memory():
    assert initial_batch_size(3.5) == 1  # L4 after loading the 9B writer: little headroom
    assert initial_batch_size(21.0) == 16  # A100 40 GB
    assert initial_batch_size(60.0) == 32  # capped
    assert initial_batch_size(0.5) == 1


def test_oom_halves_batch_and_retries_same_batch(tmp_path):
    calls, freed = [], []

    def write_batch(batch):
        calls.append([r["id"] for r in batch])
        if len(batch) > 2:
            raise OutOfMemoryError("CUDA out of memory. Tried to allocate 2.00 GiB")
        return [(_reply(r), 0.5) for r in batch]

    ckpt = tmp_path / "ckpt.jsonl"
    stats = write_with_checkpoint(RECS, write_batch, checkpoint=ckpt, settings=SETTINGS, batch_size=8,
                                  free_memory=lambda: freed.append(1), log=lambda _: None)
    assert calls[:3] == [[f"c{i}" for i in range(8)], [f"c{i}" for i in range(4)], ["c0", "c1"]]
    assert [(e["from"], e["to"]) for e in stats["oom_events"]] == [(8, 4), (4, 2)] and len(freed) == 2
    assert stats["final_batch_size"] == 2 and stats["n_written"] == 10
    rows = load_checkpoint(ckpt, SETTINGS)
    assert list(rows) == [r["id"] for r in RECS]  # every candidate once, in order
    assert rows["c3"]["instructions"]["concise"] == "concise c3" and rows["c3"]["batch_size"] == 2


def test_oom_at_batch_size_one_and_other_errors_raise(tmp_path):
    def always_oom(batch):
        raise OutOfMemoryError("CUDA out of memory")

    with pytest.raises(RuntimeError, match="batch size 1"):
        write_with_checkpoint(RECS, always_oom, checkpoint=tmp_path / "a.jsonl", settings=SETTINGS, batch_size=4,
                              log=lambda _: None)

    def broken(batch):
        raise ValueError("not an OOM")

    with pytest.raises(ValueError):
        write_with_checkpoint(RECS, broken, checkpoint=tmp_path / "b.jsonl", settings=SETTINGS, batch_size=4,
                              log=lambda _: None)


def test_resume_skips_written_and_survives_a_truncated_line(tmp_path):
    ckpt = tmp_path / "ckpt.jsonl"
    seen = []

    def crash_after_two_batches(batch):
        if len(seen) == 2:
            raise KeyboardInterrupt  # a disconnect
        seen.append([r["id"] for r in batch])
        return [(_reply(r), 0.1) for r in batch]

    with pytest.raises(KeyboardInterrupt):
        write_with_checkpoint(RECS, crash_after_two_batches, checkpoint=ckpt, settings=SETTINGS, batch_size=3,
                              log=lambda _: None)
    assert len(load_checkpoint(ckpt, SETTINGS)) == 6  # finished batches are on disk
    with ckpt.open("a", encoding="utf-8") as fh:
        fh.write('{"id": "c6", "raw": "half a li')  # killed mid-write

    resumed = []

    def write_batch(batch):
        resumed.extend(r["id"] for r in batch)
        return [(_reply(r), 0.1) for r in batch]

    stats = write_with_checkpoint(RECS, write_batch, checkpoint=ckpt, settings=SETTINGS, batch_size=3,
                                  log=lambda _: None)
    assert resumed == ["c6", "c7", "c8", "c9"] and stats["n_resumed"] == 6
    assert list(load_checkpoint(ckpt, SETTINGS)) == [r["id"] for r in RECS]


def test_checkpoint_with_other_settings_is_refused(tmp_path):
    ckpt = tmp_path / "ckpt.jsonl"
    ins.append_rows(ckpt, [{"id": "c0", "raw": "", "instructions": None, **SETTINGS}])
    with pytest.raises(SystemExit, match="other settings"):
        load_checkpoint(ckpt, {**SETTINGS, "writer_prompt_sha256": "different"})


def _rows(ids, raw=""):
    return [{"id": i, "raw": raw, "instructions": None, **SETTINGS} for i in ids]


SEPARATORS = "a\u2028b\u2029c\x85d"  # str.splitlines() splits on all three; ensure_ascii=False leaves them raw


def test_unicode_line_separators_round_trip_through_the_checkpoint(tmp_path):
    ckpt = tmp_path / "ckpt.jsonl"
    ins.append_rows(ckpt, _rows(["c0"]) + _rows(["c1"], raw=SEPARATORS) + _rows(["c2"]), log=lambda _: None)
    assert "\u2028" in ckpt.read_text(encoding="utf-8")  # still written raw: existing files and hashes unchanged
    before = ckpt.read_bytes()
    rows = load_checkpoint(ckpt, SETTINGS, log=pytest.fail)  # nothing unreadable, nothing logged
    assert list(rows) == ["c0", "c1", "c2"] and rows["c1"]["raw"] == SEPARATORS
    assert ckpt.read_bytes() == before


def test_cut_off_last_line_is_skipped_without_rewriting(tmp_path):
    ckpt = tmp_path / "ckpt.jsonl"
    ins.append_rows(ckpt, _rows(["c0", "c1"]))
    with ckpt.open("a", encoding="utf-8", newline="\n") as fh:
        fh.write('{"id": "c2", "raw": "killed mid-wri')
    before, logs = ckpt.read_bytes(), []
    assert list(load_checkpoint(ckpt, SETTINGS, log=logs.append)) == ["c0", "c1"]
    assert ckpt.read_bytes() == before  # never rewritten on load
    assert "1 unreadable line(s)" in logs[0] and "line 3" in logs[0] and "['c2']" in logs[0]

    # Resuming: the cut-off tail (never a complete row) is cut before appending, so nothing is glued onto it.
    ins.append_rows(ckpt, _rows(["c2", "c3"]), log=logs.append)
    assert list(load_checkpoint(ckpt, SETTINGS, log=pytest.fail)) == ["c0", "c1", "c2", "c3"]


def test_unreadable_middle_line_is_an_error_and_the_file_is_untouched(tmp_path):
    ckpt = tmp_path / "ckpt.jsonl"
    ins.append_rows(ckpt, _rows(["c0"]))
    with ckpt.open("a", encoding="utf-8", newline="\n") as fh:
        fh.write('{"id": "c1", "raw": "half\n"x"}\n')  # what splitlines() used to make of a row
    ins.append_rows(ckpt, _rows(["c2"]))
    before = ckpt.read_bytes()
    with pytest.raises(SystemExit, match=r"2 unreadable line\(s\) .*line 2, 3.*\['c1'\].*left untouched"):
        load_checkpoint(ckpt, SETTINGS)
    assert ckpt.read_bytes() == before


def test_session_rows_are_merged_and_a_missing_id_fails_with_the_ids(tmp_path, monkeypatch):
    real_load = ins.load_checkpoint
    monkeypatch.setattr(ins, "load_checkpoint", lambda path, settings, **kw: {
        k: v for k, v in real_load(path, settings, **kw).items() if k != "c9"})  # the re-read misses c9
    done, run = ins.generate_all(RECS, lambda b: [(_reply(r), 0.1) for r in b], checkpoint=tmp_path / "a.jsonl",
                                 settings=SETTINGS, batch_size=4, log=lambda _: None)
    assert list(done) == [r["id"] for r in RECS] and done["c9"]["instructions"]["concise"] == "concise c9"
    assert run["n_written"] == 10 and "rows" not in run

    def drops_c9(batch):  # e.g. a model wrapper returning fewer outputs than prompts
        return [(_reply(r), 0.1) for r in batch if r["id"] != "c9"]

    with pytest.raises(SystemExit, match=r"1 id\(s\) have no row .*\['c9'\]"):
        ins.generate_all(RECS, drops_c9, checkpoint=tmp_path / "b.jsonl", settings=SETTINGS, batch_size=4,
                         log=lambda _: None)


def test_cli_mock_resume_writes_candidate_order(tmp_path):
    cands = tmp_path / "cands.jsonl"
    rec = {k: v for k, v in REC.items() if k != "id"}
    cands.write_text("".join(json.dumps({**rec, "id": f"k{i}"}) + "\n" for i in range(5)), encoding="utf-8")
    common = ["--candidates", str(cands), "--model", "mock", "--checkpoint", str(tmp_path / "ck.jsonl"),
              "--out", str(tmp_path / "out.jsonl"), "--meta", str(tmp_path / "meta.json"), "--batch-size", "2"]
    assert ins.main(common + ["--limit", "3"]) == 0  # interrupted run: only the first 3
    assert ins.main(common) == 0
    out = [json.loads(line) for line in (tmp_path / "out.jsonl").read_text(encoding="utf-8").splitlines()]
    assert [o["id"] for o in out] == [f"k{i}" for i in range(5)] and all(o["instructions"] for o in out)
    meta = json.loads((tmp_path / "meta.json").read_text(encoding="utf-8"))
    assert meta["this_session"]["n_resumed"] == 3 and meta["n_parsed"] == 5

    # Complete checkpoint: no writer is built, --out is rebuilt identically, the meta is left alone.
    before = (tmp_path / "out.jsonl").read_bytes(), (tmp_path / "meta.json").read_bytes()
    (tmp_path / "out.jsonl").unlink()

    def no_model(*a, **k):
        raise AssertionError("the writer must not be loaded")

    ins.MockWriter, saved = no_model, ins.MockWriter
    try:
        assert ins.main(common) == 0
    finally:
        ins.MockWriter = saved
    assert ((tmp_path / "out.jsonl").read_bytes(), (tmp_path / "meta.json").read_bytes()) == before
