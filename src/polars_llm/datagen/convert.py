"""Turn Polars doc examples into benchmark-format task candidates (data v1).

A *program* is an ordered list of statements from one doc page: the doctest
examples of one API docstring, or the top-level statements of one user-guide
snippet file. A *candidate* is made for every statement that displays a value
(a doctest line with expected output, or a ``print(x)`` call):

1. **Backward slice.** Starting from the displayed expression ``E``, walk the
   earlier statements and keep those that define (or mutate) a name ``E``
   still needs, transitively. Names that nothing defines (e.g. a frame read
   from a file in a skipped statement) make the candidate unusable.
2. **Split setup / solution.** Statements that construct input data
   (``pl.DataFrame(...)``, ``pl.LazyFrame(...)``, ``pl.Series(...)``,
   ``pl.from_dict(s)(...)``, optionally followed by ``.lazy()``) go to
   ``setup_code``, together with any plain values only they use. Every other
   kept statement is a solution step. The constructed variables the solution
   uses become ``solve``'s parameters and the ``inputs`` dict keys, in
   construction order.
3. **Build the solution**: imports it needs, ``def solve(<params>):`` with the
   steps, then ``return E``. ``validate.py`` later appends ``.collect()`` when
   ``E`` turns out to be a LazyFrame and drops non-DataFrame results.

Anything touching IO, config, plotting, SQL, testing, exceptions, the network,
randomness or an optional dependency (numpy, pandas, pyarrow, ...) is skipped,
as are doctest lines marked ``+SKIP``. The conversion is purely static; the
sandbox (``validate.py``) is the judge of whether a candidate actually runs.
"""

from __future__ import annotations

import ast
import builtins
import doctest
import hashlib
import re
import textwrap
from dataclasses import asdict, dataclass, field

BUILTIN_NAMES = frozenset(dir(builtins))
PRELUDE_NAMES = frozenset({"pl"})  # `import polars as pl` is always available

CONSTRUCTORS = frozenset({"DataFrame", "LazyFrame", "Series", "from_dict", "from_dicts", "from_records"})
ALLOWED_IMPORT_MODULES = frozenset(
    {"polars", "polars.selectors", "datetime", "math", "decimal", "zoneinfo", "functools", "operator", "string"}
)

# Source patterns that disqualify a candidate (checked on every kept statement).
SKIP_PATTERNS: dict[str, re.Pattern] = {
    "io": re.compile(
        r"\b(read|scan|write|sink)_\w+|\bfrom_(pandas|numpy|arrow|torch|repr)\b|\bto_(pandas|numpy|arrow|torch|jax|init_repr)\b"
        r"|\bopen\(|\bPath\(|\btempfile\b|\bio\.|\bBytesIO\b|\bStringIO\b|\bserialize\b|\bdeserialize\b"
        r"|\bhf://|\bs3://|\bhttps?://|\burl\b|\bstorage_options\b|\bcloud\b|\bdatabase\b|\bexcel\b"
    ),
    "config": re.compile(r"\bConfig\b|\bthread_pool|\bstring_cache\b|\bStringCache\b|\bset_random_seed\b"),
    "plotting": re.compile(r"\.plot\b|\bhvplot\b|\baltair\b|\bmatplotlib\b|\bseaborn\b|\bplotly\b|\.style\b"),
    "sql": re.compile(r"\bsql\b|\bSQLContext\b", re.IGNORECASE),
    "testing": re.compile(r"\btesting\b|\bassert_\w+|\bassert\b|\bpytest\b"),
    "exceptions": re.compile(r"\braise\b|\btry\b|\bexcept\b|Error\b|Warning\b"),
    "optional_dependency": re.compile(
        r"\bnp\.|\bnumpy\b|\bpd\.|\bpandas\b|\bpa\.|\bpyarrow\b|\btorch\b|\bjax\b|\bgreat_tables\b|\bxlsx"
    ),
    "engine_or_plugin": re.compile(
        r"\bgpu\b|\bGPUEngine\b|\bplugin\w*\b|\bregister_\w+|\bshow_graph\b|\bexplain\b|\bprofile\b|\bengine=|\bstreaming\b"
    ),
    "random": re.compile(r"\bsample\b|\bshuffle\b|\brandom\b|\bseed\b"),
    "system": re.compile(r"\bos\.|\bsys\.|\bsubprocess\b|\btime\.sleep\b|\bthreading\b|\bmultiprocessing\b"),
}


@dataclass
class Stmt:
    code: str
    displays: bool = False  # doctest example with expected output
    skip: bool = False  # doctest +SKIP


@dataclass
class Program:
    source: str  # "docstring" | "user_guide"
    page: str  # API object name or user-guide file path (the split group)
    statements: list[Stmt]
    labels: list[str] = field(default_factory=list)  # per-statement section label (user guide)


@dataclass
class Candidate:
    id: str
    source: str
    page: str
    label: str
    setup_code: str
    signature: str
    solution: str  # ``return E`` form; validate.py may add .collect()
    result_expr: str
    params: list[str]
    param_types: list[str]
    solution_imports: list[str] = field(default_factory=list)
    steps: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return asdict(self)


# ---------------------------------------------------------------------------
# name analysis
# ---------------------------------------------------------------------------


class _Loads(ast.NodeVisitor):
    """Names an expression/statement reads, minus names bound inside it by
    lambdas and comprehensions."""

    def __init__(self):
        self.loads: set[str] = set()
        self.bound: set[str] = set()

    def visit_Name(self, node):
        if isinstance(node.ctx, ast.Load):
            self.loads.add(node.id)

    def visit_Lambda(self, node):
        for a in node.args.args + node.args.kwonlyargs + node.args.posonlyargs:
            self.bound.add(a.arg)
        self.generic_visit(node)

    def _comp(self, node):
        for gen in node.generators:
            for n in ast.walk(gen.target):
                if isinstance(n, ast.Name):
                    self.bound.add(n.id)
        self.generic_visit(node)

    visit_ListComp = visit_SetComp = visit_DictComp = visit_GeneratorExp = _comp

    def visit_FunctionDef(self, node):
        # A helper function: its parameters are local to it.
        for a in node.args.args + node.args.kwonlyargs + node.args.posonlyargs:
            self.bound.add(a.arg)
        for d in node.decorator_list:
            self.visit(d)
        for stmt in node.body:
            self.visit(stmt)
        # Names assigned in its body are local too.
        for n in ast.walk(node):
            if isinstance(n, ast.Name) and isinstance(n.ctx, ast.Store):
                self.bound.add(n.id)


def loaded_names(node: ast.AST) -> set[str]:
    v = _Loads()
    v.visit(node)
    return v.loads - v.bound


def stored_names(stmt: ast.stmt) -> set[str]:
    if isinstance(stmt, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
        return {stmt.name}
    if isinstance(stmt, (ast.Import, ast.ImportFrom)):
        return {(a.asname or a.name).split(".")[0] for a in stmt.names}
    out: set[str] = set()
    targets: list[ast.AST] = []
    if isinstance(stmt, ast.Assign):
        targets = stmt.targets
    elif isinstance(stmt, (ast.AugAssign, ast.AnnAssign)):
        targets = [stmt.target]
    elif isinstance(stmt, (ast.For, ast.With)):
        targets = [stmt.target] if isinstance(stmt, ast.For) else [i.optional_vars for i in stmt.items if i.optional_vars]
    for t in targets:
        for n in ast.walk(t):
            if isinstance(n, ast.Name) and isinstance(n.ctx, ast.Store):
                out.add(n.id)
    return out


def mutated_names(stmt: ast.stmt) -> set[str]:
    """Names mutated in place: ``df.columns = ...``, ``df["a"] = ...``, ``x += 1``."""

    out: set[str] = set()
    targets = stmt.targets if isinstance(stmt, ast.Assign) else [stmt.target] if isinstance(stmt, ast.AugAssign) else []
    for t in targets:
        if isinstance(stmt, ast.AugAssign) and isinstance(t, ast.Name):
            out.add(t.id)
        base = t
        while isinstance(base, (ast.Attribute, ast.Subscript)):
            base = base.value
        if base is not t and isinstance(base, ast.Name):
            out.add(base.id)
    return out


def _call_root(node: ast.AST) -> ast.AST:
    while isinstance(node, ast.Call) or isinstance(node, ast.Attribute):
        node = node.func if isinstance(node, ast.Call) else node.value
    return node


def construction_type(stmt: ast.stmt) -> str | None:
    """'DataFrame' / 'LazyFrame' / 'Series' when ``stmt`` is ``x = pl.<ctor>(...)[.lazy()]``."""

    if not (isinstance(stmt, ast.Assign) and len(stmt.targets) == 1 and isinstance(stmt.targets[0], ast.Name)):
        return None
    value = stmt.value
    lazy = False
    if (
        isinstance(value, ast.Call)
        and isinstance(value.func, ast.Attribute)
        and value.func.attr == "lazy"
        and not value.args
        and not value.keywords
    ):
        lazy, value = True, value.func.value
    if not (isinstance(value, ast.Call) and isinstance(value.func, ast.Attribute)):
        return None
    f = value.func
    if not (isinstance(f.value, ast.Name) and f.value.id == "pl" and f.attr in CONSTRUCTORS):
        return None
    if lazy:
        return "LazyFrame"
    return {"DataFrame": "DataFrame", "LazyFrame": "LazyFrame", "Series": "Series"}.get(f.attr, "DataFrame")


def import_problem(stmt: ast.stmt) -> str | None:
    if isinstance(stmt, ast.Import):
        mods = [a.name for a in stmt.names]
    elif isinstance(stmt, ast.ImportFrom):
        mods = [stmt.module or ""]
    else:
        return None
    for m in mods:
        if not any(m == ok or m.startswith(ok + ".") for ok in ALLOWED_IMPORT_MODULES):
            return "optional_dependency"
    return None


def skip_reason(code: str) -> str | None:
    for reason, pat in SKIP_PATTERNS.items():
        if pat.search(code):
            return reason
    return None


# ---------------------------------------------------------------------------
# candidate construction
# ---------------------------------------------------------------------------


def _displayed_expr(stmt: ast.stmt, displays: bool) -> ast.expr | None:
    if not isinstance(stmt, ast.Expr):
        return None
    v = stmt.value
    if isinstance(v, ast.Call) and isinstance(v.func, ast.Name) and v.func.id == "print":
        return v.args[0] if len(v.args) == 1 and not v.keywords else None
    return v if displays else None


def _segment(code: str, node: ast.AST) -> str:
    """Source of ``node`` inside ``code``, with its own relative indentation kept.
    (AST column offsets count UTF-8 bytes, so slicing is done on bytes.)"""

    lines = [l.encode("utf-8") for l in code.splitlines()[node.lineno - 1 : node.end_lineno]]
    lines[-1] = lines[-1][: node.end_col_offset]
    lines[0] = b" " * node.col_offset + lines[0][node.col_offset :]
    return textwrap.dedent("\n".join(l.decode("utf-8") for l in lines))


def _replace_nodes(code: str, replacements: list[tuple[ast.AST, str]]) -> str:
    """``code`` with each node's source span replaced by the given text."""

    lines = [l.encode("utf-8") for l in code.splitlines(keepends=True)]
    for node, text in sorted(replacements, key=lambda r: (r[0].lineno, r[0].col_offset), reverse=True):
        first, last = node.lineno - 1, node.end_lineno - 1
        lines[first : last + 1] = [lines[first][: node.col_offset] + text.encode("utf-8") + lines[last][node.end_col_offset :]]
    return b"".join(lines).decode("utf-8")


FRAME_CONSTRUCTORS = frozenset({"DataFrame", "LazyFrame", "from_dict", "from_dicts", "from_records"})


def frame_literal_type(node: ast.AST) -> str | None:
    """'DataFrame' / 'LazyFrame' when ``node`` is ``pl.<frame ctor>(...)``, optionally ``.lazy()``."""

    lazy = False
    if (isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and node.func.attr == "lazy"
            and not node.args and not node.keywords):
        lazy, node = True, node.func.value
    if not (isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)):
        return None
    f = node.func
    if not (isinstance(f.value, ast.Name) and f.value.id == "pl" and f.attr in FRAME_CONSTRUCTORS):
        return None
    return "LazyFrame" if lazy or f.attr == "LazyFrame" else "DataFrame"


class _FrameLiterals(ast.NodeVisitor):
    """Outermost frame literals in a statement; ``nested`` is set when one sits
    inside a lambda, function, class, loop or comprehension (not liftable: it
    would be built once instead of per call/iteration)."""

    def __init__(self):
        self.found: list[tuple[ast.Call, str]] = []
        self.nested = False
        self._depth = 0

    def visit_Call(self, node):
        typ = frame_literal_type(node)
        if typ:
            if self._depth:
                self.nested = True
            self.found.append((node, typ))
            return  # the literal's own arguments belong to it
        self.generic_visit(node)

    def _scope(self, node):
        self._depth += 1
        self.generic_visit(node)
        self._depth -= 1

    visit_Lambda = visit_FunctionDef = visit_AsyncFunctionDef = visit_ClassDef = _scope
    visit_For = visit_While = visit_ListComp = visit_SetComp = visit_DictComp = visit_GeneratorExp = _scope


def frame_literals(node: ast.AST) -> tuple[list[tuple[ast.Call, str]], bool]:
    v = _FrameLiterals()
    v.visit(node)
    return v.found, v.nested


def _is_receiver_root(value: ast.AST, lit: ast.AST) -> bool:
    """``lit`` is where the method chain ``value`` starts (``lit.a(...).b(...)``)."""

    cur = value
    while cur is not lit and isinstance(cur, (ast.Call, ast.Attribute)):
        cur = cur.func if isinstance(cur, ast.Call) else cur.value
    return cur is lit


def return_statement(expr_src: str, collect: bool = False) -> str:
    """``return E`` (``E.collect()`` if needed); multi-line E is parenthesised."""

    suffix = ".collect()" if collect else ""
    if "\n" not in expr_src:
        simple = re.fullmatch(r"[\w.\[\]\"'(), =:+\-*/<>!&|~]*", expr_src) is not None
        try:
            node = ast.parse(expr_src, mode="eval").body
            atomic = isinstance(node, (ast.Name, ast.Call, ast.Attribute, ast.Subscript))
        except SyntaxError:
            atomic = simple
        return f"return {expr_src}{suffix}" if (atomic or not collect) else f"return ({expr_src}){suffix}"
    return "return (\n" + textwrap.indent(expr_src, "    ") + "\n)" + suffix


def build_solution(signature: str, imports: list[str], steps: list[str], expr_src: str, collect: bool = False) -> str:
    body = "\n".join(steps + [return_statement(expr_src, collect)])
    head = "\n".join(imports) + "\n\n" if imports else ""
    return f"{head}{signature}\n" + textwrap.indent(body, "    ")


def candidate_id(page: str, label: str, setup: str, solution: str) -> str:
    h = hashlib.sha256(f"{setup}\n\x1e\n{solution}".encode()).hexdigest()[:10]
    stem = re.sub(r"[^A-Za-z0-9]+", "_", page).strip("_")[-60:]
    return f"{stem}__{h}"


def program_candidates(program: Program) -> tuple[list[Candidate], dict[str, int]]:
    """All candidates of a program, plus counts of why others were rejected."""

    parsed: list[ast.stmt | None] = []
    for st in program.statements:
        try:
            mod = ast.parse(st.code)
            parsed.append(mod.body[0] if len(mod.body) == 1 else None)
        except SyntaxError:
            parsed.append(None)

    rejects: dict[str, int] = {}

    def reject(reason: str) -> None:
        rejects[reason] = rejects.get(reason, 0) + 1

    out: list[Candidate] = []
    for i, (st, node) in enumerate(zip(program.statements, parsed)):
        if node is None:
            continue
        expr = _displayed_expr(node, st.displays)
        if expr is None:
            continue
        expr_src = _segment(st.code, expr)
        cand, reason = _make_candidate(program, parsed, i, expr, expr_src)
        if cand is None:
            reject(reason)
        else:
            out.append(cand)
    return out, rejects


def _make_candidate(program, parsed, i, expr, expr_src) -> tuple[Candidate | None, str]:
    stmts = program.statements
    if stmts[i].skip:
        return None, "doctest_skip"
    if (r := skip_reason(stmts[i].code)) is not None:
        return None, r

    needed = loaded_names(expr)
    picked: list[int] = []
    for j in range(i - 1, -1, -1):
        node = parsed[j]
        if node is None:
            continue
        defs, muts = stored_names(node), mutated_names(node)
        if (defs | muts) & needed:
            picked.append(j)
            needed -= defs - muts
            needed |= loaded_names(node) | muts
    unresolved = needed - BUILTIN_NAMES - PRELUDE_NAMES
    if unresolved:
        return None, "unresolved_name"
    picked.sort()

    for j in picked:
        if stmts[j].skip:
            return None, "doctest_skip"
        if (r := skip_reason(stmts[j].code)) is not None:
            return None, r
        if (r := import_problem(parsed[j])) is not None:
            return None, r

    ctor = {j: construction_type(parsed[j]) for j in picked}
    imports = [j for j in picked if isinstance(parsed[j], (ast.Import, ast.ImportFrom))]
    # Plain values used only by constructions belong to the setup as well.
    setup_js = {j for j in picked if ctor[j]}
    users: dict[int, set] = {}
    for j in picked:
        for k in picked + [None]:
            node = expr if k is None else parsed[k]
            if k is not None and k <= j:
                continue
            if stored_names(parsed[j]) & loaded_names(node):
                users.setdefault(j, set()).add(k)
    for j in sorted(picked, reverse=True):
        if j in setup_js or j in imports:
            continue
        if users.get(j) and all(u in setup_js for u in users[j]):
            setup_js.add(j)
    step_js = [j for j in picked if j not in setup_js and j not in imports]
    # A plain value used by both the setup and the solution is defined in both.
    shared_js = {j for j in step_js if users.get(j, set()) & setup_js}

    lifted = _lift_frame_literals(stmts, parsed, picked, ctor, imports, setup_js | shared_js, step_js, i)
    if lifted is None:
        return None, "frame_literal_not_liftable"
    step_code, expr_src, lifts, copied_js = lifted
    if lifts:
        # A step that only fed the lifted literals now lives in the setup only.
        step_js = _drop_unused_steps(step_js, step_code, parsed, expr_src, copied_js)
    step_nodes = [ast.parse(step_code[j]) for j in step_js]
    expr_node = _parse_expr(expr_src)

    # Inputs: constructed variables the solution (steps + E) reads, then the lifted literals.
    sol_loads = loaded_names(expr_node)
    for n in step_nodes:
        sol_loads |= loaded_names(n)
        for s in n.body:
            sol_loads |= mutated_names(s)
    param_js = [j for j in picked if ctor[j] and (stored_names(parsed[j]) & sol_loads)]
    if not param_js and not lifts:
        return None, "no_input_frame"
    names = [next(iter(stored_names(parsed[j]))) for j in param_js] + [n for n, _, _ in lifts]
    types = [ctor[j] for j in param_js] + [t for _, t, _ in lifts]

    # setup: imports used by setup + setup statements + lifted literals + inputs dict.
    def used_imports(nodes) -> list[str]:
        loads: set[str] = set()
        for n in nodes:
            loads |= loaded_names(n)
        return [stmts[j].code.strip() for j in imports if stored_names(parsed[j]) & loads]

    setup_order = sorted(setup_js | shared_js | copied_js)
    lift_lines = [f"{n} = {src}" for n, _, src in lifts]
    setup_nodes = [parsed[j] for j in setup_order] + [ast.parse(l) for l in lift_lines]
    setup_lines = (used_imports(setup_nodes) + [textwrap.dedent(stmts[j].code).strip() for j in setup_order]
                   + lift_lines)
    inputs = "inputs = {" + ", ".join(f'"{n}": {n}' for n in names) + "}"
    setup_code = "\n".join(setup_lines + [inputs])

    sol_imports = [s for s in used_imports(step_nodes + [expr_node]) if not s.startswith("import polars as pl")]
    signature = "def solve(" + ", ".join(f"{n}: pl.{t}" for n, t in zip(names, types)) + ") -> pl.DataFrame:"
    steps = [step_code[j] for j in step_js]
    solution = build_solution(signature, sol_imports, steps, expr_src)
    label = program.labels[i] if program.labels else ""
    return (
        Candidate(
            id=candidate_id(program.page, label, setup_code, solution),
            source=program.source,
            page=program.page,
            label=label,
            setup_code=setup_code,
            signature=signature,
            solution=solution,
            result_expr=expr_src,
            params=names,
            param_types=types,
            solution_imports=sol_imports,
            steps=steps,
        ),
        "",
    )


def _lift_frame_literals(stmts, parsed, picked, ctor, imports, setup_set, step_js, i):
    """Lift every DataFrame/LazyFrame literal in the solution (steps + displayed
    statement ``i``) into the setup, as a new input.

    Each literal is replaced by a parameter name: the assignment target when the
    literal starts the assigned method chain (``p = pl.LazyFrame({...}).sort("d")``
    -> setup ``p = pl.LazyFrame({...})``, step ``p = p.sort("d")``), otherwise a
    fresh ``other``/``other2``... The names a literal reads must be available in
    the setup: imports, setup statements, or plain-value steps that are then
    copied to the setup (recursively). A literal reading an input or a computed
    value, or one inside a lambda/function/loop/comprehension, is not liftable.

    Returns (step code by index, displayed expression source, [(name, type,
    literal source)], step indices copied to the setup), or None if not liftable.
    """

    taken: set[str] = set()
    for node in parsed:
        if node is not None:
            taken |= {n.id for n in ast.walk(node) if isinstance(n, ast.Name)}
    defined_at = {j: stored_names(parsed[j]) for j in picked}

    def definer(name: str, before: int) -> int | None:
        js = [j for j in picked if j < before and name in defined_at[j]]
        return max(js) if js else None

    ctor_names = {n for j in picked if ctor[j] for n in defined_at[j]}
    copied: set[int] = set()

    def resolvable(names: set[str], before: int) -> bool:
        for n in names - BUILTIN_NAMES - PRELUDE_NAMES:
            j = definer(n, before)
            if j is None:
                return False
            if ctor[j]:
                return False  # built from an input frame: not a literal
            if j in imports or j in setup_set or j in copied:
                continue
            if j not in step_js:
                return False
            node = parsed[j]
            if not isinstance(node, ast.Assign) or frame_literals(node)[0] or mutated_names(node):
                return False
            copied.add(j)
            if not resolvable(loaded_names(node), j):
                return False
        return True

    lifts: list[tuple[str, str, str]] = []
    used: set[str] = set()

    def fresh(preferred: str) -> str:
        name, k = preferred, 2
        while name in used or name in taken:
            name, k = f"{preferred}{k}", k + 1
        used.add(name)
        return name

    step_code: dict[int, str] = {}
    new_expr_src = None
    for j in list(step_js) + [i]:
        code = stmts[j].code
        found, nested = frame_literals(parsed[j])
        if nested:
            return None
        repl = []
        for lit, typ in found:
            if not resolvable(loaded_names(lit), j):
                return None
            node = parsed[j]
            root_target = (
                isinstance(node, ast.Assign) and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name)
                and node.value is not lit and _is_receiver_root(node.value, lit)
            )
            if root_target and node.targets[0].id not in used | ctor_names:
                name = node.targets[0].id
                used.add(name)
            else:
                name = fresh("other")
            lifts.append((name, typ, _segment(code, lit).lstrip()))  # continuation lines sit inside brackets
            repl.append((lit, name))
        new_code = _replace_nodes(code, repl) if repl else code
        if j == i:
            node = ast.parse(new_code).body[0]
            new_expr_src = _segment(new_code, _displayed_expr(node, stmts[i].displays))
        else:
            step_code[j] = textwrap.dedent(new_code).strip()
    return step_code, new_expr_src, lifts, copied


def _parse_expr(expr_src: str) -> ast.AST:
    """A displayed expression (possibly a multi-line method chain) as an AST."""

    return ast.parse("(\n" + expr_src + "\n)", mode="eval")


def _drop_unused_steps(step_js, step_code, parsed, expr_src, copied_js) -> list[int]:
    """Drop copied plain-value steps that the solution no longer reads."""

    keep: list[int] = []
    needed = loaded_names(_parse_expr(expr_src))
    for j in sorted(step_js, reverse=True):
        node = ast.parse(step_code[j])
        if j in copied_js and not (stored_names(parsed[j]) & needed):
            continue
        keep.append(j)
        needed |= loaded_names(node)
        for s in node.body:
            needed |= mutated_names(s)
    return sorted(keep)


# ---------------------------------------------------------------------------
# programs from the two sources
# ---------------------------------------------------------------------------


def docstring_program(page: str, doc: str) -> Program | None:
    """The doctest examples of one API docstring as a program."""

    try:
        examples = doctest.DocTestParser().get_examples(doc or "")
    except ValueError:
        return None
    if not examples:
        return None
    stmts = []
    for ex in examples:
        want = ex.want.strip()
        stmts.append(
            Stmt(
                code=ex.source.rstrip("\n"),
                displays=bool(want) and not want.startswith("Traceback"),
                skip=bool(ex.options.get(doctest.SKIP)) or want.startswith("Traceback"),
            )
        )
    return Program(source="docstring", page=page, statements=stmts)


_SECTION_RE = re.compile(r"#\s*--8<--\s*\[start:([\w-]+)\]")


def user_guide_program(page: str, code: str) -> Program | None:
    """Top-level statements of one user-guide snippet file as a program."""

    try:
        mod = ast.parse(code)
    except SyntaxError:
        return None
    lines = code.splitlines()
    section_at: list[str] = []
    current = ""
    for line in lines:
        m = _SECTION_RE.search(line)
        if m:
            current = m.group(1)
        section_at.append(current)
    stmts, labels = [], []
    for node in mod.body:
        seg = "\n".join(lines[node.lineno - 1 : node.end_lineno])
        stmts.append(Stmt(code=seg))
        labels.append(section_at[node.lineno - 1] if node.lineno - 1 < len(section_at) else "")
    return Program(source="user_guide", page=page, statements=stmts, labels=labels)
