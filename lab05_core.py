"""Lab 05 toolbox: the Dala Insight data, five tools, a reference agent loop and the harness layers.

Pasted unchanged into one notebook cell. Everything the lab measures lives here, so the notebook
and the instructor's reference run execute the same code. All data is synthetic.

Layout: data and tools -> tool-call parsing -> answer scoring -> the questions -> the model ->
the agent loop with its harness layers -> the fixed workflow -> statistics.
"""

from __future__ import annotations

import ast
import csv
import json
import math
import operator
import re
import time
from collections import Counter
from collections.abc import Callable
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Protocol

MODEL_ID = "Qwen/Qwen2.5-3B-Instruct"
MAX_NEW_TOKENS = 256

SYSTEM_PROMPT = (
    "You are an analyst at Dala Insight, a small firm that writes economic briefs from regional "
    "statistics of Kazakhstan. All data is synthetic. Use the tools to look numbers up; never "
    "answer a number from memory. Report values in the unit the tool returns. When you are done, "
    "end your reply with one line: ANSWER: <value>. For a region question the value is the region "
    "name. Do all arithmetic with the calculate tool, not in your head. If the data does not exist, "
    "write ANSWER: NO DATA."
)


class ToolError(Exception):
    """A tool refused its arguments. The message is written for the model to read."""


# ----------------------------------------------------------------------------- data and tools


@dataclass(frozen=True)
class Dataset:
    """The two CSV files: yearly values per region and the indicator catalogue."""

    rows: tuple[dict[str, Any], ...]
    indicators: dict[str, dict[str, str]]

    @property
    def regions(self) -> list[str]:
        """Sorted region names."""
        return sorted({r["region"] for r in self.rows})


def load_dataset(data_dir: str | Path) -> Dataset:
    """Read ``regional_stats.csv`` and ``indicators.csv`` from ``data_dir``."""
    data_dir = Path(data_dir)
    with open(data_dir / "regional_stats.csv", newline="", encoding="utf-8") as f:
        rows = [dict(r, year=int(r["year"]), value=float(r["value"])) for r in csv.DictReader(f)]
    with open(data_dir / "indicators.csv", newline="", encoding="utf-8") as f:
        indicators = {r["indicator"]: r for r in csv.DictReader(f)}
    return Dataset(tuple(rows), indicators)


_OPS: dict[type, Callable[[float, float], float]] = {
    ast.Add: operator.add, ast.Sub: operator.sub, ast.Mult: operator.mul, ast.Div: operator.truediv,
}


def _eval_arith(node: ast.AST) -> float:
    """Evaluate a parsed arithmetic expression: numbers, + - * / and unary minus only."""
    if isinstance(node, ast.Expression):
        return _eval_arith(node.body)
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)) and not isinstance(node.value, bool):
        return float(node.value)
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.USub):
        return -_eval_arith(node.operand)
    if isinstance(node, ast.BinOp) and type(node.op) in _OPS:
        return _OPS[type(node.op)](_eval_arith(node.left), _eval_arith(node.right))
    raise ToolError("Only numbers, + - * / and parentheses are allowed in an expression.")


class Toolbox:
    """The five tools. Each returns a JSON string; bad arguments raise ``ToolError``."""

    def __init__(self, data: Dataset) -> None:
        self.data = data

    def run(self, name: str, arguments: Any) -> str:
        """Call tool ``name`` with ``arguments`` and return its JSON output.

        Raises:
            ToolError: the tool name or an argument is wrong (the message says how).
        """
        tool = getattr(self, f"tool_{name}", None) if re.fullmatch(r"[a-z_]+", name or "") else None
        if tool is None:
            raise ToolError(f"Unknown tool {name!r}. Tools: {', '.join(TOOL_NAMES)}.")
        if not isinstance(arguments, dict):
            raise ToolError("Tool arguments must be a JSON object.")
        try:
            result = tool(**arguments)
        except TypeError as exc:
            raise ToolError(f"Wrong arguments for {name}: {exc}") from exc
        return json.dumps(result, ensure_ascii=False, separators=(",", ":"))

    def _region(self, region: Any) -> str:
        for known in self.data.regions:
            if isinstance(region, str) and known.lower() == region.strip().lower():
                return known
        raise ToolError(f"Unknown region {region!r}. Call list_regions to see the valid names.")

    def _indicator(self, indicator: Any) -> str:
        if indicator not in self.data.indicators:
            raise ToolError(f"Unknown indicator {indicator!r}. Valid: {', '.join(sorted(self.data.indicators))}.")
        return indicator

    @staticmethod
    def _year(value: Any, label: str) -> int | None:
        if value is None:
            return None
        if isinstance(value, bool) or not isinstance(value, int):
            raise ToolError(f"{label} must be an integer year, got {value!r}.")
        return value

    def tool_list_regions(self) -> list[str]:
        """List every region and city of republican significance in the dataset."""
        return self.data.regions

    def tool_list_indicators(self) -> list[dict[str, str]]:
        """List the indicators with their units and definitions."""
        return list(self.data.indicators.values())

    def tool_get_series(self, region: Any, indicator: Any, start_year: Any = None, end_year: Any = None) -> list[dict]:
        """Yearly values of one indicator for one region; years with no data are absent."""
        region, indicator = self._region(region), self._indicator(indicator)
        lo, hi = self._year(start_year, "start_year"), self._year(end_year, "end_year")
        out = [
            {"year": r["year"], "value": r["value"], "unit": r["unit"]}
            for r in self.data.rows
            if r["region"] == region and r["indicator"] == indicator
            and (lo is None or r["year"] >= lo) and (hi is None or r["year"] <= hi)
        ]
        return sorted(out, key=lambda r: r["year"])

    def tool_compare_regions(self, indicator: Any, year: Any) -> list[dict]:
        """Rank all regions that have data for one indicator in one year, highest first."""
        indicator, year = self._indicator(indicator), self._year(year, "year")
        ranked = [
            {"region": r["region"], "value": r["value"], "unit": r["unit"]}
            for r in self.data.rows if r["indicator"] == indicator and r["year"] == year
        ]
        if not ranked:
            raise ToolError(f"No data for {indicator!r} in {year}.")
        return sorted(ranked, key=lambda r: r["value"], reverse=True)

    def tool_calculate(self, expression: Any) -> dict[str, float]:
        """Evaluate an arithmetic expression such as (120 - 100) / 100 * 100."""
        if not isinstance(expression, str):
            raise ToolError("expression must be a string.")
        try:
            value = _eval_arith(ast.parse(expression.strip(), mode="eval"))
        except SyntaxError as exc:
            raise ToolError(f"Cannot parse expression {expression!r}.") from exc
        except ZeroDivisionError as exc:
            raise ToolError("Division by zero.") from exc
        return {"result": round(value, 4)}


def _schema(name: str, description: str, properties: dict[str, dict], required: list[str]) -> dict:
    return {"type": "function", "function": {"name": name, "description": description,
            "parameters": {"type": "object", "properties": properties, "required": required}}}


TOOL_SCHEMAS = [
    _schema("list_regions", "List every region and city in the dataset.", {}, []),
    _schema("list_indicators", "List the indicators with their units and definitions.", {}, []),
    _schema("get_series", "Yearly values of one indicator for one region. Years with no data are absent.", {
        "region": {"type": "string", "description": "Region name exactly as list_regions prints it"},
        "indicator": {"type": "string", "description": "population, avg_monthly_wage, unemployment_rate, cpi_yoy or retail_trade"},
        "start_year": {"type": "integer"}, "end_year": {"type": "integer"}}, ["region", "indicator"]),
    _schema("compare_regions", "Rank ALL regions by one indicator in one year, highest first. Use it to find which region is highest or lowest.", {
        "indicator": {"type": "string"}, "year": {"type": "integer"}}, ["indicator", "year"]),
    _schema("calculate", "Evaluate arithmetic: numbers, + - * / and parentheses.", {
        "expression": {"type": "string", "description": "for example (120 - 100) / 100 * 100"}}, ["expression"]),
]
TOOL_NAMES = [s["function"]["name"] for s in TOOL_SCHEMAS]


# ----------------------------------------------------------------------------- tool-call parsing

_CALL_RE = re.compile(r"<tool_call>\s*(.*?)\s*(?:</tool_call>|$)", re.DOTALL)


@dataclass(frozen=True)
class ToolCall:
    """One tool call found in a model reply. ``error`` is set when the JSON was not usable."""

    name: str
    arguments: Any
    error: str | None = None


def parse_tool_calls(text: str) -> list[ToolCall]:
    """Find the ``<tool_call>`` blocks in a reply (Qwen's format: a JSON object with name and arguments).

    A block whose JSON cannot be read comes back as a ``ToolCall`` with ``error`` set, so the caller
    can tell "the model wrote no call" from "the model wrote a broken call".
    """
    calls = []
    for body in _CALL_RE.findall(text):
        try:
            obj = json.loads(body)
        except json.JSONDecodeError as exc:
            calls.append(ToolCall("", None, f"Tool call is not valid JSON ({exc.msg})."))
            continue
        if not isinstance(obj, dict) or "name" not in obj:
            calls.append(ToolCall("", None, 'Tool call must be a JSON object with "name" and "arguments".'))
            continue
        calls.append(ToolCall(str(obj["name"]), obj.get("arguments", {})))
    return calls


# ----------------------------------------------------------------------------- numbers and answers

_NUM_RE = re.compile(r"(?<![\w.])-?(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?")


def numbers_in(text: str) -> list[str]:
    """Numbers as written in ``text`` ("1,234.5" counts as one); thousands commas kept."""
    return _NUM_RE.findall(text)


def to_float(token: str) -> float:
    """Parse a number token from ``numbers_in``."""
    return float(token.replace(",", ""))


def extract_answer(text: str) -> str | None:
    """The value after the last ``ANSWER:`` line, or ``None`` if there is no such line."""
    found = re.findall(r"^\s*ANSWER:\s*(.+?)\s*$", text, flags=re.MULTILINE | re.IGNORECASE)
    return found[-1] if found else None


def is_no_data(answer: str) -> bool:
    """True for the agreed "the data does not exist" answer."""
    return re.sub(r"[^a-z ]", "", answer.lower()).strip() == "no data"


def is_grounded(answer: str, tool_outputs: list[str]) -> tuple[bool, str]:
    """Does every value in ``answer`` appear in some tool output? Returns ``(ok, reason)``.

    Numbers are compared after rounding the tool value to as many decimals as the answer shows, so
    "95.2" matches 95.1999. Years (1900-2100 written as integers) are exempt. A name must occur in
    a tool output as text. "NO DATA" is grounded when a tool returned an empty list or "No data".
    """
    joined = " ".join(tool_outputs)
    if is_no_data(answer):
        empty = any(o.strip() in ("[]", "") or o.startswith("Error: No data") for o in tool_outputs)
        return (empty, "" if empty else "no tool returned an empty result, so NO DATA is not supported")
    tool_numbers = [to_float(t) for o in tool_outputs for t in numbers_in(o)]
    tokens = numbers_in(answer)
    if not tokens:
        ok = answer.strip().lower() in joined.lower()
        return (ok, "" if ok else f"{answer!r} does not occur in any tool output")
    for token in tokens:
        value = to_float(token)
        if "." not in token and 1900 <= value <= 2100:
            continue
        decimals = len(token.split(".")[1]) if "." in token else 0
        if not any(round(t, decimals) == round(value, decimals) for t in tool_numbers):
            return (False, f"the number {token} is not in any tool output")
    return (True, "")


@dataclass(frozen=True)
class Question:
    """A question with the answer computed from the data (never typed by hand)."""

    qid: str
    kind: str            # lookup | rank | growth | pp | nodata
    text: str
    expected: Any        # float, region name, or None for "no data"
    tol: float = 0.0
    shape: str = "open"  # "fixed" if a one-call workflow can answer it
    params: dict[str, Any] = field(default_factory=dict)


def score_answer(q: Question, answer: str | None) -> str:
    """``correct``, ``wrong`` or ``none`` (no usable answer). A refusal to answer is ``none``, not wrong.

    Scored by content, so a reply in a sentence counts as long as it contains the expected value:
    a number matches if it is within ``q.tol`` (years written as 1900-2100 are ignored); a region
    must appear by its full name; "no data" is correct only for the questions that have none.
    """
    if answer is None or not answer.strip() or answer.strip().upper() == "UNVERIFIED":
        return "none"
    if q.expected is None:
        return "correct" if re.search(r"\bno data\b", answer, flags=re.IGNORECASE) else "wrong"
    if re.search(r"\bno data\b", answer, flags=re.IGNORECASE):
        return "wrong"
    if isinstance(q.expected, str):
        return "correct" if re.search(rf"\b{re.escape(q.expected)}\b", answer, flags=re.IGNORECASE) else "wrong"
    values = [to_float(t) for t in numbers_in(answer) if not ("." not in t and 1900 <= to_float(t) <= 2100)]
    return "correct" if any(abs(v - q.expected) <= q.tol for v in values) else "wrong"


_LABEL = {
    "population": "population", "avg_monthly_wage": "average monthly wage",
    "unemployment_rate": "unemployment rate", "cpi_yoy": "consumer price inflation (CPI, year on year)",
    "retail_trade": "retail trade turnover",
}


def build_questions(data: Dataset) -> list[Question]:
    """The 24 questions of the lab; every expected value is computed from ``data``."""
    tb = Toolbox(data)

    def value(region: str, ind: str, year: int) -> float:
        series = tb.tool_get_series(region, ind, year, year)
        return series[0]["value"]

    def pick(ind: str, year: int, *, highest: bool) -> str:
        ranked = tb.tool_compare_regions(ind, year)
        top = ranked[0] if highest else ranked[-1]
        assert sum(r["value"] == top["value"] for r in ranked) == 1, "tie in a rank question"
        return top["region"]

    qs: list[Question] = []
    for qid, region, ind, year in [
        ("q01", "Karaganda", "avg_monthly_wage", 2024), ("q02", "Shymkent", "population", 2025),
        ("q03", "Atyrau", "unemployment_rate", 2023), ("q04", "Kostanay", "cpi_yoy", 2022),
        ("q05", "Astana", "retail_trade", 2025), ("q06", "Almaty City", "population", 2025),
    ]:
        qs.append(Question(qid, "lookup", f"What was the {_LABEL[ind]} in {region} in {year}?",
                           value(region, ind, year), 0.06, "fixed",
                           {"region": region, "indicator": ind, "year": year}))
    for qid, ind, year, hi in [
        ("q07", "unemployment_rate", 2025, True), ("q08", "avg_monthly_wage", 2024, False),
        ("q09", "population", 2023, True), ("q10", "cpi_yoy", 2022, True), ("q11", "retail_trade", 2021, False),
    ]:
        word = "highest" if hi else "lowest"
        qs.append(Question(qid, "rank", f"Which region had the {word} {_LABEL[ind]} in {year}? Answer with the region name.",
                           pick(ind, year, highest=hi), 0.0, "fixed",
                           {"indicator": ind, "year": year, "order": word}))
    for qid, region, ind, y0, y1 in [
        ("q12", "Pavlodar", "avg_monthly_wage", 2020, 2025), ("q13", "Mangystau", "retail_trade", 2021, 2025),
        ("q14", "Turkistan", "population", 2019, 2025), ("q15", "Kyzylorda", "avg_monthly_wage", 2022, 2024),
    ]:
        a, b = value(region, ind, y0), value(region, ind, y1)
        qs.append(Question(qid, "growth",
                           f"By how many percent did the {_LABEL[ind]} in {region} change between {y0} and {y1}? "
                           "Give the percent change with one decimal.",
                           round((b - a) / a * 100, 1), 0.15, "open"))
    for qid, region, ind, y0, y1 in [
        ("q16", "Astana", "unemployment_rate", 2021, 2025), ("q17", "Aktobe", "unemployment_rate", 2020, 2023),
        ("q18", "Jambyl", "cpi_yoy", 2022, 2025),
    ]:
        diff = round(value(region, ind, y1) - value(region, ind, y0), 1)
        assert diff != 0
        qs.append(Question(qid, "pp",
                           f"By how many percentage points did the {_LABEL[ind]} in {region} "
                           f"{'rise' if diff > 0 else 'fall'} between {y0} and {y1}?",
                           abs(diff), 0.06, "open"))
    for qid, region, ind, year in [("q19", "Ulytau", "avg_monthly_wage", 2020), ("q20", "Abai", "unemployment_rate", 2019)]:
        assert not tb.tool_get_series(region, ind, year, year)
        qs.append(Question(qid, "nodata", f"What was the {_LABEL[ind]} in {region} in {year}?", None, 0.0, "open"))
    # Names as people really write them: an old name, a transliteration, a descriptive phrase. The tools
    # accept only the dataset's names, so the first call fails and the agent has to look the name up.
    for qid, alias, canonical, ind, year in [
        ("q21", "Nur-Sultan", "Astana", "unemployment_rate", 2022),
        ("q22", "Chimkent", "Shymkent", "avg_monthly_wage", 2023),
        ("q23", "Karagandy", "Karaganda", "retail_trade", 2024),
        ("q24", "the city of Almaty", "Almaty City", "population", 2024),
    ]:
        assert alias not in data.regions and canonical in data.regions
        qs.append(Question(qid, "alias", f"What was the {_LABEL[ind]} in {alias} in {year}?",
                           value(canonical, ind, year), 0.06, "fixed",
                           {"region": canonical, "indicator": ind, "year": year, "alias": alias}))
    return qs


# ----------------------------------------------------------------------------- the model


@dataclass(frozen=True)
class Generation:
    """One model reply and what it cost."""

    text: str
    prompt_tokens: int
    new_tokens: int


class Model(Protocol):
    """Anything that turns chat messages into a reply (the real model, or a script in tests)."""

    def generate(self, messages: list[dict], *, use_tools: bool = True,
                 max_new_tokens: int = MAX_NEW_TOKENS) -> Generation: ...


def pick_device() -> str:
    """``cuda`` if present, else Apple ``mps``, else ``cpu``."""
    import torch

    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


class LocalLLM:
    """``Qwen2.5-Instruct`` in eval mode, greedy decoding: the same messages give the same reply."""

    def __init__(self, model_id: str = MODEL_ID, device: str | None = None) -> None:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.device = device or pick_device()
        dtype = torch.float32 if self.device == "cpu" else torch.float16
        self.tokenizer = AutoTokenizer.from_pretrained(model_id)
        self.model = AutoModelForCausalLM.from_pretrained(model_id, dtype=dtype).to(self.device)
        self.model.eval()

    def generate(self, messages: list[dict], *, use_tools: bool = True,
                 max_new_tokens: int = MAX_NEW_TOKENS) -> Generation:
        """Reply to ``messages``; the tool schemas are put into the prompt when ``use_tools``."""
        import torch

        prompt = self.tokenizer.apply_chat_template(
            messages, tools=TOOL_SCHEMAS if use_tools else None, add_generation_prompt=True, tokenize=False)
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        with torch.inference_mode():
            # repetition_penalty=1.0: the checkpoint's default is 1.1, which punishes copying a digit twice.
            out = self.model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False,
                                      repetition_penalty=1.0, temperature=None, top_p=None, top_k=None)
        new_ids = out[0][inputs["input_ids"].shape[1]:]
        text = self.tokenizer.decode(new_ids, skip_special_tokens=True)
        return Generation(text, int(inputs["input_ids"].shape[1]), int(new_ids.shape[0]))


# ----------------------------------------------------------------------------- the agent loop


def build_messages(question: str, notes: str = "") -> list[dict]:
    """The starting conversation: system prompt (plus the firm's notes, if any) and the question."""
    system = SYSTEM_PROMPT + (f"\n\nFirm notes:\n{notes.strip()}" if notes.strip() else "")
    return [{"role": "system", "content": system}, {"role": "user", "content": question}]


@dataclass(frozen=True)
class Harness:
    """Which protective layers are on. ``BARE`` is the loop with none of them."""

    validate: bool = False   # layer 1: a bad call goes back to the model as an error message
    limit: bool = False      # layer 2: tight step budget, repeated calls cut, a forced final answer
    ground: bool = False     # layer 3: the answer line must exist and every value must come from a tool
    max_steps: int = 8
    max_fixes: int = 2       # how many times a layer may send the model back before giving up


BARE = Harness()
LAYER_1 = Harness(validate=True)
LAYER_2 = Harness(validate=True, limit=True, max_steps=5)
LAYER_3 = Harness(validate=True, limit=True, ground=True, max_steps=5)


@dataclass
class RunResult:
    """Everything one question left behind: the answer, the verdict, the cost and the trace."""

    qid: str
    answer: str | None
    outcome: str                      # correct | wrong | none
    failure: str                      # "" or a tag from FAILURES
    model_calls: int
    prompt_tokens: int
    new_tokens: int
    seconds: float
    trace: list[dict]

    def to_dict(self) -> dict[str, Any]:
        """JSON-ready form."""
        return asdict(self)


FAILURES = ("bad_json", "unknown_tool", "bad_args", "step_limit", "no_answer_line", "ungrounded")


def run_agent(model: Model, toolbox: Toolbox, q: Question, harness: Harness = BARE, notes: str = "") -> RunResult:
    """The reference loop. The student's ``my_agent`` in Part 2 is the ``BARE`` path of this function.

    Loop: ask the model; no tool call means that reply is the final answer; otherwise run every call,
    append the results and ask again. The layers in ``harness`` change what happens when something
    goes wrong, never the happy path. ``failure`` is the first thing that went wrong in the run,
    even if a layer then recovered from it.
    """
    t0 = time.perf_counter()
    messages = build_messages(q.text, notes)
    trace: list[dict] = []
    evidence: list[str] = []          # tool outputs the answer may rely on
    seen: set[str] = set()
    count = {"calls": 0, "prompt": 0, "new": 0, "fixes": 0}
    failure = ""

    def ask(use_tools: bool = True) -> str:
        gen = model.generate(messages, use_tools=use_tools)
        count["calls"] += 1
        count["prompt"] += gen.prompt_tokens
        count["new"] += gen.new_tokens
        messages.append({"role": "assistant", "content": gen.text})
        trace.append({"kind": "model", "text": gen.text})
        return gen.text

    def send(kind: str, role: str, text: str) -> None:
        trace.append({"kind": kind, "text": text})
        messages.append({"role": role, "content": text})

    def done(final: str | None) -> RunResult:
        return RunResult(q.qid, final, score_answer(q, final), failure, count["calls"], count["prompt"],
                         count["new"], round(time.perf_counter() - t0, 2), trace)

    for step in range(1, harness.max_steps + 1):
        text = ask()
        calls = parse_tool_calls(text)

        if not calls:                                           # the model replied without a tool call
            if "<tool_call>" in text:
                failure = failure or "bad_json"
            candidate = extract_answer(text)
            if candidate is None:
                failure = failure or "no_answer_line"
            if harness.ground:
                problem = ""
                if candidate is None:
                    problem = "Your reply has no line 'ANSWER: <value>'. Finish with that line."
                else:
                    ok, why = is_grounded(candidate, evidence)
                    if not ok:
                        failure = failure or "ungrounded"
                        problem = (f"Your answer is not supported by the tool results: {why}. "
                                   "Use tool results; if the value comes from arithmetic, compute it with the calculate tool. "
                                   "If the data does not exist, answer ANSWER: NO DATA.")
                if problem:
                    if count["fixes"] >= harness.max_fixes:
                        return done("UNVERIFIED")
                    count["fixes"] += 1
                    send("reject", "user", problem)
                    continue
            return done(candidate if candidate is not None else text.strip())

        repeated = False
        for call in calls:                                      # run every call the model wrote
            ok = False
            if call.error:
                result, tag = f"Error: {call.error}", "bad_json"
            else:
                try:
                    result, tag, ok = toolbox.run(call.name, call.arguments), "", True
                    evidence.append(result)
                except ToolError as exc:
                    result = f"Error: {exc}"
                    tag = "unknown_tool" if str(exc).startswith("Unknown tool") else "bad_args"
                    if str(exc).startswith("No data"):
                        evidence.append(result)
            failure = failure or tag
            trace.append({"kind": "tool", "name": call.name, "arguments": call.arguments, "ok": ok, "text": result})
            if not ok and not harness.validate:
                return done(None)                               # layer 0: the program falls over here
            messages.append({"role": "tool", "content": result})
            key = json.dumps([call.name, call.arguments], sort_keys=True, default=str)
            repeated = repeated or key in seen
            seen.add(key)

        if harness.limit and (repeated or step == harness.max_steps):
            failure = failure or "step_limit"
            send("reject", "user", "Stop calling tools. Write the final reply now, using only the tool "
                                   "results above, ending with ANSWER: <value>.")
            candidate = extract_answer(ask(use_tools=False))
            if candidate is not None and harness.ground and not is_grounded(candidate, evidence)[0]:
                failure = failure or "ungrounded"
                candidate = "UNVERIFIED"
            return done(candidate)

    failure = failure or "step_limit"                           # budget spent, no final reply
    return done(None)


def _finish(q: Question, answer: str | None, failure: str, calls: int, p_tok: int, n_tok: int,
            t0: float, trace: list[dict]) -> RunResult:
    return RunResult(q.qid, answer, score_answer(q, answer), failure, calls, p_tok, n_tok,
                     round(time.perf_counter() - t0, 2), trace)


# ----------------------------------------------------------------------------- the fixed workflow

WORKFLOW_PROMPT = (
    "Read the question and reply with ONE JSON object and nothing else.\n"
    'For a lookup of one value: {"kind":"lookup","region":"<region>","indicator":"<indicator>","year":<year>}\n'
    'For the region with the highest or lowest value: {"kind":"rank","indicator":"<indicator>","year":<year>,"order":"highest" or "lowest"}\n'
    "indicator is one of: population, avg_monthly_wage, unemployment_rate, cpi_yoy, retail_trade.\n"
    "Region names: Abai, Akmola, Aktobe, Almaty City, Almaty Region, Astana, Atyrau, East Kazakhstan, Jambyl, "
    "Jetisu, Karaganda, Kostanay, Kyzylorda, Mangystau, North Kazakhstan, Pavlodar, Shymkent, Turkistan, "
    "Ulytau, West Kazakhstan.\n"
    "Question: "
)


def run_workflow(model: Model, toolbox: Toolbox, q: Question) -> RunResult:
    """Fixed pipeline: one model call extracts the parameters, plain code does the rest.

    No loop, no choice of tools. If the reply is not a usable JSON object the result is ``none``.
    """
    t0 = time.perf_counter()
    gen = model.generate([{"role": "user", "content": WORKFLOW_PROMPT + q.text}], use_tools=False, max_new_tokens=80)
    trace = [{"step": 1, "kind": "model", "text": gen.text}]
    answer, failure = None, ""
    match = re.search(r"\{.*\}", gen.text, flags=re.DOTALL)
    try:
        spec = json.loads(match.group(0)) if match else None
        if not isinstance(spec, dict):
            raise TypeError("no JSON object")
        if spec.get("kind") == "lookup":
            rows = json.loads(toolbox.run("get_series", {"region": spec["region"], "indicator": spec["indicator"],
                                                         "start_year": spec["year"], "end_year": spec["year"]}))
            answer = str(rows[0]["value"]) if rows else "NO DATA"
        elif spec.get("kind") == "rank":
            ranked = json.loads(toolbox.run("compare_regions", {"indicator": spec["indicator"], "year": spec["year"]}))
            answer = (ranked[0] if spec.get("order") == "highest" else ranked[-1])["region"]
        else:
            raise ValueError("unsupported kind")
    except (ValueError, TypeError, KeyError, ToolError) as exc:
        failure = "bad_args" if isinstance(exc, ToolError) else "bad_json"
        trace.append({"step": 1, "kind": "tool", "ok": False, "text": f"Error: {exc}"})
    return _finish(q, answer, failure, 1, gen.prompt_tokens, gen.new_tokens, t0, trace)


# ----------------------------------------------------------------------------- statistics


def wilson(successes: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """Wilson score interval for a proportion (95 % for ``z=1.96``); stays inside [0, 1]."""
    if n == 0:
        return (0.0, 1.0)
    p = successes / n
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return (max(0.0, centre - half), min(1.0, centre + half))


def summarize(results: list[RunResult]) -> dict[str, Any]:
    """Counts, Wilson interval, cost and failure tags for one configuration."""
    n = len(results)
    correct = sum(r.outcome == "correct" for r in results)
    lo, hi = wilson(correct, n)
    return {
        "n": n, "correct": correct, "wrong": sum(r.outcome == "wrong" for r in results),
        "none": sum(r.outcome == "none" for r in results), "ci95": (round(lo, 3), round(hi, 3)),
        "model_calls": sum(r.model_calls for r in results),
        "tokens": sum(r.prompt_tokens + r.new_tokens for r in results),
        "seconds": round(sum(r.seconds for r in results), 1),
        "failures": dict(Counter(r.failure for r in results if r.failure)),
    }


# ----------------------------------------------------------------------------- reports and checks

_SYMBOL = {"correct": "ok", "wrong": "WRONG", "none": "-"}


def summary_line(label: str, results: list[RunResult]) -> str:
    """One printed line per configuration; the notebook and the reference run use the same format."""
    s = summarize(results)
    return (f"{label:14s} correct {s['correct']:2d}/{s['n']}  wrong {s['wrong']:2d}  no answer {s['none']:2d}  "
            f"95% [{s['ci95'][0]:.2f}, {s['ci95'][1]:.2f}]  model calls {s['model_calls']:3d}  "
            f"tokens {s['tokens']:6d}  seconds {s['seconds']:6.1f}")


def outcome_table(runs: dict[str, list[RunResult]], questions: list[Question]) -> str:
    """Outcome of every question under every configuration (``ok``, ``WRONG`` or ``-`` for no answer)."""
    lines = [f"{'id':4s} {'kind':7s} " + " ".join(f"{name:>9s}" for name in runs)]
    for i, q in enumerate(questions):
        lines.append(f"{q.qid:4s} {q.kind:7s} " + " ".join(f"{_SYMBOL[res[i].outcome]:>9s}" for res in runs.values()))
    return "\n".join(lines)


def _flat(text: str, limit: int) -> str:
    return text.replace("\n", " ")[:limit]


def format_trace(result: RunResult) -> str:
    """The run as readable text: what the model said, which tools ran, what the harness answered."""
    head = f"{result.qid}: answer {result.answer!r} -> {result.outcome}"
    lines = [head + (f" (first thing that went wrong: {result.failure})" if result.failure else "")]
    for e in result.trace:
        if e["kind"] == "tool":
            args = json.dumps(e.get("arguments"), ensure_ascii=False)
            lines.append(f"  tool   {e.get('name', 'workflow')}({args}) -> "
                         f"{'ok' if e['ok'] else 'ERROR'}: {_flat(e['text'], 150)}")
        elif e["kind"] == "reject":
            lines.append(f"  harness: {_flat(e['text'], 200)}")
        else:
            lines.append(f"  model: {_flat(e['text'], 200)}")
    return "\n".join(lines)


def check_loop(agent: Callable[[str], tuple[str, list[dict]]], model: Model, toolbox: Toolbox,
               questions: list[Question], qids: tuple[str, ...] = ("q01", "q07", "q12")) -> bool:
    """Compare the student's ``agent(question) -> (final reply, messages)`` with the reference ``BARE`` loop.

    Decoding is greedy, so a correct loop produces the same final reply and the same number of tool
    results as the reference on every question. Returns True when all questions match.
    """
    by_id = {q.qid: q for q in questions}
    all_ok = True
    for qid in qids:
        ref = run_agent(model, toolbox, by_id[qid], BARE)
        ref_reply = next(e["text"] for e in reversed(ref.trace) if e["kind"] == "model")
        ref_tools = sum(e["kind"] == "tool" for e in ref.trace)
        try:
            reply, messages = agent(by_id[qid].text)
        except Exception as exc:  # noqa: BLE001 - the student's code may fail in any way; report it, do not hide it
            print(f"{qid}: your loop raised {type(exc).__name__}: {exc}")
            all_ok = False
            continue
        tools = sum(m["role"] == "tool" for m in messages)
        same = reply.strip() == ref_reply.strip() and tools == ref_tools
        all_ok = all_ok and same
        print(f"{qid}: {'MATCHES' if same else 'DIFFERS FROM'} the reference loop "
              f"(tool results: yours {tools}, reference {ref_tools})")
        if not same:
            print(f"   yours:     {reply.strip()[:150]!r}\n   reference: {ref_reply.strip()[:150]!r}")
    print("all three match" if all_ok else "not yet: read the lines above, then fix the loop")
    return all_ok


def notes_leak(notes: str, questions: list[Question]) -> list[str]:
    """Question ids whose numeric answer appears in ``notes``. Notes may hold conventions, never answers."""
    written = [to_float(t) for t in numbers_in(notes)]
    return [q.qid for q in questions
            if isinstance(q.expected, float) and any(abs(w - q.expected) <= q.tol for w in written)]
