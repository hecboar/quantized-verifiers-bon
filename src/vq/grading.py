"""Final-answer extraction and correctness grading.

Fixed before any scoring (plan §4.1). Unparseable answers count as incorrect.
"Last number in the completion" is deliberately NOT an accepted extractor.
"""
from __future__ import annotations

import re
import sys
from fractions import Fraction

_BOX_CMDS = (r"\boxed", r"\fbox")
_ANSWER_PHRASE = re.compile(
    r"(?:final answer|the answer)\s*(?:is|:)\s*\$?\s*([-+]?[\d,]*\.?\d+(?:/\d+)?)",
    re.IGNORECASE,
)
_HASH = re.compile(r"####\s*([^\n]+)")


def extract_boxed(text: str) -> str | None:
    """Content of the last \\boxed{...} / \\fbox{...}, with balanced braces."""
    start = max(text.rfind(cmd) for cmd in _BOX_CMDS)
    if start < 0:
        return None
    brace = text.find("{", start)
    if brace < 0:
        return None
    depth = 0
    for i in range(brace, len(text)):
        if text[i] == "{":
            depth += 1
        elif text[i] == "}":
            depth -= 1
            if depth == 0:
                return text[brace + 1 : i].strip()
    return None


_FRAC = re.compile(r"\\[dt]?frac\{\s*([-+]?[\d.]+)\s*\}\{\s*([-+]?[\d.]+)\s*\}")
_KEEP_CONTENT = re.compile(r"\\(?:textbf|mathbf|mathrm|boldsymbol)\{([^{}]*)\}")
_DROP_TEXT = re.compile(r"\\(?:text|mbox|textrm)\{[^{}]*\}")


def _strip_latex_number(s: str) -> str:
    """Numeric LaTeX produced inside \\boxed{} -> plain number text."""
    s = _KEEP_CONTENT.sub(r"\1", s)
    s = _DROP_TEXT.sub("", s)  # units such as \text{ dollars}
    s = s.replace("{,}", "")
    for tok in (r"\,", r"\!", r"\;", r"\ ", r"\$", r"\%", "^\\circ", r"\left", r"\right"):
        s = s.replace(tok, "")
    return _FRAC.sub(r"\1/\2", s)


def normalize_number(raw: str) -> str | None:
    s = _strip_latex_number(raw.strip())
    s = s.replace(",", "").replace("$", "").replace("%", "").strip()
    s = s.rstrip(".")
    try:
        value = Fraction(s)
    except (ValueError, ZeroDivisionError):
        return None
    if value.denominator == 1:
        return str(value.numerator)
    return repr(float(value)).rstrip("0").rstrip(".")


def extract_final_answer(text: str) -> str | None:
    if not text:
        return None
    boxed = extract_boxed(text)
    if boxed is not None:
        return boxed
    m = _HASH.findall(text)
    if m:
        return m[-1].strip()
    m = _ANSWER_PHRASE.findall(text)
    if m:
        return m[-1].strip()
    return None


def gsm8k_gold(solution: str) -> str:
    m = _HASH.findall(solution)
    if not m:
        raise ValueError("GSM8K solution without '####' marker")
    out = normalize_number(m[-1])
    if out is None:
        raise ValueError(f"unparseable GSM8K gold: {m[-1]!r}")
    return out


def _math_equal(pred: str, gold: str) -> bool:
    from math_verify import parse, verify

    # math_verify's timeouts use signals/subprocesses that break on Windows.
    timeout = None if sys.platform == "win32" else 5
    wrap = lambda s: f"${s}$"
    try:
        g = parse(wrap(gold), parsing_timeout=timeout)
        p = parse(wrap(pred), parsing_timeout=timeout)
        return bool(verify(g, p, timeout_seconds=timeout))
    except Exception:
        return False


def grade(completion: str, gold: str, kind: str) -> bool:
    """kind='numeric' (GSM8K) or 'math' (MATH500)."""
    pred = extract_final_answer(completion)
    if pred is None:
        return False
    if kind == "numeric":
        p, g = normalize_number(pred), normalize_number(gold)
        return p is not None and p == g
    if kind == "math":
        p, g = normalize_number(pred), normalize_number(gold)
        if p is not None and g is not None:
            return p == g
        return _math_equal(pred, gold)
    raise ValueError(f"unknown kind {kind!r}")


# ------------------------------------------------ exploratory sensitivity only
_LAST_NUMBER = re.compile(r"[-+]?\d[\d,]*(?:\.\d+)?")
_LAST_MATH = re.compile(r"\$\$(.+?)\$\$|\$(.+?)\$|\\((.+?)\\)|\\[(.+?)\\]", re.DOTALL)


def grade_lenient(completion: str, gold: str, kind: str) -> bool:
    """Sensitivity analysis for prose answers (NOT the registered grader):
    identical to `grade` whenever a final answer is extractable; otherwise
    accepts the last number (GSM8K) or the last inline math expression, else
    the last number (MATH), in the final 400 characters."""
    if extract_final_answer(completion) is not None:
        return grade(completion, gold, kind)
    tail = (completion or "")[-400:]
    g = normalize_number(gold)
    if kind == "math":
        exprs = [e for m in _LAST_MATH.findall(tail) if (e := next((x for x in m if x.strip()), None))]
        if exprs:
            e = exprs[-1].strip()
            p = normalize_number(e)
            return p == g if (p is not None and g is not None) else _math_equal(e, gold)
    nums = _LAST_NUMBER.findall(tail)
    if not nums or g is None:
        return False
    return normalize_number(nums[-1].rstrip(".,")) == g
