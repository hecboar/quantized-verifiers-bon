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


def normalize_number(raw: str) -> str | None:
    s = raw.strip().replace(",", "").replace("$", "").replace("%", "").strip()
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
