"""Programmatic verifiers. DESIGN.md section 6.

Three return values, and the distinction matters:
  1    -> verified good
  0    -> verified bad, escalate to a stronger arm
  None -> cannot tell, so commit this answer and wait for the sampled judge

Returning 0 where None is meant would cascade on unverifiable output and burn
money for nothing.
"""

import ast
import json
import re

_FENCE = re.compile(r"```(\w+)?\s*\n(.*?)```", re.DOTALL)
_SUPPORTED_CODE_LANGS = {"python", "py", None}


def extract_code_block(text):
    """Return (language, code) from the first fenced block, or (None, None)."""
    if not text:
        return None, None
    match = _FENCE.search(text)
    if match:
        lang = (match.group(1) or "").lower() or None
        return lang, match.group(2)
    return None, None


def extract_boxed(text):
    r"""Return the contents of the last \boxed{...}, brace-balanced."""
    if not text:
        return None
    marker = r"\boxed{"
    start = text.rfind(marker)
    if start == -1:
        return None
    i = start + len(marker)
    depth = 1
    out = []
    while i < len(text) and depth > 0:
        ch = text[i]
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                break
        out.append(ch)
        i += 1
    if depth != 0:
        return None
    return "".join(out).strip()


def _extract_json(text):
    """Parse JSON from raw text or a fenced block. Returns (ok, value)."""
    if not text:
        return False, None
    candidates = []
    _, fenced = extract_code_block(text)
    if fenced:
        candidates.append(fenced)
    candidates.append(text)
    for candidate in candidates:
        stripped = candidate.strip()
        try:
            return True, json.loads(stripped)
        except (json.JSONDecodeError, ValueError):
            continue
    return False, None


def _check_structured(text, schema):
    ok, value = _extract_json(text)
    if not ok:
        return 0
    if not schema:
        return 1
    if schema.get("type") == "object":
        if not isinstance(value, dict):
            return 0
        for key in schema.get("required", []):
            if key not in value:
                return 0
    elif schema.get("type") == "array" and not isinstance(value, list):
        return 0
    return 1


def _check_code(text):
    lang, code = extract_code_block(text)
    if code is None:
        # No fence. Treat the whole response as Python only if it parses as
        # more than a bare expression, otherwise we cannot tell.
        try:
            tree = ast.parse(text or "")
        except SyntaxError:
            return None
        if not tree.body or all(
            isinstance(node, ast.Expr) for node in tree.body
        ):
            return None
        return 1
    if lang not in _SUPPORTED_CODE_LANGS:
        return None
    try:
        ast.parse(code)
        return 1
    except SyntaxError:
        return 0


def _check_math(text, expected):
    if expected is None:
        return None
    answer = extract_boxed(text)
    if answer is None:
        return None
    if answer.strip() == str(expected).strip():
        return 1
    import sympy

    try:
        got = sympy.sympify(answer.replace("\\", ""))
        want = sympy.sympify(str(expected).replace("\\", ""))
        return 1 if sympy.simplify(got - want) == 0 else 0
    except Exception:
        # Unparseable: cannot tell, not wrong.
        return None


def check(task_type, response_text, expected=None, schema=None):
    if task_type == "structured":
        return _check_structured(response_text, schema)
    if task_type == "code":
        return _check_code(response_text)
    if task_type == "math":
        return _check_math(response_text, expected)
    return None
