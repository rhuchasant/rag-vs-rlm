"""RLM utility functions - code block parsing, final answer detection."""

import re
from typing import List, Optional, Tuple


def is_likely_code(s: str) -> bool:
    """Heuristic: value looks like code rather than a proper answer."""
    if not s or len(s) < 50:
        return False
    s_lower = s.lower()
    code_indicators = (
        "```" in s
        or "def " in s_lower
        or "import " in s_lower
        or ("for " in s_lower and " in " in s_lower)
        or "llm_query(" in s_lower
        or s.strip().startswith("#")
    )
    return code_indicators and len(s) > 100


def sanitize_numeric_result(s: str) -> Optional[str]:
    """
    Sanitize RLM return value: deduplicate XX pattern, reject variable names.
    Returns None if value looks invalid (e.g. variable name instead of number).
    """
    if not s or not isinstance(s, str):
        return None
    s = str(s).strip()
    
    # Reject variable-name-like values (e.g. total_rollstotal_rolls)
    var_names = ("total_rolls", "total_count", "count", "results", "output", "answer", "final_answer")
    s_lower = s.lower()
    if s_lower in var_names or any(s_lower == v + v for v in var_names):
        return None
    if not any(c.isdigit() for c in s):
        return None
    
    # Deduplicate XX pattern (2424 -> 24, 7373 -> 73, 103103 -> 103)
    if len(s) >= 2 and len(s) % 2 == 0 and s[: len(s) // 2] == s[len(s) // 2 :]:
        s = s[: len(s) // 2]
    
    return s if s else None


def extract_number_from_text(s: str) -> Optional[str]:
    """Extract the most likely numeric answer from text (e.g. when model stored code)."""
    if not s:
        return None
    # Find all integers
    nums = re.findall(r"\b\d+\b", s)
    if not nums:
        return None
    # Prefer the last substantial number (often the computed result)
    for n in reversed(nums):
        if int(n) <= 10000:  # Sanity: avoid line numbers etc.
            return n
    return nums[-1] if nums else None


def find_code_blocks(text: str) -> Optional[List[str]]:
    """Extract ```repl code blocks from model response."""
    pattern = r"```repl\s*\n(.*?)\n```"
    matches = re.findall(pattern, text, re.DOTALL)
    if not matches:
        return None
    return [m.strip() for m in matches]


def find_final_answer(text: str) -> Optional[Tuple[str, str]]:
    """Find FINAL(...) or FINAL_VAR(...) in response. Returns (type, content) or None."""
    for pattern, kind in [
        (r"FINAL_VAR\((.*?)\)", "FINAL_VAR"),
        (r"FINAL\((.*?)\)", "FINAL"),
    ]:
        match = re.search(pattern, text, re.DOTALL)
        if match:
            return (kind, match.group(1).strip().strip('"').strip("'"))
    return None


def format_execution_result(
    stdout: str, stderr: str, locals_dict: dict, prefix_len: int = 100
) -> str:
    """
    Format REPL execution result as metadata (MIT paper: constant-size summary).
    Returns e.g. stdout: {len} chars, prefix: "{first 100 chars}..."
    instead of full output, so root model relies on variables/sub-calls.
    """
    parts = []
    if stdout:
        n = len(stdout)
        prefix = stdout[:prefix_len].replace("\n", " ")
        suffix = "..." if n > prefix_len else ""
        parts.append(f'stdout: {n} chars, prefix: "{prefix}{suffix}"')
    if stderr:
        n = len(stderr)
        prefix = stderr[:prefix_len].replace("\n", " ")
        suffix = "..." if n > prefix_len else ""
        parts.append(f'stderr: {n} chars, prefix: "{prefix}{suffix}"')
    return "\n".join(parts) if parts else "No output"
