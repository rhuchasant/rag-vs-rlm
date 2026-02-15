"""Evaluation metrics for RLM vs RAG comparison."""

import re
from typing import List, Set


def extract_tab_names_from_response(response: str) -> Set[str]:
    """Extract tab/section names mentioned in model response."""
    # Look for patterns like "Tab X", "## Name", "Tab_Name", "Customer_Data", etc.
    patterns = [
        r"Tab[_\s]?\d+[_\s]?([\w_]+)",
        r"##\s*([\w_]+)",
        r"([\w_]+)_Data",
        r"([\w_]+)_Info",
        r"([\w_]+)_Export",
        r"([\w_]+)_Log",
        r"([\w_]+)_Catalog",
        r"([\w_]+)_Records",
    ]
    found = set()
    for p in patterns:
        for m in re.finditer(p, response, re.IGNORECASE):
            found.add(m.group(1).lower())
    return found


def _normalize_tab_name(name: str) -> str:
    """Normalize tab name for comparison (e.g., Tab_1_Customer_Data -> customer_data)."""
    # Take the part after the last Tab_N_ prefix if present
    import re
    m = re.match(r"Tab_\d+_(.+)", name, re.I)
    if m:
        return m.group(1).lower().replace("-", "_")
    return name.lower().replace("-", "_")


def compute_coverage_sections(expected_section_ids: List[str], response: str) -> float:
    """
    Coverage for legal section IDs (e.g., 240.1, 240.2).
    Uses word boundaries to avoid 240.1 matching 240.10.
    """
    if not expected_section_ids:
        return 1.0
    covered = 0
    for sec_id in expected_section_ids:
        # Escape dots for regex; match as whole token
        escaped = re.escape(sec_id)
        if re.search(rf"(^|[\s,]){escaped}($|[\s,])", response):
            covered += 1
        elif response.strip().endswith(sec_id):
            covered += 1
    return covered / len(expected_section_ids)


def compute_coverage(
    expected_tabs: List[str],
    response: str,
) -> float:
    """
    Compute what fraction of expected tabs/sections were covered in the response.
    For data layout extraction: did we get layouts from all tabs?
    """
    if not expected_tabs:
        return 1.0
    response_lower = response.lower()
    covered = 0
    for tab in expected_tabs:
        # Normalize: Tab_1_Customer_Data -> customer_data, Customer_Data
        norm = _normalize_tab_name(tab)
        # Check if tab name appears in response (flexible matching)
        if (norm in response_lower or
            norm.replace("_", " ") in response_lower or
            tab.lower() in response_lower):
            covered += 1
    return covered / len(expected_tabs)


def compute_exact_match(predicted: str, gold: str) -> bool:
    """Exact string match (for simple QA)."""
    return predicted.strip().lower() == gold.strip().lower()


def count_tabs_in_response(response: str) -> int:
    """Count distinct tab/section headers in response."""
    return len(extract_tab_names_from_response(response))
