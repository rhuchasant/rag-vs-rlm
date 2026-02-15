"""
Download a real CFR (Code of Federal Regulations) document from GovInfo.
Uses public-domain U.S. government content - no API key required.

Example: 17 CFR Part 240 (SEC rules under Securities Exchange Act)
URL: https://www.govinfo.gov/content/pkg/CFR-2024-title17-vol4/html/CFR-2024-title17-vol4.htm
"""

import json
import re
import ssl
import urllib.request
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# GovInfo CFR URLs (public domain, no auth)
# XML has cleaner structure than HTML for parsing
CFR_SOURCES = {
    "17-240": {
        "url": "https://www.govinfo.gov/content/pkg/CFR-2024-title17-vol4/xml/CFR-2024-title17-vol4.xml",
        "description": "17 CFR Part 240 - SEC General Rules (Securities Exchange Act)",
    },
}


def fetch_url(url: str) -> str:
    """Fetch URL content. Uses default SSL context."""
    req = urllib.request.Request(url, headers={"User-Agent": "RLM-vs-RAG-Research/1.0"})
    ctx = ssl.create_default_context()
    with urllib.request.urlopen(req, context=ctx, timeout=60) as resp:
        return resp.read().decode("utf-8", errors="replace")


def parse_cfr_xml(xml: str, part_number: int = 240) -> Tuple[Dict[str, str], List[str]]:
    """Parse GovInfo CFR XML. Structure: <SECTNO>240.0-1</SECTNO><SUBJECT>...</SUBJECT>"""
    sections: Dict[str, str] = {}
    ordered_ids: List[str] = []

    # GovInfo CFR XML: SECTNO and SUBJECT are siblings in SUBJGRP
    # Match <SECTNO>240.0-1</SECTNO> ... <SUBJECT>Definitions.</SUBJECT>
    sectno_pattern = re.compile(rf"<SECTNO>({part_number}\.[^<]+?)</SECTNO>")
    # Get all SECTNO values and their positions
    for m in sectno_pattern.finditer(xml):
        sid = m.group(1).strip()
        # Skip ranges like "240.3a4-2—240.3a4-6"
        if "\u2014" in sid or "—" in sid:
            continue
        start = m.end()
        # Find next SUBJECT (content until next SECTNO)
        subj_match = re.search(r"<SUBJECT>([^<]*)</SUBJECT>", xml[start : start + 500])
        content = subj_match.group(1).strip() if subj_match else ""
        # Find next SECTNO - content is everything between this SECTNO and next
        next_sect = sectno_pattern.search(xml, start + 50)
        if next_sect:
            chunk = xml[start : next_sect.start()]
            # Extract all text from P, SUBJECT, etc.
            text = re.sub(r"<[^>]+>", " ", chunk)
            text = re.sub(r"\s+", " ", text).strip()
            content = f"{content}. {text}" if text else content
        if sid not in sections:
            sections[sid] = content or sid
            ordered_ids.append(sid)

    def sort_key(s: str):
        # 240.0-1, 240.3a4-1 -> sortable tuple
        m = re.match(r"(\d+)\.(\d+)(?:-(\d+))?([a-z])?", s, re.I)
        if m:
            a, b, c, d = m.groups()
            return (int(a), int(b), int(c or 0), ord(d or " "))
        return (0, 0, 0, 0)

    ordered_ids = sorted(set(ordered_ids), key=sort_key)
    return sections, ordered_ids


def parse_cfr_html(html: str, part_number: int = 240) -> Tuple[Dict[str, str], List[str]]:
    """
    Parse GovInfo CFR HTML to extract sections.
    GovInfo uses format: "240.0-1", "240.0-2", "240.1", "240.10a" etc.
    Section boundaries: standalone "240.X" or "240.X-Y" at line start.
    """
    sections: Dict[str, str] = {}
    ordered_ids: List[str] = []

    # GovInfo section pattern: 240.0-1, 240.1, 240.10a (part.section or part.section-sub)
    # Match at word boundary to avoid matching "240.1" inside "240.10"
    section_start = re.compile(
        rf"(?:^|\n)\s*({part_number}\.\d+(?:-\d+)?[a-z]?)\s+",
        re.MULTILINE | re.IGNORECASE,
    )
    matches = list(section_start.finditer(html))

    for i, m in enumerate(matches):
        sec_id = m.group(1)
        start = m.start()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(html)
        chunk = html[start:end]

        # Strip HTML, normalize whitespace
        text = re.sub(r"<[^>]+>", " ", chunk)
        text = re.sub(r"\s+", " ", text).strip()
        if len(text) > 100:  # Skip TOC entries and tiny fragments
            sections[sec_id] = text[:50000]
            if sec_id not in ordered_ids:
                ordered_ids.append(sec_id)

    def sort_key(s: str):
        # 240.0-1 -> (240, 0, 1), 240.10a -> (240, 10, ord('a'))
        m = re.match(r"(\d+)\.(\d+)(?:-(\d+))?([a-z])?", s, re.I)
        if m:
            a, b, c, d = m.groups()
            return (int(a), int(b), int(c or 0), ord(d or " "))
        return (0, 0, 0, 0)

    ordered_ids = sorted(set(ordered_ids), key=sort_key)
    return sections, ordered_ids


def download_legal_benchmark(
    source_key: str = "17-240",
    part_number: int = 240,
    output_dir: Optional[Path] = None,
) -> Dict:
    """Download real CFR from GovInfo and save as benchmark."""
    if output_dir is None:
        output_dir = Path(__file__).parent

    if source_key not in CFR_SOURCES:
        raise ValueError(f"Unknown source: {source_key}. Choose from {list(CFR_SOURCES)}")

    url = CFR_SOURCES[source_key]["url"]
    print(f"Fetching {url}...")
    content = fetch_url(url)

    is_xml = "/xml/" in url or content.strip().startswith("<?xml") or content.strip().startswith("<")
    print(f"Parsing {'XML' if is_xml else 'HTML'} ({len(content):,} chars)...")
    if is_xml:
        sections, ordered_ids = parse_cfr_xml(content, part_number)
    else:
        sections, ordered_ids = parse_cfr_html(content, part_number)

    if not sections:
        print("WARNING: No sections parsed. GovInfo HTML structure may have changed.")
        print("Falling back to synthetic generator. Run: python benchmarks/generate_legal_benchmark.py")
        import sys
        sys.path.insert(0, str(output_dir))
        from generate_legal_benchmark import generate_legal_benchmark
        return generate_legal_benchmark(part_number=part_number, num_sections=50, output_dir=output_dir)

    full_text = "\n\n".join(f"{sec_id}\n{sections[sec_id]}" for sec_id in ordered_ids)

    json_path = output_dir / "legal_benchmark.json"
    txt_path = output_dir / "legal_benchmark.txt"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(sections, f, indent=2)
    with open(txt_path, "w", encoding="utf-8") as f:
        f.write(full_text)

    print(f"Downloaded {len(sections)} sections, {len(full_text):,} chars")
    print(f"Saved to {json_path} and {txt_path}")

    return {
        "sections": sections,
        "text": full_text,
        "expected_section_ids": ordered_ids,
        "source": url,
        "part_number": part_number,
    }


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", default="17-240", choices=list(CFR_SOURCES))
    parser.add_argument("--part", type=int, default=240)
    args = parser.parse_args()

    download_legal_benchmark(source_key=args.source, part_number=args.part)
