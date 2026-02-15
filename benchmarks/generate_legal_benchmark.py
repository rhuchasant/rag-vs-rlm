"""
Generate legal-document benchmark (CFR-style structure).
Simulates Code of Federal Regulations: Parts, Subparts, Sections.
Scales to exceed context (~700k chars) for RLM vs RAG comparison.
"""

import json
import random
from pathlib import Path
from typing import Dict, List, Optional

# Legal terminology for realistic section content
LEGAL_TERMS = [
    "pursuant to", "herein", "thereof", "aforementioned", "notwithstanding",
    "shall", "may", "must", "required", "prohibited", "permitted",
    "applicant", "registrant", "filing", "disclosure", "compliance",
    "effective date", "amendment", "revocation", "suspension",
]

DEFINITION_TEMPLATES = [
    "The term '{term}' means {definition}.",
    "For purposes of this section, '{term}' refers to {definition}.",
    "'{term}' has the meaning set forth in paragraph (a) of this section.",
]

REQUIREMENT_TEMPLATES = [
    "Each {entity} shall {action} within {timeframe}.",
    "No {entity} may {action} without prior {approval}.",
    "The {agency} may {action} upon a showing of {condition}.",
    "Compliance with paragraph (a) is required for {scope}.",
]


def _random_definition() -> str:
    term = random.choice(["registrant", "filing", "disclosure", "material fact", "affiliate"])
    definition = random.choice([
        "any person required to file reports under this part",
        "the submission of documents to the Commission",
        "information required to be made public under this regulation",
    ])
    return random.choice(DEFINITION_TEMPLATES).format(term=term, definition=definition)


def _random_requirement() -> str:
    entity = random.choice(["registrant", "applicant", "filer", "issuer"])
    action = random.choice(["file", "disclose", "submit", "notify", "amend"])
    timeframe = random.choice(["30 days", "10 business days", "the effective date"])
    approval = random.choice(["Commission approval", "written consent", "prior notice"])
    condition = random.choice(["good cause", "exceptional circumstances", "hardship"])
    agency = random.choice(["Commission", "Director", "Secretary"])
    scope = random.choice(["all filings", "registered offerings", "this subpart"])
    return random.choice(REQUIREMENT_TEMPLATES).format(
        entity=entity, action=action, timeframe=timeframe,
        approval=approval, condition=condition, agency=agency, scope=scope,
    )


def generate_section_content(
    section_id: str,
    part_num: int,
    target_chars: int = 2000,
) -> str:
    """Generate one CFR-style section with definitions and requirements."""
    lines = [
        f"§ {section_id}",
        "",
        f"(a) Definitions. {_random_definition()}",
        "",
        f"(b) General requirements. {_random_requirement()}",
        "",
    ]
    current_len = len("\n".join(lines))
    chars_needed = target_chars - current_len

    if chars_needed > 300:
        num_paragraphs = max(2, chars_needed // 400)
        for i in range(num_paragraphs):
            letter = chr(ord("c") + i)
            lines.append(f"({letter}) {_random_requirement()}")
            lines.append("")
        lines.append("")

        # Pad with legal boilerplate
        remaining = target_chars - len("\n".join(lines))
        if remaining > 200:
            boilerplate = [
                "Authority: 15 U.S.C. 78a et seq.; 17 CFR 240.0-1.",
                "Source: 50 FR 12345, Mar. 28, 1985, unless otherwise noted.",
                "Cross-reference: See also § 240.0-1 for general applicability.",
            ]
            for _ in range(remaining // 150):
                lines.append(random.choice(boilerplate))

    return "\n".join(lines).strip()


def generate_legal_benchmark(
    part_number: int = 240,
    num_sections: int = 50,
    target_total_chars: int = 750000,
    seed: int = 42,
    output_dir: Optional[Path] = None,
) -> Dict:
    """
    Generate a CFR-style legal document benchmark.
    Structure: Part X, Subparts A–E, Sections §X.1 through §X.N.
    """
    random.seed(seed)
    if output_dir is None:
        output_dir = Path(__file__).parent

    chars_per_section = max(500, target_total_chars // num_sections)
    sections: Dict[str, str] = {}
    full_text_parts = [
        f"PART {part_number} - GENERAL RULES AND REGULATIONS",
        "",
        "Subpart A - Definitions and Scope",
        "",
    ]

    for i in range(1, num_sections + 1):
        # Add subpart headers every ~10 sections (before section content)
        if i > 1 and i % 10 == 1:
            subpart_letter = chr(ord("A") + (i - 1) // 10)
            full_text_parts.append(f"\nSubpart {subpart_letter} - Additional Provisions\n")

        section_id = f"{part_number}.{i}"
        content = generate_section_content(section_id, part_number, target_chars=chars_per_section)
        sections[section_id] = content
        full_text_parts.append(content)
        full_text_parts.append("")

    full_text = "\n".join(full_text_parts)
    expected_section_ids = [f"{part_number}.{i}" for i in range(1, num_sections + 1)]

    # Save
    json_path = output_dir / "legal_benchmark.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(sections, f, indent=2)

    txt_path = output_dir / "legal_benchmark.txt"
    with open(txt_path, "w", encoding="utf-8") as f:
        f.write(full_text)

    return {
        "sections": sections,
        "text": full_text,
        "expected_section_ids": expected_section_ids,
        "part_number": part_number,
        "num_sections": num_sections,
    }


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--part", type=int, default=240, help="CFR part number")
    parser.add_argument("--sections", type=int, default=50, help="Number of sections")
    parser.add_argument("--size", type=int, default=750000, help="Target total chars")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    data = generate_legal_benchmark(
        part_number=args.part,
        num_sections=args.sections,
        target_total_chars=args.size,
        seed=args.seed,
    )
    print(f"Generated legal benchmark: Part {data['part_number']}, {data['num_sections']} sections")
    print(f"Total chars: {len(data['text']):,} (~{len(data['text'])//4:,} tokens)")
    print(f"Saved to benchmarks/legal_benchmark.json and .txt")
