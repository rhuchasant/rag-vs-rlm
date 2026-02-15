"""Quick script to inspect GovInfo CFR HTML structure."""
import re
import ssl
import urllib.request

url = "https://www.govinfo.gov/content/pkg/CFR-2024-title17-vol4/html/CFR-2024-title17-vol4.htm"
req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
with urllib.request.urlopen(req, context=ssl.create_default_context(), timeout=60) as resp:
    html = resp.read().decode("utf-8", errors="replace")

# Section pattern: 240.0-1, 240.1, 240.10a, etc.
section_refs = re.findall(r"240\.(\d+(?:-\d+)?[a-z]?)", html)
print("Unique section refs (first 30):", sorted(set(section_refs))[:30])

# Find id= with section-like
ids = re.findall(r'id="([^"]*240[^"]*)"', html)
print("Sample ids:", ids[:10])

# Find where section content starts - look for "240.0-1" followed by text
m = re.search(r"240\.0-1[^<]*<[^>]+>([^<]{100,})", html)
if m:
    print("Sample content after 240.0-1:", repr(m.group(1)[:300]))
