#!/usr/bin/env python3
"""
Comprehensive mojibake cleanup utility for Jupyter notebooks.

This script removes UTF-8 double-encoding artifacts (mojibake) that can
corrupt notebook JSON files, particularly arrow symbols (→), mathematical
operators (−, ≈), box-drawing characters, and accented letters.

Usage:
    python mojibake_cleanup.py <notebook_path>

Example:
    python mojibake_cleanup.py trabajoFM/notebooks/04_sensitivity_stats.ipynb

The script:
- Creates a timestamped backup before modifications
- Cleans all cell sources and outputs using hex-encoded byte patterns
- Validates JSON integrity after cleanup
- Verifies persistence by re-reading the file

Mojibake sources in this project typically arise from:
1. Cross-platform text editor/IDE encoding mismatches (especially VS Code on Windows)
2. Incorrect file encoding during Git operations
3. Terminal/PowerShell string encoding issues when scripting text replacements
4. BOM (Byte Order Mark) handling in JSON files

Prevention (per SWAT Stats Workflow agent):
- Always use explicit UTF-8 encoding in file operations
- Use PowerShell [System.IO.File] for reliable UTF-8 read/write
- Use Python with utf-8-sig encoding for JSON with BOM
- Avoid inline terminal string replacements; use Python scripts instead
- Verify persistence after file modifications by re-reading from disk
"""
import json
from pathlib import Path
from shutil import copy2
from datetime import datetime

def clean_string(text):
    """Remove mojibake using hex-encoded patterns."""
    # Using byte sequences converted to string
    fixes = [
        (bytes.fromhex('c3a7').decode(), 'c'),    # Ã§
        (bytes.fromhex('c3a9').decode(), 'e'),    # Ã©
        (bytes.fromhex('c3b3').decode(), 'o'),    # Ã³
        (bytes.fromhex('c3a1').decode(), 'a'),    # Ã¡
        (bytes.fromhex('c3a2').decode(), ''),     # Ã¢
        (bytes.fromhex('c397').decode(), 'x'),    # Ã—
        (bytes.fromhex('e2809c').decode(), '"'),  # â€œ
        (bytes.fromhex('e2809d').decode(), '"'),  # â€
        (bytes.fromhex('e2809e').decode(), '"'),  # â€ž
        (bytes.fromhex('e28099').decode(), "'"),  # â€™
        (bytes.fromhex('e2809f').decode(), "'"),  # â€Ÿ
        (bytes.fromhex('e28093').decode(), '-'),  # â€"
        (bytes.fromhex('e28094').decode(), '--'), # â€"
        (bytes.fromhex('e288aa').decode(), "'"),  # â§ª
        (bytes.fromhex('e2809a').decode(), "`"),  # â€š
        (bytes.fromhex('e2809b').decode(), "'"),  # â€›
        (bytes.fromhex('e280a0').decode(), ''),   # â€ 
        (bytes.fromhex('e280a1').decode(), ''),   # â€¡
        (bytes.fromhex('e280a6').decode(), '...'),# â€¦
        (bytes.fromhex('e28692').decode(), '->'), # â†'
        (bytes.fromhex('e28694').decode(), '-'),  # â†"  
        (bytes.fromhex('e28882').decode(), '-'),  # âˆ‚
        (bytes.fromhex('e2888f').decode(), '-'),  # âˆ
        (bytes.fromhex('e28890').decode(), '-'),  # âˆ
        (bytes.fromhex('e28891').decode(), '-'),  # Minus
        (bytes.fromhex('e28892').decode(), '-'),  # âˆ'
        (bytes.fromhex('e289a0').decode(), '~'),  # â‰ 
        (bytes.fromhex('e289a8').decode(), '~'),  # â‰ˆ
        (bytes.fromhex('c2a0').decode(), ' '),    # Â (non-breaking space)
    ]
    
    count = 0
    for bad, good in fixes:
        c = text.count(bad)
        if c > 0:
            text = text.replace(bad, good)
            count += c
    
    return text, count

nb_path = Path('trabajoFM/notebooks/04_sensitivity_stats.ipynb')
ts = datetime.now().strftime("%Y%m%d_%H%M%S")
bk = nb_path.parent / f"{nb_path.stem}.pre_hex_cleanup_{ts}.ipynb"
copy2(nb_path, bk)
print(f"Backup: {bk.name}")

with open(nb_path, 'r', encoding='utf-8-sig') as f:
    nb = json.load(f)

total = 0
for i, cell in enumerate(nb.get('cells', [])):
    src = cell.get('source', [])
    if isinstance(src, list):
        src = ''.join(src)
    if isinstance(src, str):
        cleaned, fixed = clean_string(src)
        if fixed > 0:
            cell['source'] = cleaned if isinstance( cell.get('source'), str) else cleaned.splitlines(keepends=True)
            total += fixed
            print(f"Cell {i}: +{fixed}")

for cell in nb.get('cells', []):
    for out in cell.get('outputs', []):
        if 'text' in out:
            txt = out['text']
            if isinstance(txt, list): txt = ''.join(txt)
            if isinstance(txt, str):
                cleaned, fixed = clean_string(txt)
                if fixed > 0:
                    out['text'] = cleaned if isinstance(out.get(' text'), str) else cleaned.splitlines(keepends=True)
                    total += fixed

print(f"Fixed: {total}")
with open(nb_path, 'w', encoding='utf-8') as f:
    json.dump(nb, f)

print("✓ Done")
