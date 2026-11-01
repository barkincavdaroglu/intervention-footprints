#!/usr/bin/env python3
"""Check delivered files against their current sanitized file hashes; no model execution."""
from pathlib import Path
import hashlib, json
root = Path(__file__).resolve().parent
manifest = json.loads((root / 'SOURCE_MANIFEST.json').read_text())
bad = []
for row in manifest['files']:
    p = root / row['file']
    if p.is_symlink() or not p.is_file():
        bad.append(row['file']); continue
    h = hashlib.sha256()
    with p.open('rb') as f:
        for chunk in iter(lambda: f.read(1 << 20), b''): h.update(chunk)
    if h.hexdigest() != row['sha256']: bad.append(row['file'])
if bad: raise SystemExit('Changed or missing since sanitized manifest: ' + ', '.join(bad))
print(f"PASS: {len(manifest['files'])} files match the current sanitized manifest.")
