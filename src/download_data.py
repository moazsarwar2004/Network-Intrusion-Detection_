"""Download the two pinned NSL-KDD benchmark files and verify SHA-256."""
from pathlib import Path
import hashlib
import json
from urllib.request import urlopen
ROOT = Path(__file__).resolve().parents[1]
manifest = json.loads((ROOT / 'data/manifest.json').read_text())
for name, info in manifest.items():
    destination = ROOT / 'data' / name
    if destination.exists() and hashlib.sha256(destination.read_bytes()).hexdigest() == info['sha256']:
        print(name, 'already verified'); continue
    with urlopen(info['url'], timeout=60) as response:
        content = response.read()
    if hashlib.sha256(content).hexdigest() != info['sha256']:
        raise ValueError(f'{name}: checksum mismatch; no file was saved')
    destination.write_bytes(content)
    print(name, 'downloaded and verified')
