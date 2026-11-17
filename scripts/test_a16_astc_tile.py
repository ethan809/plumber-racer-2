#!/usr/bin/env python3
"""Integration smoke test for independently cached 64 KiB BC7 -> ASTC tiles."""

import argparse
import hashlib
from pathlib import Path
import subprocess
import sys
import tempfile


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--astcenc", default="astcenc")
    args = parser.parse_args()
    converter = Path(__file__).with_name("bc_to_astc.py")
    # A 256x256 BC7 tile is 64 KiB, and its ASTC 4x4 counterpart is also 64 KiB.
    # Each repeated 16-byte BC7 block has a valid mode-6 prefix.
    block = bytes([0x40]) + bytes(15)
    with tempfile.TemporaryDirectory(prefix="a16_tile_test_") as temp:
        root = Path(temp)
        source = root / "tile.bc7"
        output = root / "tile.astcraw"
        cache = root / "cache"
        source.write_bytes(block * (64 * 64))
        command = [sys.executable, str(converter), "--input", str(source),
                   "--output", str(output), "--width", "256", "--height", "256",
                   "--format", "BC7", "--quality", "fastest", "--cache-dir", str(cache),
                   "--astcenc", args.astcenc]
        first = subprocess.run(command, capture_output=True, text=True, check=True)
        assert output.stat().st_size == 65536, first.stdout
        digest = hashlib.sha256(output.read_bytes()).hexdigest()
        output.unlink()
        second = subprocess.run(command, capture_output=True, text=True, check=True)
        assert "ASTC cache hit" in second.stdout, second.stdout
        assert hashlib.sha256(output.read_bytes()).hexdigest() == digest
        assert len(list(cache.glob("*.json"))) == 1
        print("A16 tile conversion OK: 64 KiB BC7 -> 64 KiB ASTC; cache hit verified")


if __name__ == "__main__":
    main()
