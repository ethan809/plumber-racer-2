#!/usr/bin/env python3
"""Integration smoke test for independently cached 64 KiB BC7 -> ASTC tiles."""

import argparse
import hashlib
import importlib.util
from pathlib import Path
import subprocess
import sys
import tempfile


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--astcenc", default="astcenc")
    args = parser.parse_args()
    converter = Path(__file__).with_name("bc_to_astc.py")
    spec = importlib.util.spec_from_file_location("bc_to_astc", converter)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    # A 256x256 BC7 tile is 64 KiB, and its ASTC 4x4 counterpart is also 64 KiB.
    # Each repeated 16-byte BC7 block has a valid mode-6 prefix.
    block = bytes([0x40]) + bytes(15)
    with tempfile.TemporaryDirectory(prefix="a16_tile_test_") as temp:
        root = Path(temp)
        source = root / "tile.bc7"
        output = root / "tile.astcraw"
        cache = root / "cache"
        source.write_bytes(block * (64 * 64))
        # Verify NVN GOB deswizzling with distinct data per 16-byte segment,
        # independently of BC decoding (which can hide addressing mistakes).
        row_bytes, rows, bh = 1024, 64, 2
        linear = bytes((i * 73 + i // 16) & 255 for i in range(row_bytes * rows))
        for switch2 in (False, True):
            swizzled = bytearray(len(linear))
            for y in range(rows):
                for x in range(0, row_bytes, 16):
                    if switch2:
                        gob = (x % 64 // 32) * 256 + (y % 8 // 4) * 128 + (x % 32 // 16) * 64 + (y % 4 // 2) * 32 + (y % 2) * 16
                    else:
                        gob = (x % 64 // 32) * 256 + (y % 8 // 2) * 64 + (x % 32 // 16) * 32 + (y % 2) * 16
                    off = (y // (8 * bh)) * (row_bytes // 64) * (512 * bh) + (x // 64) * (512 * bh) + (y % (8 * bh) // 8) * 512 + gob
                    swizzled[off:off + 16] = linear[y * row_bytes + x:y * row_bytes + x + 16]
            assert module.deswizzle_nvn(swizzled, row_bytes, rows, 1, switch2) == linear
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
        tiled = root / "tile_tiled.bc7"
        # A repeated block is invariant under swizzling, but exercises the
        # full converter including the new pre-decode NVN tile path.
        tiled.write_bytes(source.read_bytes())
        tiled_out = root / "tiled.astcraw"
        tiled_command = [sys.executable, str(converter), "--input", str(tiled),
                         "--output", str(tiled_out), "--width", "256", "--height", "256",
                         "--format", "BC7", "--quality", "fastest", "--nvn-swizzled",
                         "--block-height-log2", "1", "--astcenc", args.astcenc]
        subprocess.run(tiled_command, capture_output=True, text=True, check=True)
        assert tiled_out.stat().st_size == 65536
        print("A16 tile conversion OK: 64 KiB BC7 -> 64 KiB ASTC; cache hit verified")


if __name__ == "__main__":
    main()
