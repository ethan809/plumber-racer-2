#!/usr/bin/env python3
"""Integration smoke test for independently cached 64 KiB BC7 -> ASTC tiles."""

import argparse
import hashlib
import importlib.util
from pathlib import Path
import subprocess
import sys
import tempfile


def swizzle_tile(linear: bytes, row_bytes: int, rows: int, bh: int, switch2: bool) -> bytes:
    """Reference tile layout constructed independently of the converter."""
    swizzled = bytearray(len(linear))
    for y in range(rows):
        for x in range(0, row_bytes, 16):
            if switch2:
                gob = (x % 64 // 32) * 256 + (y % 8 // 4) * 128 + (x % 32 // 16) * 64 + (y % 4 // 2) * 32 + (y % 2) * 16
            else:
                gob = (x % 64 // 32) * 256 + (y % 8 // 2) * 64 + (x % 32 // 16) * 32 + (y % 2) * 16
            off = (y // (8 * bh)) * (row_bytes // 64) * (512 * bh) + (x // 64) * (512 * bh) + (y % (8 * bh) // 8) * 512 + gob
            swizzled[off:off + 16] = linear[y * row_bytes + x:y * row_bytes + x + 16]
    return bytes(swizzled)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--astcenc", default="astcenc")
    parser.add_argument("--swizzle-only", action="store_true", help="Only test tile permutation; no astcenc needed")
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
            swizzled = swizzle_tile(linear, row_bytes, rows, bh, switch2)
            assert module.deswizzle_nvn(swizzled, row_bytes, rows, 1, switch2) == linear
        if args.swizzle_only:
            print("NVN tile layout OK: Switch 1 and Switch 2 64 KiB permutations")
            return
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
        # Distinct BC7 mode-6 blocks make an incorrect swizzle order visible.
        # Compare the *encoded* ASTC bytes, not just the intermediate BC data.
        varied = b"".join(bytes([0x40]) + bytes([(i * 17 + j * 11) & 255
                            for j in range(15)]) for i in range(64 * 64))
        source.write_bytes(varied)
        expected_astc = root / "varied_linear.astcraw"
        command_linear = [sys.executable, str(converter), "--input", str(source),
                          "--output", str(expected_astc), "--width", "256", "--height", "256",
                          "--format", "BC7", "--quality", "fastest", "--astcenc", args.astcenc]
        subprocess.run(command_linear, capture_output=True, text=True, check=True)
        for switch2 in (False, True):
            tiled = root / ("tile_switch2.bc7" if switch2 else "tile_switch1.bc7")
            tiled.write_bytes(swizzle_tile(varied, row_bytes, rows, bh, switch2))
            tiled_out = root / ("switch2.astcraw" if switch2 else "switch1.astcraw")
            tiled_command = [sys.executable, str(converter), "--input", str(tiled),
                             "--output", str(tiled_out), "--width", "256", "--height", "256",
                             "--format", "BC7", "--quality", "fastest", "--nvn-swizzled",
                             "--block-height-log2", "1", "--astcenc", args.astcenc]
            if switch2:
                tiled_command.append("--switch2")
            subprocess.run(tiled_command, capture_output=True, text=True, check=True)
            assert tiled_out.read_bytes() == expected_astc.read_bytes(), "ASTC output differs after NVN deswizzle"
        print("A16 tile conversion OK: distinct BC7 blocks, both Switch tile layouts, ASTC parity and cache")


if __name__ == "__main__":
    main()
