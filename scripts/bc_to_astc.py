#!/usr/bin/env python3
"""Convert one *linear* BC1-5/BC7 texture to 4x4 ASTC blocks for Metal.

Development helper, not a game asset converter: it does not understand Switch
RomFS containers, tiled NVN layouts, sparse textures or game texture metadata.
Requires a C compiler and ARM's `astcenc` command-line encoder in PATH.
"""

import argparse
import ctypes
import hashlib
import json
from pathlib import Path
import shutil
import struct
import subprocess
import sys
import tempfile
import zlib


def png_chunk(kind: bytes, payload: bytes) -> bytes:
    data = kind + payload
    return struct.pack(">I", len(payload)) + data + struct.pack(">I", zlib.crc32(data))


def write_png(path: Path, rgba: bytes, width: int, height: int) -> None:
    pitch = width * 4
    scanlines = b"".join(b"\0" + rgba[y * pitch:(y + 1) * pitch] for y in range(height))
    header = b"\x89PNG\r\n\x1a\n"
    header += png_chunk(b"IHDR", struct.pack(">IIBBBBB", width, height, 8, 6, 0, 0, 0))
    header += png_chunk(b"IDAT", zlib.compress(scanlines, level=1))
    header += png_chunk(b"IEND", b"")
    path.write_bytes(header)


def convert(args: argparse.Namespace) -> None:
    if not 1 <= args.width <= 16384 or not 1 <= args.height <= 16384:
        raise ValueError("width and height must be between 1 and 16384")
    if args.width * args.height > 64 * 1024 * 1024:
        raise ValueError("texture exceeds safety limit of 64 million pixels")
    bc = int(args.format[2:])
    bpb = 8 if bc in (1, 4) else 16
    expected = ((args.width + 3) // 4) * ((args.height + 3) // 4) * bpb
    source = args.input.read_bytes()
    if len(source) != expected:
        raise ValueError(f"expected {expected} BC bytes, got {len(source)}; input must be deswizzled")
    if args.signed and bc not in (4, 5):
        raise ValueError("--signed is only valid for BC4 and BC5")
    cache_key = hashlib.sha256(
        json.dumps({"version": 1, "format": args.format, "width": args.width,
                    "height": args.height, "signed": args.signed, "srgb": args.srgb,
                    "quality": args.quality}, sort_keys=True).encode() + b"\0" + source
    ).hexdigest()
    cache_file = args.cache_dir / (cache_key + ".astcraw") if args.cache_dir else None
    if cache_file is not None and cache_file.is_file():
        cached = cache_file.read_bytes()
        if len(cached) == expected_astc_size(args.width, args.height):
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_bytes(cached)
            print(f"ASTC cache hit: {cache_key}: {args.output}")
            return
    encoder = shutil.which(args.astcenc)
    if not encoder:
        raise FileNotFoundError("astcenc not found; install ARM astcenc and add it to PATH")
    tex_dir = Path(__file__).resolve().parents[1] / "dm/gfx/tex"
    with tempfile.TemporaryDirectory(prefix="bc_astc_") as td:
        temp = Path(td)
        dylib = temp / ("libbc.dylib" if sys.platform == "darwin" else "libbc.so")
        command = ["cc", "-O2", "-fPIC", "-dynamiclib" if sys.platform == "darwin" else "-shared",
                   "-o", str(dylib), str(tex_dir / "nvn_bc_decode.c")]
        subprocess.run(command, check=True)
        decoder = ctypes.CDLL(str(dylib)).nvn_bc_decode_rgba8
        decoder.argtypes = [ctypes.c_uint, ctypes.c_int, ctypes.c_void_p, ctypes.c_size_t,
                            ctypes.c_uint32, ctypes.c_uint32, ctypes.c_void_p, ctypes.c_size_t]
        decoder.restype = ctypes.c_int
        pixels = ctypes.create_string_buffer(args.width * args.height * 4)
        encoded = ctypes.create_string_buffer(source)
        if decoder(bc, int(args.signed), encoded, len(source), args.width, args.height,
                   pixels, len(pixels)) != 0:
            raise RuntimeError("BC decoder rejected input")
        png = temp / "decoded.png"
        astc = temp / "encoded.astc"
        write_png(png, pixels.raw, args.width, args.height)
        subprocess.run([encoder, "-cs" if args.srgb else "-cl", str(png), str(astc),
                        "4x4", f"-{args.quality}"], check=True)
        data = astc.read_bytes()
        # ASTC file header: magic, 3 block dimensions, 3-byte little-endian x/y/z dimensions.
        if len(data) < 16 or data[:4] != b"\x13\xab\xa1\x5c" or data[4:7] != b"\x04\x04\x01":
            raise ValueError("astcenc output has an unexpected ASTC header")
        w = int.from_bytes(data[7:10], "little")
        h = int.from_bytes(data[10:13], "little")
        depth = int.from_bytes(data[13:16], "little")
        if (w, h, depth) != (args.width, args.height, 1) or len(data) != 16 + expected_astc_size(w, h):
            raise ValueError("astcenc output dimensions or data size do not match input")
        args.output.parent.mkdir(parents=True, exist_ok=True)
        raw_astc = data[16:]
        args.output.write_bytes(raw_astc)  # raw ASTC blocks (no 16-byte file header)
        if cache_file is not None:
            cache_file.parent.mkdir(parents=True, exist_ok=True)
            cache_file.write_bytes(raw_astc)
            cache_file.with_suffix(".json").write_text(json.dumps({
                "source_sha256": hashlib.sha256(source).hexdigest(),
                "cache_key": cache_key, "source_format": args.format,
                "width": args.width, "height": args.height,
                "signed": args.signed, "srgb": args.srgb,
                "quality": args.quality, "astc_format": "ASTC_4x4",
                "source_bytes": len(source), "astc_bytes": len(raw_astc),
            }, indent=2) + "\n")
        print(f"{args.format} -> ASTC 4x4: {len(source)} -> {len(data)-16} bytes: {args.output}")


def expected_astc_size(width: int, height: int) -> int:
    return ((width + 3) // 4) * ((height + 3) // 4) * 16


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input", type=Path, required=True, help="Raw deswizzled BC blocks")
    p.add_argument("--output", type=Path, required=True, help="Raw ASTC 4x4 blocks")
    p.add_argument("--width", type=int, required=True)
    p.add_argument("--height", type=int, required=True)
    p.add_argument("--format", choices=["BC1", "BC2", "BC3", "BC4", "BC5", "BC7"], required=True)
    p.add_argument("--quality", choices=["fastest", "fast", "medium", "thorough"], default="fast")
    p.add_argument("--srgb", action="store_true", help="Use sRGB ASTC colour encoding")
    p.add_argument("--signed", action="store_true", help="BC4/BC5 channels are signed")
    p.add_argument("--astcenc", default="astcenc", help="ASTC encoder executable")
    p.add_argument("--cache-dir", type=Path, help="Store and reuse converted blocks by content hash")
    args = p.parse_args()
    try:
        convert(args)
    except (OSError, ValueError, RuntimeError, subprocess.CalledProcessError) as exc:
        p.error(str(exc))


if __name__ == "__main__":
    main()
