#!/usr/bin/env python3
"""Recover rewritten module images embedded in an nxnative iOS Mach-O.

The app embeds original guest code and data separately in Mach-O sections:
__TEXT,__dm<name>x0; __DM<name>W,__img; __TEXT,__dm<name>x1.
Rejoining these reconstructs patched.bin for ordinary positive PATCH_OFF
builds, but NOT the original ExeFS/NSO or the needed fixups/shader metadata.
Only use with your own legally obtained app executable.
"""

import argparse
from pathlib import Path
import re
import struct


def get_sections(data: bytes) -> dict[tuple[str, str], bytes]:
    if len(data) < 32 or struct.unpack_from("<I", data)[0] != 0xFEEDFACF:
        raise ValueError("expected a little-endian 64-bit Mach-O executable")
    filetype, ncmds, sizeofcmds = struct.unpack_from("<III", data, 12)
    if filetype != 2 or ncmds > 1000 or 32 + sizeofcmds > len(data):
        raise ValueError("invalid Mach-O executable header")
    result = {}
    off = 32
    for _ in range(ncmds):
        if off + 8 > 32 + sizeofcmds:
            raise ValueError("truncated load command")
        command, length = struct.unpack_from("<II", data, off)
        if length < 8 or off + length > 32 + sizeofcmds:
            raise ValueError("invalid load command length")
        if command == 0x19:  # LC_SEGMENT_64
            if length < 72:
                raise ValueError("short segment command")
            segment = data[off + 8:off + 24].split(b"\0", 1)[0].decode("ascii")
            count = struct.unpack_from("<I", data, off + 64)[0]
            if length < 72 + 80 * count:
                raise ValueError("short section table")
            for i in range(count):
                pos = off + 72 + i * 80
                name = data[pos:pos + 16].split(b"\0", 1)[0].decode("ascii")
                size = struct.unpack_from("<Q", data, pos + 40)[0]
                start = struct.unpack_from("<I", data, pos + 48)[0]
                if start + size > len(data):
                    raise ValueError(f"section {segment},{name} exceeds file")
                if name.startswith("__dm") or (name == "__img" and segment.startswith("__DM")):
                    result[segment, name] = data[start:start + size]
        off += length
    return result


def recover(data: bytes, module: str) -> bytes:
    if not re.fullmatch(r"[A-Za-z0-9_]{1,12}", module):
        raise ValueError("invalid module name")
    sections = get_sections(data)
    x0 = sections.get(("__TEXT", f"__dm{module}x0"))
    x1 = sections.get(("__TEXT", f"__dm{module}x1"))
    rw = sections.get((f"__DM{module}W", "__img"))
    if x0 is None or x1 is None or rw is None:
        raise ValueError(f"missing embedded sections for {module}")
    x2 = sections.get(("__TEXT", f"__dm{module}x2"), b"")
    image = x0 + rw + x1 + x2
    if len(image) % 0x4000:
        raise ValueError("module image is not 16 KiB aligned")
    return image


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("executable", type=Path)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--modules", nargs="+", default=["main", "sdk"])
    args = ap.parse_args()
    source = args.executable.read_bytes()
    args.out.mkdir(parents=True, exist_ok=True)
    for name in args.modules:
        image = recover(source, name)
        dest = args.out / f"{name}.patched.bin"
        dest.write_bytes(image)
        print(f"{name}: {len(image):,} bytes -> {dest}")
    print("Recovered patched images only; fixups, gates and shader libraries are still required.")


if __name__ == "__main__":
    main()
