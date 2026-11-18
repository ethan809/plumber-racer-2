#!/usr/bin/env python3
"""Check reused iOS game assets before rebuilding an unsigned IPA.

This is an offline preflight: it does not upload game data, build an IPA, or
assert that the original app's shaders are compatible with changed host code.
"""

import argparse
from pathlib import Path
import plistlib
import re
import sys


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--info", type=Path, required=True, help="Original app's Info.plist")
    ap.add_argument("--romfs-id", type=Path, required=True)
    ap.add_argument("--patched", type=Path, required=True, help="Recovered .patched.bin and .fixups.tsv files")
    ap.add_argument("--msl", type=Path, required=True, help="Unzipped msl folder")
    ap.add_argument("--chain-header", type=Path, required=True)
    args = ap.parse_args()
    info = plistlib.loads(args.info.read_bytes())
    title = str(info.get("CFBundleIdentifier", "")).removeprefix("org.nxnative.").upper()
    chain = info.get("DMChain", "")
    if not re.fullmatch(r"[0-9A-F]{16}", title) or not re.fullmatch(r"[0-9a-f]{12}", chain):
        raise ValueError("original IPA is missing a recognizable game ID or shader-chain stamp")
    identity = args.romfs_id.read_text(encoding="ascii").strip()
    if not re.fullmatch(rf"DMROMFSID1 title={title} version=[^\s]+ build=[^\s]+ size=[1-9]\d* index=[0-9a-f]{{16}}", identity):
        raise ValueError("the supplied RomFS identity does not match the original app")
    modules = ("main", "subsdk0", "sdk") if title == "0400C3F00006E000" else ("main", "sdk")
    for name in modules:
        binary = args.patched / (name + ".patched.bin")
        fixups = args.patched / (name + ".fixups.tsv")
        if not binary.is_file() or not fixups.is_file():
            raise ValueError(f"missing patched image or fixup metadata: {name}")
        if binary.stat().st_size == 0 or binary.stat().st_size % 16384:
            raise ValueError(f"invalid 16 KiB-aligned patched image: {name}")
        header = fixups.read_text(encoding="ascii")
        if not re.search(r"(?m)^# BUILD_ID\s+[0-9a-f]{32}$", header):
            raise ValueError(f"missing build ID in {fixups}")
        print(f"PASS {name}: {binary.stat().st_size:,} bytes and fixup build ID")
    shaders = args.msl / ("b-" + chain)
    stamp = shaders / "inputs.tsv"
    if not stamp.is_file() or f"game\t{title}\n" not in stamp.read_text(encoding="utf-8"):
        raise ValueError("shader bundle missing or belongs to a different game")
    count = sum(1 for _ in shaders.glob("*.metallib"))
    if count == 0:
        raise ValueError("shader bundle has no compiled .metallib files")
    print(f"PASS shaders: {count:,} libraries, chain {chain}")
    h = args.chain_header.read_text(encoding="utf-8")
    match = re.search(r'^#define DMSH_CHAIN "([0-9a-f]+)"$', h, re.M)
    if not match:
        raise ValueError("source shader-chain header has no DMSH_CHAIN")
    if match.group(1) != chain:
        print(f"BLOCKED: recovered shaders are chain {chain}; current source is {match.group(1)}")
        print("Do not bypass this check without validating shader translator/host compatibility.")
        return 2
    print("PASS shader-chain version matches host source")
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except (OSError, ValueError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        sys.exit(1)
