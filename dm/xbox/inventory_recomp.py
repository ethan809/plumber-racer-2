#!/usr/bin/env python3
"""Inventory mkw-recomp runtime dependencies before targeting Xbox UWP.

Run against a checkout of the upstream source. Does not need game files.
"""
from pathlib import Path
import argparse
import re
import sys

RULES = {
    "Desktop-only Windows API": r"\b(?:CreateProcess[AW]?|ShellExecute[AW]?|LoadLibrary[AW]?|GetModuleHandle[AW]?|VirtualAlloc|VirtualProtect|CreateFile[AW]?|SetThreadContext|SuspendThread|ResumeThread|CreateRemoteThread|SetPriorityClass|QueryPerformanceCounter)\s*\(",
    "Vulkan entry points": r"\bvk[A-Z][A-Za-z0-9_]*\s*\(",
    "SDL entry points": r"\bSDL_[A-Za-z0-9_]+\s*\(",
    "Dynamic code/memory": r"\b(?:mmap|mprotect|userfaultfd|VirtualAlloc2|VirtualProtectFromApp|MapViewOfFile3)\s*\(",
    "Desktop/Unix headers": r"^\s*#\s*include\s*[<\"](?:windows\.h|dlfcn\.h|sys/mman\.h|unistd\.h|vulkan/[^>\"]+)[>\"]",
}
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("upstream", type=Path)
    ap.add_argument("--output", type=Path, default=Path("xbox-runtime-inventory.md"))
    args = ap.parse_args()
    runtime = args.upstream / "runtime"
    if not runtime.is_dir():
        sys.exit(f"Missing upstream runtime: {runtime}")
    counts = {name: 0 for name in RULES}
    lines = [
        "# mkw-recomp Xbox UWP compatibility inventory", "",
        "Generated from upstream source. Matches identify code to inspect, not proof an API is forbidden.",
        "Game assets and generated game code are not required.", "",
    ]
    for label, expression in RULES.items():
        rx = re.compile(expression, re.MULTILINE)
        lines.extend([f"## {label}", "", "| File | Line | Symbol |", "|---|---:|---|"])
        for path in sorted(runtime.rglob("*")):
            if path.suffix.lower() not in {".c", ".h", ".cpp", ".hpp"}:
                continue
            source = path.read_text(encoding="utf-8", errors="replace")
            for line_no, line in enumerate(source.splitlines(), 1):
                match = rx.search(line)
                if not match:
                    continue
                counts[label] += 1
                symbol = match.group(0).strip().replace("|", "\\|").replace("`", "'")
                lines.append(f"| `{path.relative_to(args.upstream).as_posix()}` | {line_no} | `{symbol}` |")
        if not counts[label]:
            lines.append("| — | — | No matches |")
        lines.append("")
    lines += ["## Totals", "", "| Category | Matches |", "|---|---:|"]
    lines += [f"| {name} | {count} |" for name, count in counts.items()]
    lines += ["", "Next step: replace or adapt each relevant platform API, then implement NVN2 on D3D12.", ""]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text("\n".join(lines), encoding="utf-8")
    print(f"Wrote {args.output}: " + ", ".join(f"{k}={v}" for k, v in counts.items()))
if __name__ == "__main__":
    main()
