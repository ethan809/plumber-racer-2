"""Verify the A16 BC fallback guards before building an experimental IPA.

The guards are part of the renderer so texture format changes cannot silently
remove the old build-time safety patch. The check does not modify source files.
"""

from pathlib import Path

source = Path("dm/gfx/nvn/nvn_mtl.m").read_text()
start = "static MTLTextureDescriptor *tex_desc(const tex_t *t, int packaged) {"
end = "    // Metal's limits:"
if source.count(start) != 1:
    raise SystemExit("A16 BC guard: texture descriptor missing or ambiguous")
section = source.split(start, 1)[1].split(end, 1)[0]
required = (
    "if (bc_cpu_fallback())",
    "if (!packaged)",
    "MTLPixelFormatBC6H_RGBFloat",
    "MTLPixelFormatBC6H_RGBUfloat",
    "MTLPixelFormatR8Unorm",
    "MTLPixelFormatRG8Unorm",
    "MTLPixelFormatRGBA8Unorm",
)
missing = [part for part in required if part not in section]
if missing:
    raise SystemExit(f"A16 BC guard: unsafe fallback descriptor (missing {missing})")

print("A16 BC guard verified (R8/RG8/RGBA8; non-packaged BC and BC6H refused)")

