from pathlib import Path

p = Path("dm/gfx/nvn/nvn_mtl.m")
s = p.read_text()
old = """    if (packaged && bc_cpu_fallback() && bc_rgba_kind((MTLPixelFormat)pf, &signed_bc))
        pf = bc_rgba_srgb((MTLPixelFormat)pf) ? MTLPixelFormatRGBA8Unorm_sRGB : MTLPixelFormatRGBA8Unorm;"""
new = """    if (bc_cpu_fallback()) {
        if (bc_rgba_kind((MTLPixelFormat)pf, &signed_bc)) {
            if (!packaged) {
                return nil;
            }
            pf = bc_rgba_srgb((MTLPixelFormat)pf) ? MTLPixelFormatRGBA8Unorm_sRGB : MTLPixelFormatRGBA8Unorm;
        } else if ((MTLPixelFormat)pf == MTLPixelFormatBC6H_RGBFloat ||
                   (MTLPixelFormat)pf == MTLPixelFormatBC6H_RGBUfloat) {
            return nil;
        }
    }"""
if s.count(old) != 1:
    raise SystemExit("Expected A16 BC texture conversion code not found exactly once")
p.write_text(s.replace(old, new, 1))
print("A16 BC guard applied")
