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

# A16 RGBA expansion can multiply BC texture memory by four or more.
# Extend the existing finest-mip omission to 256x256 BC textures too.
old_soft = "if (MAX(t->d.width, t->d.height) < 512 || MIN(t->d.width, t->d.height) < 256) return 0;"
new_soft = "if ((bc_cpu_fallback() ? MIN(t->d.width, t->d.height) < 256 : (MAX(t->d.width, t->d.height) < 512 || MIN(t->d.width, t->d.height) < 256))) return 0;"
if s.count(old_soft) != 1:
    raise SystemExit("A16 texture-soft threshold not found exactly once")
s = s.replace(old_soft, new_soft, 1)
p.write_text(s)
print("A16 memory-saving 256x256 BC mip reduction enabled")
