#ifndef NVN_BC_DECODE_H
#define NVN_BC_DECODE_H

#include <stddef.h>
#include <stdint.h>

// Software fallback for devices without Metal BC texture support.
// Input is linear (already deswizzled) BC blocks, output is tightly packed RGBA8.
// BC1/2/3/4/5/7 are supported; BC6H needs an HDR-preserving conversion path.
// signed_channels selects SNORM for BC4/BC5.
// Returns 0 on success, -1 on invalid arguments / unsupported format.
int nvn_bc_decode_rgba8(unsigned bc, int signed_channels,
                        const uint8_t *blocks, size_t blocks_size,
                        uint32_t width, uint32_t height,
                        uint8_t *rgba, size_t rgba_size);

// The A16 BC bridge can store BC4 as R8 and BC5 as RG8 rather than
// expanding their missing channels to RGBA8. Other BC kinds stay RGBA8.
// Signed BC4/5 retain the bridge's existing SNORM-to-UNORM conversion.
int nvn_bc_decode_tight(unsigned bc, int signed_channels,
                        const uint8_t *blocks, size_t blocks_size,
                        uint32_t width, uint32_t height,
                        uint8_t *pixels, size_t pixels_size);

#endif
