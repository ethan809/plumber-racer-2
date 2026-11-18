#pragma once
// Initial Xbox NVN2 texture translation building block.
// This is NOT a complete NVN2 renderer. It maps recognized BC families
// to hardware-compressed Direct3D formats without RGBA expansion.
#include <dxgiformat.h>
#include <cstdint>

namespace XboxNVN {
enum class Compression : uint8_t { BC1, BC2, BC3, BC4, BC5, BC6H, BC7, Unsupported };
struct TextureFormat {
    Compression compression;
    bool srgb = false;
    bool signedChannels = false;
};

inline DXGI_FORMAT ToDxgi(TextureFormat f) noexcept {
    switch (f.compression) {
    case Compression::BC1: return f.srgb ? DXGI_FORMAT_BC1_UNORM_SRGB : DXGI_FORMAT_BC1_UNORM;
    case Compression::BC2: return f.srgb ? DXGI_FORMAT_BC2_UNORM_SRGB : DXGI_FORMAT_BC2_UNORM;
    case Compression::BC3: return f.srgb ? DXGI_FORMAT_BC3_UNORM_SRGB : DXGI_FORMAT_BC3_UNORM;
    case Compression::BC4: return f.signedChannels ? DXGI_FORMAT_BC4_SNORM : DXGI_FORMAT_BC4_UNORM;
    case Compression::BC5: return f.signedChannels ? DXGI_FORMAT_BC5_SNORM : DXGI_FORMAT_BC5_UNORM;
    case Compression::BC6H: return f.signedChannels ? DXGI_FORMAT_BC6H_SF16 : DXGI_FORMAT_BC6H_UF16;
    case Compression::BC7: return f.srgb ? DXGI_FORMAT_BC7_UNORM_SRGB : DXGI_FORMAT_BC7_UNORM;
    default: return DXGI_FORMAT_UNKNOWN;
    }
}

// BC blocks are 4x4 texels. A block is either 8 or 16 bytes.
// RowPitch and SlicePitch are the D3D11_SUBRESOURCE_DATA layout values.
struct BlockLayout { uint32_t rowPitch; uint32_t slicePitch; };
inline BlockLayout Layout(Compression f, uint32_t width, uint32_t height) noexcept {
    if (f == Compression::Unsupported || !width || !height) return {0, 0};
    const uint32_t bytes = (f == Compression::BC1 || f == Compression::BC4) ? 8 : 16;
    const uint32_t columns = (width + 3) / 4;
    const uint32_t rows = (height + 3) / 4;
    return {columns * bytes, columns * rows * bytes};
}
} // namespace XboxNVN
