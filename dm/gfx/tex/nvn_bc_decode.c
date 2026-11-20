#include "nvn_bc_decode.h"
#include <string.h>
#define BCDEC_IMPLEMENTATION
#include "bcdec.h"

static uint16_t rd16(const uint8_t *p) { return (uint16_t)(p[0] | (p[1] << 8)); }
static uint32_t rd32(const uint8_t *p) { return (uint32_t)p[0] | ((uint32_t)p[1] << 8) | ((uint32_t)p[2] << 16) | ((uint32_t)p[3] << 24); }
static uint8_t expand5(unsigned x) { return (uint8_t)((x << 3) | (x >> 2)); }
static uint8_t expand6(unsigned x) { return (uint8_t)((x << 2) | (x >> 4)); }

static void colors(const uint8_t *p, uint8_t out[16][4], int force_four) {
    uint16_t a = rd16(p), b = rd16(p + 2);
    uint8_t c[4][4] = {
        {expand5(a >> 11), expand6((a >> 5) & 63), expand5(a & 31), 255},
        {expand5(b >> 11), expand6((b >> 5) & 63), expand5(b & 31), 255},
        {0, 0, 0, 255}, {0, 0, 0, 255}
    };
    if (a > b || force_four) {
        for (int k = 0; k < 3; k++) {
            c[2][k] = (uint8_t)((2 * c[0][k] + c[1][k]) / 3);
            c[3][k] = (uint8_t)((c[0][k] + 2 * c[1][k]) / 3);
        }
    } else {
        for (int k = 0; k < 3; k++) c[2][k] = (uint8_t)((c[0][k] + c[1][k]) / 2);
        c[3][3] = 0;
    }
    uint32_t indices = rd32(p + 4);
    for (int i = 0; i < 16; i++) memcpy(out[i], c[(indices >> (2 * i)) & 3], 4);
}

static int clamp_snorm(int n) { return n < -127 ? -127 : n > 127 ? 127 : n; }
static uint8_t snorm_to_byte(int n) { return (uint8_t)(((clamp_snorm(n) + 127) * 255 + 127) / 254); }

static void channel(const uint8_t *p, uint8_t out[16], int is_signed) {
    int a = is_signed ? clamp_snorm((int8_t)p[0]) : p[0];
    int b = is_signed ? clamp_snorm((int8_t)p[1]) : p[1];
    int v[8] = {a, b, 0};
    if (a > b) {
        for (int j = 1; j <= 6; j++) v[j + 1] = ((7 - j) * a + j * b) / 7;
    } else {
        for (int j = 1; j <= 4; j++) v[j + 1] = ((5 - j) * a + j * b) / 5;
        v[6] = is_signed ? -127 : 0;
        v[7] = is_signed ? 127 : 255;
    }
    uint64_t bits = 0;
    for (int j = 0; j < 6; j++) bits |= (uint64_t)p[j + 2] << (j * 8);
    for (int i = 0; i < 16; i++) {
        int n = v[(bits >> (i * 3)) & 7];
        out[i] = is_signed ? snorm_to_byte(n) : (uint8_t)n;
    }
}

static int decode(unsigned bc, int signed_channels,
                  const uint8_t *blocks, size_t blocks_size,
                  uint32_t width, uint32_t height,
                  uint8_t *pixels, size_t pixels_size, unsigned components) {
    if (bc < 1 || (bc > 5 && bc != 7) || !width || !height || !blocks || !pixels ||
        (components != 4 && !(bc == 4 && components == 1) && !(bc == 5 && components == 2))) return -1;
    const size_t bw = ((size_t)width + 3) / 4, bh = ((size_t)height + 3) / 4;
    const size_t bytes = bc == 1 || bc == 4 ? 8 : 16;
    if (bw > SIZE_MAX / bh || bw * bh > SIZE_MAX / bytes ||
        blocks_size < bw * bh * bytes || (size_t)width > SIZE_MAX / height / components ||
        pixels_size < (size_t)width * height * components) return -1;
    for (size_t by = 0; by < bh; by++) for (size_t bx = 0; bx < bw; bx++) {
        const uint8_t *p = blocks + (by * bw + bx) * bytes;
        uint8_t texels[16][4], x[16], y[16];
        if (bc == 1) colors(p, texels, 0);
        if (bc == 7) bcdec_bc7(p, texels, 4 * 4);
        if (bc == 2 || bc == 3) {
            colors(p + 8, texels, 1);
            if (bc == 3) channel(p, x, 0);
            for (int i = 0; i < 16; i++)
                texels[i][3] = bc == 2 ? (uint8_t)(((p[i / 2] >> ((i & 1) * 4)) & 15) * 17) : x[i];
        }
        if (bc == 4 || bc == 5) {
            channel(p, x, signed_channels);
            if (bc == 5) channel(p + 8, y, signed_channels);
            for (int i = 0; i < 16; i++) {
                texels[i][0] = x[i]; texels[i][1] = bc == 5 ? y[i] : 0;
                texels[i][2] = 0; texels[i][3] = 255;
            }
        }
        for (unsigned iy = 0; iy < 4; iy++) for (unsigned ix = 0; ix < 4; ix++) {
            const size_t px = bx * 4 + ix, py = by * 4 + iy;
            if (px < width && py < height) memcpy(pixels + (py * width + px) * components, texels[iy * 4 + ix], components);
        }
    }
    return 0;
}


int nvn_bc_decode_rgba8(unsigned bc, int signed_channels,
                        const uint8_t *blocks, size_t blocks_size,
                        uint32_t width, uint32_t height,
                        uint8_t *rgba, size_t rgba_size) {
    return decode(bc, signed_channels, blocks, blocks_size, width, height, rgba, rgba_size, 4);
}

int nvn_bc_decode_tight(unsigned bc, int signed_channels,
                        const uint8_t *blocks, size_t blocks_size,
                        uint32_t width, uint32_t height,
                        uint8_t *pixels, size_t pixels_size) {
    unsigned components = bc == 4 ? 1 : bc == 5 ? 2 : 4;
    return decode(bc, signed_channels, blocks, blocks_size, width, height, pixels, pixels_size, components);
}
