#include "../nvn_bc_decode.h"
#include <assert.h>
#include <stdio.h>
#include <string.h>

int main(void) {
    uint8_t result[4 * 4 * 4];
    const uint8_t bc1_red[8] = {0x00, 0xf8, 0x00, 0x00, 0, 0, 0, 0};
    assert(!nvn_bc_decode_rgba8(1, 0, bc1_red, sizeof bc1_red, 4, 4, result, sizeof result));
    for (int i = 0; i < 16; i++) {
        assert(result[i * 4] == 255 && result[i * 4 + 1] == 0);
        assert(result[i * 4 + 2] == 0 && result[i * 4 + 3] == 255);
    }
    const uint8_t bc4_white[8] = {255, 0, 0, 0, 0, 0, 0, 0};
    assert(!nvn_bc_decode_rgba8(4, 0, bc4_white, sizeof bc4_white, 4, 4, result, sizeof result));
    for (int i = 0; i < 16; i++) assert(result[i * 4] == 255 && result[i * 4 + 3] == 255);
    uint8_t r8[16 + 1]; memset(r8, 0xaa, sizeof r8);
    assert(!nvn_bc_decode_tight(4, 0, bc4_white, sizeof bc4_white, 4, 4, r8, sizeof r8 - 1));
    for (int i = 0; i < 16; i++) assert(r8[i] == result[i * 4]);
    assert(r8[16] == 0xaa);
    const uint8_t bc5_rg[16] = {255, 0, 0, 0, 0, 0, 0, 0, 0, 255, 0, 0, 0, 0, 0, 0};
    uint8_t rg8[3 * 2 * 2 + 1], bc5_rgba[3 * 2 * 4]; memset(rg8, 0xaa, sizeof rg8);
    assert(!nvn_bc_decode_rgba8(5, 1, bc5_rg, sizeof bc5_rg, 3, 2, bc5_rgba, sizeof bc5_rgba));
    assert(!nvn_bc_decode_tight(5, 1, bc5_rg, sizeof bc5_rg, 3, 2, rg8, sizeof rg8 - 1));
    for (int i = 0; i < 6; i++) {
        assert(rg8[i * 2] == bc5_rgba[i * 4]);
        assert(rg8[i * 2 + 1] == bc5_rgba[i * 4 + 1]);
    }
    assert(rg8[sizeof rg8 - 1] == 0xaa);
    assert(nvn_bc_decode_tight(5, 0, bc5_rg, sizeof bc5_rg, 3, 2, rg8, 11) == -1);
    // Cropping at non-multiple-of-four dimensions must not touch the padding.
    uint8_t cropped[3 * 2 * 4 + 1]; memset(cropped, 0xaa, sizeof cropped);
    assert(!nvn_bc_decode_rgba8(1, 0, bc1_red, sizeof bc1_red, 3, 2, cropped, sizeof cropped - 1));
    assert(cropped[sizeof cropped - 1] == 0xaa);
    const uint8_t bc7_zero[16] = {0};
    assert(!nvn_bc_decode_rgba8(7, 0, bc7_zero, sizeof bc7_zero, 4, 4, result, sizeof result));
    assert(nvn_bc_decode_rgba8(6, 0, bc7_zero, sizeof bc7_zero, 4, 4, result, sizeof result) == -1);
    puts("BC1/4/5/7, tight R/RG storage, and bounds checks passed");
    return 0;
}
