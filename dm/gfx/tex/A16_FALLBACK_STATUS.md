# A16 BC fallback — experimental work in progress

This branch adds a portable CPU BC1/BC2/BC3/BC4/BC5/BC7 decoder
(`nvn_bc_decode.c`) and a standalone BC-to-ASTC command-line converter.
It does **not** yet make the game work on A16 devices.

## Completed

- Decodes already-deswizzled 4x4 BC1–BC5 and BC7 blocks to RGBA8 (including clipped edge blocks).
- BC4 and BC5 support both unsigned and signed normalized channels.
- Standalone tests for BC1, BC4, BC7, bounds, and unsupported formats.
- `scripts/bc_to_astc.py` turns an individually extracted **linear** BC texture into
  raw ASTC 4x4 blocks using ARM's `astcenc` command. Tested on a BC1 red test block.
- Optional `--cache-dir` writes content-addressed ASTC output and JSON metadata,
  including source SHA-256, format, dimensions and encoding options. A synthetic
  256x256 BC7 tile (64 KiB) successfully converts to a 64 KiB ASTC tile; the
  cache-hit path has been tested with `scripts/test_a16_astc_tile.py`.
- The converter now accepts **one complete NVN block-linear tile** using
  `--nvn-swizzled --block-height-log2 N` and `--switch2` if necessary.
  Tests check the tile byte permutation for both Switch 1 and Switch 2 GOB
  layouts before converting a synthetic 64 KiB tile to ASTC.
- Experimental `DM_NVN_BC_RGBA=1` opt-in for devices without BC support:
  `pkg_upload_` deswizzles and decodes ordinary, non-sparse BC1–BC5/BC7
  packaged textures into RGBA8 Metal textures. It is a stepping stone for
  correctness testing, **not** the desired ASTC path and not a playable port.
  BC6H, 3D, sparse and certain non-packaged GPU-write paths are unsupported.

## Try the single-texture converter

Install the official [ARM astcenc command-line encoder](https://github.com/ARM-software/astc-encoder/releases)
and make the executable available as `astcenc` (or pass its path with `--astcenc`).

```sh
python3 scripts/bc_to_astc.py --input linear.bc7 --output linear.astcraw \
  --width 1024 --height 1024 --format BC7 --srgb --cache-dir astc-cache

# One already-extracted, complete 256x256 block-linear BC7 tile:
python3 scripts/bc_to_astc.py --input switch-tile.bc7 --output tile.astcraw \
  --width 256 --height 256 --format BC7 --nvn-swizzled \
  --block-height-log2 1 --cache-dir astc-cache
```

The input must be raw BC blocks in row-major order (or one complete block-linear
NVN tile with the matching options), **not** a Switch texture container, an
image file, or the game's entire RomFS. The output is a raw
ASTC 4x4 block stream without the 16-byte `.astc` file header, suitable for
uploading to an `MTLPixelFormatASTC_4x4_*` texture after integration. Any mip
levels, array slices or cube faces need conversion separately. Preconversion
should include a manifest tying each result to the source texture identity,
dimensions, format and checksum so the host cannot associate the wrong blocks.
The cache files produced here are **not** yet connected to game texture loading.
The converter can deswizzle a complete tile, but does not resolve virtual
pages or determine the block height itself. Metal's sparse page shape may
differ from NVN's. An equally
sized ASTC byte stream is not proof that the page can be mapped unchanged.

To check the 64 KiB BC7 tile conversion and content cache on a development host:

```sh
python3 scripts/test_a16_astc_tile.py --astcenc /path/to/astcenc
```

## Required before enabling A16

1. Add an HDR-preserving BC6H path; BC6H cannot safely be converted to LDR RGBA8.
2. Identify and extract compressed textures from the game's containers or capture
   their linearized blocks during a reference run. **No generic build-time
   extractor is implemented.** A game can create or update textures at runtime.
3. Replace the high-memory RGBA8 bridge with an ASTC cache lookup and conversion
   in `pkg_upload_` and its sparse / CPU-write paths. Keep the original guest BC
   layout, selecting ASTC Metal destinations only when matching converted blocks exist.
4. Handle views, copies, sparse textures, dynamically updated BC textures,
   mipmaps, signed data and cache invalidation.
5. Profile memory and conversion overhead on A16; ASTC 4x4 costs 16 bytes per
   4x4 block (same as BC7; twice BC1), plus staging/transient storage.
6. Test on an A16 device. `DM_NVN_BC_RGBA=1` now bypasses the BC startup
   check experimentally; it must not be presented as validated compatibility.
   The renderer file passed an ARM64 iOS Metal syntax check in GitHub Actions,
   but no complete A16 iOS app has been built or tested.

An A16 build is **not** possible merely by removing that check: Metal rejects unsupported
BC pixel formats later during texture creation. Do not interpret this decoder as a playable port.

Run this platform-independent smoke test on any C compiler:

```sh
cc -std=c11 -Wall -Wextra -Werror dm/gfx/tex/nvn_bc_decode.c \
  dm/gfx/tex/tests/t_bc_decode.c -o /tmp/t_bc_decode
/tmp/t_bc_decode
```
