# Xbox Series X port: use mkw-recomp as the game runtime

Upstream: https://github.com/adriadam10/mkw-recomp
Pinned revision: dabf868ba437182b6f51bc14186114b288429588
License: MIT (preserve upstream LICENSE and attribution).

## Architecture decision
Use upstream's AArch64-to-C static recompiler, not the experimental
`Arm64Cpu.h` interpreter. Upstream's generated C targets x86-64 and
already has Windows-native services in `runtime/platform/win32*.c`.
The generated C is **not** included in the repository: it must be
generated from the user's own decrypted `exefs/` dump.

The `romfs.img` reader currently in `dm/xbox` handles the *other*
port's DMROMFS1 resource container. It is not an executable loader and
cannot replace upstream's required `exefs/`.

## Xbox-specific implementation boundaries
1. **Platform**: adapt upstream `runtime/platform/win32*.c` for
   UWP/AppContainer APIs and permitted Xbox filesystem/memory APIs.
   Its Win32 desktop entry point cannot run unmodified on Xbox UWP.
2. **Graphics**: upstream `runtime/gfx/nvn/` is the game API layer;
   `runtime/gfx/vk/` is its Vulkan backend. Xbox UWP needs a D3D11/12
   backend with equivalent NVN2 semantics; the existing D3D11 probe
   only creates a swap chain and is not that backend.
3. **CPU**: run the recompiler's generated x64 code, not the ARM64
   interpreter. Generated code and shaders must be built from the
   owner's game dump on a compatible build host.
4. **Game setup**: adapt upstream launcher/setup to ingest the owner's
   `exefs/` and `romfs/` or supported upstream packed dump format.
5. **Packaging**: build a signed Xbox UWP package and validate Xbox
   Developer Mode API restrictions, memory limits and graphics features.

## Get the upstream source at the exact pinned revision
The GitHub Actions workflow `xbox-recomp-source.yml` fetches the pinned
MIT-licensed runtime/recompiler source and uploads a source artifact.
It does **not** compile game code, ship Nintendo files, or claim a
playable Xbox package.

The current Xbox app is not yet wired to this upstream runtime.
