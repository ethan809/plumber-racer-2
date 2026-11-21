#!/usr/bin/env bash
# Build Mario Kart World 1.0.0 for Apple Silicon from the user's own decrypted dump.
# Usage: bash scripts/build_mkw_mac.sh "/path/to/Mario Kart World"
set -euo pipefail

fail() { printf 'Error: %s\n' "$*" >&2; exit 1; }

repo_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
game_dir="${1:-}"
[[ -n "$game_dir" ]] || fail 'Supply the folder containing base/exefs and base/romfs.'
[[ -d "$game_dir" ]] || fail 'Supply an extracted game folder, not a single .img or .ipa file. RomFS.img alone is insufficient.'
game_dir="$(cd "$game_dir" && pwd)"

[[ "$(uname -s)" == Darwin && "$(uname -m)" == arm64 ]] || fail 'This build needs an Apple Silicon Mac.'
for required in base/exefs/main base/exefs/main.npdm base/exefs/subsdk0 base/exefs/sdk base/exefs/rtld base/romfs; do
  [[ -e "$game_dir/$required" ]] || fail "Game dump is missing $required. A RomFS image alone cannot build the Mac app."
done

build_id="$(xxd -s 0x40 -l 16 -p "$game_dir/base/exefs/main" | tr -d '\n')"
[[ "$build_id" == '9e244438c3d8ef1c703a6a85ef99c4f0' ]] || fail "The game is build $build_id, but this port requires version 1.0.0."

command -v brew >/dev/null || fail 'Install Homebrew first: https://brew.sh'
xcodebuild -version | grep -q 'Xcode 27.2' || fail 'Install and select Xcode 27.2 before building.'

available_kib="$(df -k "$game_dir" | awk 'NR == 2 {print $4}')"
[[ "$available_kib" =~ ^[0-9]+$ && "$available_kib" -ge 26214400 ]] || fail 'Keep at least 25 GiB free on the game drive for the native Mac build.'

printf 'Preparing Apple Silicon build tools...\n'
xcodebuild -downloadComponent MetalToolchain
brew install python@3.13 xcodegen zstd glslang
if ! command -v dotnet >/dev/null 2>&1; then
  brew install --cask dotnet-sdk
fi

venv_dir="$game_dir/.nxnative-mac-venv"
/opt/homebrew/opt/python@3.13/bin/python3.13 -m venv "$venv_dir"
"$venv_dir/bin/python" -m pip install lz4 'capstone>=5,<6' fonttools nsz zbic

printf 'Building Mario Kart World.app (this can take a long time on an M1)...\n'
"$venv_dir/bin/python" "$repo_dir/scripts/build_ipa_from_nsp.py" "$game_dir" --platform mac-app --gpu none

app="$game_dir/Mario Kart World.app"
[[ -d "$app" ]] || fail 'Build completed without producing Mario Kart World.app. Check the build logs in the game folder.'
printf '\nMac app ready: %s\n' "$app"
