#pragma once
// Xbox game boot-file discovery, using the same RomFS marker as the iOS host.
// This checks extracted user-supplied game data; it does not emulate Switch 2 CPU code.
#include <windows.h>
#include <string>
#include <vector>

namespace XboxGame {
struct BootFiles {
    std::wstring root;
    std::wstring romfs;
    std::wstring executable;
    bool hasRomfs = false;
    bool hasExecutable = false;
};
inline bool Exists(const std::wstring& path) {
    const DWORD a = GetFileAttributesW(path.c_str());
    return a != INVALID_FILE_ATTRIBUTES;
}
inline std::wstring Join(const std::wstring& a, const wchar_t* b) {
    return a + L"\\" + b;
}
inline BootFiles Discover(const std::wstring& root) {
    BootFiles result{};
    result.root = root;
    // The current iOS host expects an extracted RomFS tree with
    // System/Resource present for Mario game profiles.
    for (const auto& candidate : {Join(root,L"romfs"), root}) {
        if (Exists(Join(candidate,L"System\\Resource"))) {
            result.romfs = candidate;
            result.hasRomfs = true;
            break;
        }
    }
    // Switch 2 executable layout is still being researched; don't assume
    // a particular game binary or treat RomFS as executable.
    for (const auto& name : {L"main", L"main.nso", L"main.nro"}) {
        auto candidate = Join(root,name);
        if (Exists(candidate)) {
            result.executable = candidate;
            result.hasExecutable = true;
            break;
        }
    }
    return result;
}
} // namespace XboxGame
