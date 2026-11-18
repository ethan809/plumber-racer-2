#pragma once
// Discover the user's unpacked mkw-recomp game dump in Xbox app storage.
// mkw-recomp needs executable exefs AND resource romfs, not DMROMFS1 alone.
#include <windows.h>
#include <string>
namespace XboxRecomp {
struct Dump {
    std::wstring root;
    bool exefs=false, romfs=false, main=false;
    bool ready() const { return exefs && romfs && main; }
};
inline bool directory(const std::wstring& p) {
    DWORD a=GetFileAttributesW(p.c_str());
    return a!=INVALID_FILE_ATTRIBUTES && (a&FILE_ATTRIBUTE_DIRECTORY)!=0;
}
inline bool file(const std::wstring& p) {
    DWORD a=GetFileAttributesW(p.c_str());
    return a!=INVALID_FILE_ATTRIBUTES && (a&FILE_ATTRIBUTE_DIRECTORY)==0;
}
inline Dump discover(const std::wstring& appFolder) {
    Dump d;
    // Device Portal may place a dump under LocalFolder/game or LocalFolder.
    for(const auto& candidate : {appFolder+L"\\game", appFolder}) {
        if(!directory(candidate+L"\\exefs") || !directory(candidate+L"\\romfs"))
            continue;
        d.root=candidate;
        d.exefs=true;
        d.romfs=true;
        d.main=file(candidate+L"\\exefs\\main");
        return d;
    }
    d.root=appFolder+L"\\game";
    d.exefs=directory(d.root+L"\\exefs");
    d.romfs=directory(d.root+L"\\romfs");
    d.main=file(d.root+L"\\exefs\\main");
    return d;
}
inline void log(const Dump& d) {
    if(d.ready()) {
        OutputDebugStringA("Xbox recomp: exefs/main and romfs found; game dump is ready for recompilation.\n");
    } else {
        OutputDebugStringA("Xbox recomp: game dump incomplete. Expected LocalFolder/game/exefs/main and LocalFolder/game/romfs/.\n");
        if(!d.exefs)OutputDebugStringA("Xbox recomp: exefs directory missing.\n");
        if(!d.main)OutputDebugStringA("Xbox recomp: exefs/main executable missing.\n");
        if(!d.romfs)OutputDebugStringA("Xbox recomp: romfs directory missing.\n");
    }
}
}
