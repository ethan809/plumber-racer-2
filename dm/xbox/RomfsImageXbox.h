#pragma once
// Windows/UWP reader for the existing iOS DMROMFS1 romfs.img format.
// Uses the actual on-disk header/index format from dm/host/romfs_image.h.
#include <windows.h>
#include <cstdint>
#include <cstring>
#include <string>
#include <vector>
#include <limits>
namespace XboxRomfs {
#pragma pack(push,1)
struct Entry { uint64_t a,b; uint32_t name; uint16_t nlen; uint8_t type,flags; };
#pragma pack(pop)
static_assert(sizeof(Entry)==24,"DMROMFS1 index entry must be 24 bytes");
struct Image {
    HANDLE file=INVALID_HANDLE_VALUE;
    uint64_t indexOffset=0;
    std::vector<Entry> entries;
    std::vector<char> names;
    std::string error;
    ~Image(){close();}
    void close(){if(file!=INVALID_HANDLE_VALUE){CloseHandle(file);file=INVALID_HANDLE_VALUE;} entries.clear();names.clear();}
    bool read(uint64_t offset,void* buffer,size_t length) {
        if(file==INVALID_HANDLE_VALUE || length>0xFFFFFFFFull) return false;
        LARGE_INTEGER pos;pos.QuadPart=offset;
        if(!SetFilePointerEx(file,pos,nullptr,FILE_BEGIN))return false;
        DWORD got=0;
        return ReadFile(file,buffer,(DWORD)length,&got,nullptr) && got==length;
    }
    bool open(const std::wstring& path) {
        close();error.clear();
        CREATEFILE2_EXTENDED_PARAMETERS params={sizeof(params)};
        params.dwFileAttributes=FILE_ATTRIBUTE_NORMAL;
        file=CreateFile2(path.c_str(),GENERIC_READ,FILE_SHARE_READ,OPEN_EXISTING,&params);
        if(file==INVALID_HANDLE_VALUE){error="Cannot open romfs.img in app LocalFolder";return false;}
        LARGE_INTEGER length;
        uint8_t header[32];
        if(!GetFileSizeEx(file,&length)||length.QuadPart<32||!read(0,header,32)||memcmp(header,"DMROMFS1",8)){
            error="Not a DMROMFS1 image";close();return false;
        }
        uint64_t indexSize=0,count=0;
        memcpy(&indexOffset,header+8,8);
        memcpy(&indexSize,header+16,8);
        memcpy(&count,header+24,8);
        uint64_t total=(uint64_t)length.QuadPart;
        if(!count||count>UINT32_MAX||indexOffset<32||indexOffset>total||
           indexSize!=total-indexOffset||count>indexSize/sizeof(Entry)||
           indexSize>256ull*1024*1024) {
            error="Incomplete or oversized RomFS index";close();return false;
        }
        entries.resize((size_t)count);
        if(!read(indexOffset,entries.data(),entries.size()*sizeof(Entry))){
            error="Could not read RomFS entries";close();return false;
        }
        names.resize((size_t)(indexSize-count*sizeof(Entry)));
        if(!names.empty()&&!read(indexOffset+count*sizeof(Entry),names.data(),names.size())){
            error="Could not read RomFS names";close();return false;
        }
        uint64_t next=1;
        for(size_t i=0;i<entries.size();i++){
            const auto& e=entries[i];
            if(uint64_t(e.name)+e.nlen>names.size()||e.type>1||e.flags>(e.type?1:0)){
                error="Invalid RomFS entry";close();return false;
            }
            if(i==0&&(e.type||e.nlen)){error="Invalid RomFS root";close();return false;}
            if(i && (!e.nlen||e.nlen>0x300)){error="Invalid RomFS filename";close();return false;}
            if(e.type){
                if(e.a<32||e.a>indexOffset||e.b>indexOffset-e.a){
                    error="RomFS file outside image";close();return false;
                }
            }else{
                if(e.a!=next||e.a<=i||e.a>count||e.b>count-e.a){
                    error="Invalid RomFS directory tree";close();return false;
                }
                next+=e.b;
            }
        }
        if(next!=count){error="Incomplete RomFS directory tree";close();return false;}
        return true;
    }
    // Returns an index entry by slash-separated UTF-8 path, or nullptr.
    const Entry* find(const std::string& path) const {
        if(entries.empty())return nullptr;
        uint64_t current=0;size_t start=0;
        while(start<path.size()){
            size_t end=path.find('/',start);
            if(end==std::string::npos)end=path.size();
            const auto& dir=entries[(size_t)current];
            if(dir.type)return nullptr;
            uint64_t found=UINT64_MAX;
            for(uint64_t i=dir.a;i<dir.a+dir.b;i++){
                const auto& e=entries[(size_t)i];
                if(end-start==e.nlen && !memcmp(names.data()+e.name,path.data()+start,e.nlen)){
                    found=i;break;
                }
            }
            if(found==UINT64_MAX)return nullptr;
            current=found;start=end+1;
        }
        return &entries[(size_t)current];
    }
};
}
