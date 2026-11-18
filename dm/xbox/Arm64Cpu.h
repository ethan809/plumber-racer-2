#pragma once
// ARM64 guest interpreter core for the Xbox x64 port.
// First execution backend: integer register operations and control flow.
// Unsupported opcodes trap; memory, SIMD, SVC and threading still need implementation.
#include <cstdint>
#include <cstddef>
#include <cstring>

namespace XboxA64 {
enum class Stop { Running, Halt, InvalidInstruction, OutOfBounds, StepLimit };
struct Cpu {
    uint64_t x[31] = {};
    uint64_t sp = 0, pc = 0;
    uint32_t nzcv = 0;
    Stop stop = Stop::Running;
};
struct Memory {
    const uint8_t* bytes = nullptr;
    size_t size = 0;
    uint64_t base = 0;
    bool fetch(uint64_t addr, uint32_t& out) const {
        if (!bytes || addr < base || (addr - base) > size ||
            size - static_cast<size_t>(addr - base) < 4) return false;
        std::memcpy(&out, bytes + (addr - base), 4);
        return true;
    }
};
inline int64_t signExtend(uint64_t n, unsigned bits) {
    const uint64_t mask = uint64_t(1) << (bits - 1);
    return static_cast<int64_t>((n ^ mask) - mask);
}
inline uint64_t reg(const Cpu& c, unsigned n) { return n == 31 ? 0 : c.x[n]; }
inline void write(Cpu& c, unsigned n, uint64_t value, bool sf) {
    if (n != 31) c.x[n] = sf ? value : uint32_t(value);
}
inline bool condition(const Cpu& c, unsigned cond) {
    const bool n = (c.nzcv >> 31) & 1, z = (c.nzcv >> 30) & 1;
    const bool carry = (c.nzcv >> 29) & 1, v = (c.nzcv >> 28) & 1;
    bool r = false;
    switch (cond >> 1) {
        case 0: r=z; break; case 1: r=carry; break; case 2: r=n; break;
        case 3: r=v; break; case 4: r=carry&&!z; break;
        case 5: r=n==v; break; case 6: r=!z&&(n==v); break;
        case 7: r=true; break;
    }
    return (cond & 1) && cond != 15 ? !r : r;
}
inline void step(Cpu& c, const Memory& m) {
    if (c.stop != Stop::Running) return;
    uint32_t op;
    if (!m.fetch(c.pc, op)) { c.stop=Stop::OutOfBounds; return; }
    const uint64_t next = c.pc + 4;
    const bool sf = (op >> 31) != 0;
    // NOP
    if (op == 0xD503201F) { c.pc=next; return; }
    // B / BL immediate
    if ((op & 0x7C000000u) == 0x14000000u) {
        if (op & 0x80000000u) c.x[30]=next;
        c.pc += signExtend(uint64_t(op & 0x03FFFFFFu) << 2, 28);
        return;
    }
    // RET / BR / BLR
    if ((op & 0xFFFFFC1Fu) == 0xD65F0000u ||
        (op & 0xFFFFFC1Fu) == 0xD61F0000u ||
        (op & 0xFFFFFC1Fu) == 0xD63F0000u) {
        unsigned rn=(op>>5)&31;
        if ((op & 0xFFFFFC1Fu) == 0xD63F0000u) c.x[30]=next;
        c.pc=reg(c,rn); return;
    }
    // B.cond
    if ((op & 0xFF000010u) == 0x54000000u) {
        c.pc=condition(c,op&15) ?
            c.pc+signExtend(uint64_t((op>>5)&0x7FFFFu)<<2,21) : next;
        return;
    }
    // CBZ / CBNZ (32 and 64-bit)
    if ((op & 0x7E000000u) == 0x34000000u) {
        uint64_t value=reg(c,op&31);
        if (!sf) value=uint32_t(value);
        bool zero=value==0, nonzero=(op>>24)&1;
        c.pc=(zero != nonzero) ?
            c.pc+signExtend(uint64_t((op>>5)&0x7FFFFu)<<2,21) : next;
        return;
    }
    // MOVZ / MOVN / MOVK, wide immediate
    if ((op & 0x1F800000u) == 0x12800000u) {
        unsigned opc=(op>>29)&3, shift=((op>>21)&3)*16;
        if (opc==1 || (!sf && shift>=32)) {c.stop=Stop::InvalidInstruction;return;}
        uint64_t mask=sf ? ~uint64_t(0) : uint64_t(0xFFFFFFFFu);
        uint64_t imm=uint64_t((op>>5)&0xFFFFu)<<shift;
        uint64_t value=opc==0 ? (~imm&mask) :
                       opc==2 ? imm : ((reg(c,op&31)&~(uint64_t(0xFFFF)<<shift))|imm);
        write(c,op&31,value,sf);c.pc=next;return;
    }
    // ADD/SUB immediate, including SP as source/destination when S=0
    if ((op & 0x1F000000u) == 0x11000000u) {
        bool sub=(op>>30)&1, flags=(op>>29)&1;
        unsigned rn=(op>>5)&31, rd=op&31;
        uint64_t lhs=rn==31?c.sp:reg(c,rn);
        uint64_t imm=(op>>10)&0xFFFu;
        if ((op>>22)&1) imm<<=12;
        uint64_t result=sub?lhs-imm:lhs+imm;
        if (!sf) result=uint32_t(result);
        if (flags) {
            const unsigned width=sf?64:32;
            uint64_t sign=uint64_t(1)<<(width-1);
            uint64_t rhs=sub?~imm:imm;
            bool carry=sub?(lhs>=imm):(result<lhs);
            bool overflow=((~(lhs^rhs)&(lhs^result)&sign)!=0);
            c.nzcv=uint32_t(((result&sign)?8:0)|((result==0)?4:0)|(carry?2:0)|(overflow?1:0))<<28;
        }
        if (rd==31 && !flags) c.sp=result;
        else write(c,rd,result,sf);
        c.pc=next;return;
    }
    c.stop=Stop::InvalidInstruction;
}
inline Stop run(Cpu& c, const Memory& m, uint64_t maxInstructions) {
    for (uint64_t i=0;i<maxInstructions && c.stop==Stop::Running;i++) step(c,m);
    return c.stop==Stop::Running ? Stop::StepLimit : c.stop;
}
} // namespace XboxA64
