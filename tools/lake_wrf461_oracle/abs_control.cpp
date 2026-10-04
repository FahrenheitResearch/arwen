// Compare the production ABS helper with the native Fortran intrinsic.
#include <cstdint>
#include <cstdio>
#include <cstring>
#include "lake_support.cuh"
int main() {
    const uint32_t single[] = {0x80000000u,0u,1u,0x80000001u,0x3f800000u,
        0xbf800000u,0x7f800000u,0xff800000u,0x7fc12345u,0xffc12345u,
        0x7f812345u,0xff812345u};
    const uint64_t dual[] = {0x8000000000000000ULL,0ULL,1ULL,
        0x8000000000000001ULL,0x3ff0000000000000ULL,0xbff0000000000000ULL,
        0x7ff0000000000000ULL,0xfff0000000000000ULL,0x7ff8123456789abcULL,
        0xfff8123456789abcULL,0x7ff0123456789abcULL,0xfff0123456789abcULL};
    for(int i=0;i<12;++i) {
        float f; double d;
        std::memcpy(&f,&single[i],sizeof f);
        std::memcpy(&d,&dual[i],sizeof d);
        f=lake_abs(f); d=lake_abs(d);
        uint32_t fo;uint64_t dout;
        std::memcpy(&fo,&f,sizeof f);std::memcpy(&dout,&d,sizeof d);
        std::printf("%08X %08X %016llX %016llX\n",single[i],fo,
                    static_cast<unsigned long long>(dual[i]),
                    static_cast<unsigned long long>(dout));
    }
}
