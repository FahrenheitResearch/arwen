#ifndef ARWEN_HYDROFABRIC_H
#define ARWEN_HYDROFABRIC_H
#include <stddef.h>
#include <stdint.h>
#ifdef __cplusplus
extern "C" {
#endif
uint32_t arwen_hydrofabric_abi_version(void); /* remains 1 */
const char *arwen_hydrofabric_source_revision(void); /* library-owned */
/* Existing raw transport: caller supplies the expected SHA and byte count. */
int32_t arwen_hydrofabric_read_into(const char *path, const char *sha256,
                                  void *destination, size_t length);
/* Manifest-backed field transport. All descriptors validated in Rust; selected
 * path/length/hash checked before writing. No dtype conversion or state init.
 * Destination is unchanged on failure. 0 success, 1 bad pointer/UTF8,
 * 2 validation/I/O error, 3 panic. Strings NUL-terminated; destination writable
 * for length bytes. Startup-only; allocation/binding belongs to driver. */
int32_t arwen_hydrofabric_manifest_read_into(const char *manifest_path,
    const char *field_name, void *destination, size_t length);
int32_t arwen_hydrofabric_verify_memory(const char *sha256,
                                      const void *data, size_t length);
#ifdef __cplusplus
}
#endif
#endif
