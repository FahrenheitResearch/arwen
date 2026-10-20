/* SPDX-License-Identifier: Apache-2.0 */
#ifndef NEXTGEN_RAW_H
#define NEXTGEN_RAW_H
#include <stddef.h>
#include <stdint.h>
#ifdef __cplusplus
extern "C" {
#endif
typedef struct {
    const char *path, *sha256;
    uint8_t *destination;
    size_t bytes;
} nextgen_raw_read_item;
typedef struct {
    const char *path;
    const uint8_t *source;
    size_t bytes;
} nextgen_raw_write_item;
/* All calls: 0 success, 2 error; error is optional NUL-terminated text.
 * These are synchronous HOST-buffer operations for offline/setup use.
 * CUDA pointers must never be passed: use pinned host staging and CUDA copies.
 * Batch failures may leave other items completed. Discard a failed read batch.
 * New output files are never overwritten. Text arguments are UTF-8. */
int nextgen_raw_load(const char *path, const char *sha256, uint8_t *destination,
                     size_t bytes, char *error, size_t error_cap);
/* Packed-file range: sha256 seals the whole file. No alignment restriction.
 * Reads/hashes the whole file per call; invalid seal/range copies no bytes.
 * Module owners supply offsets/layout; this function performs no decoding. */
int nextgen_raw_load_slice(const char *path, const char *sha256, size_t offset,
                           uint8_t *destination, size_t bytes,
                           char *error, size_t error_cap);
int nextgen_raw_store(const char *new_path, const uint8_t *source, size_t bytes,
                      char *error, size_t error_cap);
int nextgen_raw_sha256(const char *path, char digest[65], char *error,
                       size_t error_cap);
int nextgen_raw_read_many(const nextgen_raw_read_item *items, size_t count,
                          char *error, size_t error_cap);
int nextgen_raw_write_many(const nextgen_raw_write_item *items, size_t count,
                           char *error, size_t error_cap);
/* Optional Rust-owned pageable HOST buffers. Output is NULL on error; on success
 * it is non-null even for zero bytes. Output must not already own a buffer.
 * Use existing nextgen_raw_store to save. Await CUDA copies before freeing;
 * these pointers are not CUDA-pinned. Free through this same library only.
 * free(NULL) succeeds; non-live addresses fail. Never reuse stale pointers:
 * a later allocation could recycle the address of an earlier allocation. */
int nextgen_raw_alloc(size_t bytes, uint8_t **output,
                      char *error, size_t error_cap);
int nextgen_raw_load_owned(const char *path, const char *sha256, size_t bytes,
                           uint8_t **output, char *error, size_t error_cap);
int nextgen_raw_free(uint8_t *pointer, char *error, size_t error_cap);
#ifdef __cplusplus
}
#endif
#endif
