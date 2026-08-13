#ifndef REGION_GLOBAL_DEALIAS_H
#define REGION_GLOBAL_DEALIAS_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

#ifndef BW_API
#define BW_API
#endif

#define BW_ABI_VERSION 1u
#define BW_RIFT_API_VERSION 1u

#define BW_OK 0
#define BW_ERR_NULL (-1)
#define BW_ERR_DIMS (-2)
#define BW_ERR_ALIGN (-3)
#define BW_ERR_VERSION (-4)
#define BW_ERR_LENGTH (-5)
#define BW_ERR_LIMIT (-6)
#define BW_ERR_REFERENCE (-7)
#define BW_ERR_ALIAS (-8)
#define BW_ERR_GEOMETRY (-9)

#define BW_RIFT_REF_CALLER 0u
#define BW_RIFT_REF_TEMPORAL 1u
#define BW_RIFT_REF_VERTICAL 2u
#define BW_RIFT_REF_ENVIRONMENTAL 3u

#define BW_RIFT_FLAG_DISABLE_AUTOMATIC_SINGLE_SWEEP (1u << 0)

#define BW_RIFT_REASON_RESIDUE_TRIGGER (1u << 0)
#define BW_RIFT_REASON_BRANCH_UNSTABLE (1u << 1)
#define BW_RIFT_REASON_TEMPORAL_ANCHOR (1u << 2)
#define BW_RIFT_REASON_VERTICAL_ANCHOR (1u << 3)
#define BW_RIFT_REASON_ENVIRONMENTAL_ANCHOR (1u << 4)
#define BW_RIFT_REASON_CALLER_ANCHOR (1u << 5)
#define BW_RIFT_REASON_VORTEX_PROPOSAL (1u << 6)
#define BW_RIFT_REASON_FUSION_ACCEPTED (1u << 7)
#define BW_RIFT_REASON_CONFLICTING_REFERENCES (1u << 8)
#define BW_RIFT_REASON_LOW_COVERAGE (1u << 9)
#define BW_RIFT_REASON_ABSTAINED (1u << 10)
#define BW_RIFT_REASON_BUDGET_EXCEEDED (1u << 11)
#define BW_RIFT_REASON_NYQUIST_TRANSITION (1u << 12)

typedef struct BwStats {
    uint32_t gates_total;
    uint32_t gates_finite;
    uint32_t gates_modified;
    uint32_t max_abs_fold;
    uint32_t wraps;
} BwStats;

/*
 * Set struct_size to sizeof(BwRiftOptionsV1). A null options pointer selects
 * reference-only defaults. Automatic single-sweep refinement requires finite
 * first_gate_m >= 0 and gate_spacing_m > 0.
 */
typedef struct BwRiftOptionsV1 {
    uint32_t struct_size;
    uint32_t flags;
    uint32_t max_abs_fold;
    uint32_t max_rois;
    uint32_t max_roi_gates;
    uint32_t max_total_roi_gates;
    uint32_t min_confidence;
    float first_gate_m;
    float gate_spacing_m;
    uint32_t reserved[3];
} BwRiftOptionsV1;

typedef struct BwRiftStatsV1 {
    uint32_t struct_size;
    uint32_t gates_total;
    uint32_t gates_finite;
    uint32_t gates_modified;
    uint32_t max_abs_fold;
    uint32_t wraps;
    uint32_t rois_detected;
    uint32_t rois_solved;
    uint32_t rois_accepted;
    uint32_t gates_refined;
    uint32_t gates_ambiguous;
    uint32_t budget_aborts;
    uint32_t reason_flags;
    uint32_t abstain_flags;
    uint32_t reserved[2];
} BwRiftStatsV1;

BW_API uint32_t bw_abi_version(void);
BW_API uint32_t bw_rift_api_version(void);
BW_API uint8_t *bw_alloc(size_t len);
BW_API void bw_free(uint8_t *ptr, size_t len);

/*
 * All velocity arrays are row-major f32 values in m/s. azim and nyq contain
 * one value per ray. out must not alias any input. stats may be null.
 */
BW_API int32_t bw_dealias(const float *vel,
                          const float *azim,
                          const float *nyq,
                          size_t rows,
                          size_t gates,
                          float *out,
                          BwStats *stats);

/*
 * reference_velocity is reference_count consecutive rows*gates fields.
 * reference_quality may be null or have the same flattened length;
 * reference_kind has reference_count entries. reflectivity, spectrum_width,
 * rho_hv, out_folds, out_confidence, out_reasons, and stats may independently
 * be null. out_velocity is required. All writable ranges must be disjoint from
 * every input and from each other. reference_count is limited to four.
 */
BW_API int32_t bw_dealias_rift_v1(const float *vel,
                                  const float *azim,
                                  const float *nyq,
                                  size_t rows,
                                  size_t gates,
                                  const float *reference_velocity,
                                  const uint8_t *reference_quality,
                                  const uint8_t *reference_kind,
                                  uint32_t reference_count,
                                  const float *reflectivity,
                                  const float *spectrum_width,
                                  const float *rho_hv,
                                  const BwRiftOptionsV1 *options,
                                  float *out_velocity,
                                  int8_t *out_folds,
                                  uint8_t *out_confidence,
                                  uint16_t *out_reasons,
                                  BwRiftStatsV1 *stats);

#ifdef __cplusplus
}
static_assert(sizeof(BwStats) == 20, "unexpected BwStats layout");
static_assert(sizeof(BwRiftOptionsV1) == 48, "unexpected BwRiftOptionsV1 layout");
static_assert(sizeof(BwRiftStatsV1) == 64, "unexpected BwRiftStatsV1 layout");
#elif defined(__STDC_VERSION__) && __STDC_VERSION__ >= 201112L
_Static_assert(sizeof(BwStats) == 20, "unexpected BwStats layout");
_Static_assert(sizeof(BwRiftOptionsV1) == 48, "unexpected BwRiftOptionsV1 layout");
_Static_assert(sizeof(BwRiftStatsV1) == 64, "unexpected BwRiftStatsV1 layout");
#endif

#endif
