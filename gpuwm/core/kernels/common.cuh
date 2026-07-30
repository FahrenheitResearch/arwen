// gpuwm/core/kernels/common.cuh
#pragma once
#define I3(k, j, i, ny, nx)  ((size_t)(k) * (ny) * (nx) + (size_t)(j) * (nx) + (i))
#define I3S(k, j, i, ny, nxs)  ((size_t)(k) * (ny) * (nxs) + (size_t)(j) * (nxs) + (i))
#define IDX3(k, j, i)  I3(k, j, i, ny, nx)
#define PERIODIC(i, n) ((((i) % (n)) + (n)) % (n))
typedef float real;
