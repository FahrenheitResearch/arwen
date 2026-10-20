// Rasterize a 2D `f64` grid into a packed-RGBA8 image, mirroring
// `rustwx_render::rasterize::rasterize_grid` byte-for-byte where the inputs
// are finite, and `LeveledColormap::map` byte-for-byte for the colormap
// lookup.
//
// One thread per output pixel. Output is stored as `unsigned int` per pixel
// in little-endian RGBA byte layout (R = bits 0..8, G = 8..16, B = 16..24,
// A = 24..32) so reinterpreting the buffer as a byte array gives the same
// memory layout as `image::Rgba<u8>`.

__device__ inline unsigned int colormap_lookup(
    double value,
    const double* __restrict__ levels,
    int n_levels,                       // n_intervals + 1
    const unsigned int* __restrict__ colors,
    int n_intervals,                    // n_levels - 1
    int has_under, unsigned int under_color,
    int has_over,  unsigned int over_color,
    int has_mask_below, double mask_below
) {
    // NaN -> transparent. Matches `LeveledColormap::map` first branch.
    if (isnan(value)) return 0u;

    // mask_below cutoff -> transparent.
    if (has_mask_below && value < mask_below) return 0u;

    // Empty colormap is impossible here (caller guarantees), but guard anyway.
    if (n_levels < 1 || n_intervals < 1) return 0u;

    // Below first level boundary.
    if (value < levels[0]) {
        return has_under ? under_color : 0u;
    }

    // partition_point(|l| *l <= value): first index where levels[i] > value.
    int lo = 0;
    int hi = n_levels;
    while (lo < hi) {
        int mid = (lo + hi) >> 1;
        if (levels[mid] <= value) lo = mid + 1;
        else                       hi = mid;
    }
    int idx = lo;

    // CPU code: `if idx <= n_intervals { colors[saturating_sub(1).min(...)] }`.
    // After the `value < levels[0]` early-out idx >= 1.
    if (idx <= n_levels - 1) {
        int ci = idx - 1;
        if (ci < 0) ci = 0;
        if (ci > n_intervals - 1) ci = n_intervals - 1;
        return colors[ci];
    }

    // value >= last level boundary.
    return has_over ? over_color : colors[n_intervals - 1];
}

__device__ inline double bilinear_sample(
    double v00, double v10, double v01, double v11,
    double fx, double fy
) {
    bool all_finite = isfinite(v00) && isfinite(v10)
                   && isfinite(v01) && isfinite(v11);
    if (all_finite) {
        if (__double_as_longlong(v00) == __double_as_longlong(v10)
            && __double_as_longlong(v00) == __double_as_longlong(v01)
            && __double_as_longlong(v00) == __double_as_longlong(v11)) return v00;
        double south = v00 * (1.0 - fx) + v10 * fx;
        double north = v01 * (1.0 - fx) + v11 * fx;
        return south * (1.0 - fy) + north * fy;
    }
    // CPU fallback: return the first finite corner, else NaN.
    if (isfinite(v00)) return v00;
    if (isfinite(v10)) return v10;
    if (isfinite(v01)) return v01;
    if (isfinite(v11)) return v11;
    return __longlong_as_double(0x7ff8000000000000LL);
}

// The CPU's fill and edge feather use the same input for every pixel. Evaluate
// neighbour alpha from that input so feathering needs no second image or launch.
__device__ inline unsigned int sample_pixel(
    int px, int py, const double* data, int ny, int nx,
    const double* levels, int n_levels, const unsigned int* colors, int n_colors,
    int has_under, unsigned int under_color, int has_over, unsigned int over_color,
    int has_mask, double mask_below, int width, int height,
    int axes_mode, double x0, double y0, double dx, double dy, int nearest
) {
    double gx, gy;
    if (axes_mode) {
        gx = ((double)px - x0) / dx;
        gy = ((double)py - y0) / dy;
        if (!isfinite(gx) || !isfinite(gy) || gx < 0 || gy < 0
            || gx > nx - 1 || gy > ny - 1) return 0;
    } else {
        gx = (double)px / (double)max(width-1, 1) * (double)max(nx-1, 1);
        gy = (double)(height-1-py) / (double)max(height-1, 1) * (double)max(ny-1, 1);
    }
    int i0 = (int)floor(gx), j0 = (int)floor(gy);
    int i1 = min(i0+1, nx-1), j1 = min(j0+1, ny-1);
    double fx = (i0 == i1 && axes_mode) ? 0.0 : gx - (double)i0;
    double fy = (j0 == j1 && axes_mode) ? 0.0 : gy - (double)j0;
    double value;
    if (nearest) {
        int i = min((int)round(gx), nx-1);
        int j = min((int)round(gy), ny-1);
        if (axes_mode) { i = fx < 0.5 ? i0 : i1; j = fy < 0.5 ? j0 : j1; }
        value = data[j*nx+i];
    } else {
        if (axes_mode) {
            int ci = min(i0, nx-2), cj = min(j0, ny-2);
            if (!isfinite(data[cj*nx+ci]) || !isfinite(data[cj*nx+ci+1])
                || !isfinite(data[(cj+1)*nx+ci]) || !isfinite(data[(cj+1)*nx+ci+1])) return 0;
        }
        value = bilinear_sample(data[j0*nx+i0], data[j0*nx+i1],
            data[j1*nx+i0], data[j1*nx+i1], fx, fy);
    }
    unsigned int color = colormap_lookup(value, levels, n_levels, colors, n_colors,
        has_under, under_color, has_over, over_color, has_mask, mask_below);
    return axes_mode && (color >> 24) == 0 ? 0 : color;
}

// Structured native map fill. One thread owns one pixel, with no atomics.
// axes_mode=0 matches rasterize_grid; 1 includes projected edge feathering.
extern "C" __global__ __launch_bounds__(256)
void rasterize_structured(
    const double* data, int ny, int nx,
    const double* levels, int n_levels, const unsigned int* colors, int n_colors,
    int has_under, unsigned int under_color, int has_over, unsigned int over_color,
    int has_mask, double mask_below, unsigned int* out, int width, int height,
    int axes_mode, double x0, double y0, double dx, double dy, int nearest
) {
    int pixel = blockIdx.x * blockDim.x + threadIdx.x;
    if (pixel >= width * height) return;
    int px = pixel % width, py = pixel / width;
    unsigned int color = sample_pixel(px, py, data, ny, nx, levels, n_levels,
        colors, n_colors, has_under, under_color, has_over, over_color,
        has_mask, mask_below, width, height, axes_mode, x0, y0, dx, dy, nearest);
    if (axes_mode && width >= 2 && height >= 2 && (color >> 24) > 216) {
        for (int sy = -1; sy <= 1; ++sy) {
            for (int sx = -1; sx <= 1; ++sx) {
                if (sx == 0 && sy == 0) continue;
                int ax = px + sx, ay = py + sy;
                bool transparent = ax < 0 || ay < 0 || ax >= width || ay >= height;
                if (!transparent) {
                    unsigned int neighbour = sample_pixel(ax, ay, data, ny, nx,
                        levels, n_levels, colors, n_colors, has_under, under_color,
                        has_over, over_color, has_mask, mask_below, width, height,
                        axes_mode, x0, y0, dx, dy, nearest);
                    transparent = (neighbour >> 24) == 0;
                }
                if (transparent) {
                    out[pixel] = (color & 0x00ffffffU) | (216U << 24);
                    return;
                }
            }
        }
    }
    out[pixel] = color;
}

// Robinson inverse helpers seeded from Drew's production CUDA renderer.
__device__ const double INV_R_EARTH = 6370000.0;
__device__ const double INV_DEG2RAD = 3.14159265358979323846264338327950288 / 180.0;
__device__ const double INV_RAD2DEG = 180.0 / 3.14159265358979323846264338327950288;

__device__ __constant__ double ROBINSON_X_INV[19] = {
    1.0000, 0.9986, 0.9954, 0.9900, 0.9822, 0.9730, 0.9600, 0.9427, 0.9216, 0.8962, 0.8679, 0.8350,
    0.7986, 0.7597, 0.7186, 0.6732, 0.6213, 0.5722, 0.5322
};

__device__ __constant__ double ROBINSON_Y_INV[19] = {
    0.0000, 0.0620, 0.1240, 0.1860, 0.2480, 0.3100, 0.3720, 0.4340, 0.4958, 0.5571, 0.6176, 0.6769,
    0.7346, 0.7903, 0.8435, 0.8936, 0.9394, 0.9761, 1.0000
};

__device__ inline double inv_stabilize_latitude(double lat_deg) {
    return fmin(89.999, fmax(-89.999, lat_deg));
}

__device__ inline double inv_normalize_projection_lon(double lon_deg) {
    double lon = fmod(lon_deg, 360.0);
    if (lon > 180.0) lon -= 360.0;
    else if (lon <= -180.0) lon += 360.0;
    return lon;
}

__device__ inline double inv_normalize_grid_lon(double lon_deg) {
    while (lon_deg < -180.0) lon_deg += 360.0;
    while (lon_deg >= 180.0) lon_deg -= 360.0;
    return lon_deg;
}

__device__ inline double inv_rem_euclid(double value, double modulus) {
    double r = fmod(value, modulus);
    if (r < 0.0) r += modulus;
    return r;
}

__device__ inline bool inv_clip_contains(
    int has_clip,
    double west_deg,
    double east_deg,
    double south_deg,
    double north_deg,
    double lat_deg,
    double lon_deg
) {
    if (!has_clip) return true;
    if (!isfinite(lat_deg) || !isfinite(lon_deg)) return false;
    if (lat_deg < south_deg || lat_deg > north_deg) return false;

    double raw_span = fabs(east_deg - west_deg);
    double span;
    if (raw_span >= 359.0) {
        span = fmin(raw_span, 360.0);
    } else {
        double west = inv_normalize_projection_lon(west_deg);
        double east = inv_normalize_projection_lon(east_deg);
        span = (west <= east) ? (east - west) : (east + 360.0 - west);
    }
    if (span >= 359.0) return true;

    double west = inv_normalize_projection_lon(west_deg);
    double east = inv_normalize_projection_lon(east_deg);
    double lon = inv_normalize_projection_lon(lon_deg);
    if (west <= east) return lon >= west && lon <= east;
    return lon >= west || lon <= east;
}

__device__ inline bool inv_unproject(
    int kind,
    double p0, double p1, double p2, double p3, double p4, double p5,
    double x,
    double y,
    double* lat_out,
    double* lon_out
) {
    (void)p4;
    (void)p5;

    if (!isfinite(x) || !isfinite(y)) return false;

    if (kind == 0) {
        // Geographic: p0 = central meridian in degrees.
        *lat_out = inv_stabilize_latitude(y);
        *lon_out = inv_normalize_projection_lon(x + p0);
        return true;
    }

    if (kind == 1) {
        // Robinson: p0 = central meridian in degrees.
        double scaled_y = fabs(y / (INV_R_EARTH * 1.3523));
        if (scaled_y > 1.0 + 1.0e-9) return false;
        int band = 17;
        for (int idx = 0; idx < 18; ++idx) {
            if (scaled_y <= ROBINSON_Y_INV[idx + 1] + 1.0e-12) {
                band = idx;
                break;
            }
        }
        double y0 = ROBINSON_Y_INV[band];
        double y1 = ROBINSON_Y_INV[band + 1];
        double t = 0.0;
        if (fabs(y1 - y0) > 1.0e-12) {
            t = fmin(1.0, fmax(0.0, (scaled_y - y0) / (y1 - y0)));
        }
        double lat_abs = ((double)band + t) * 5.0;
        double x_scale = ROBINSON_X_INV[band] + (ROBINSON_X_INV[band + 1] - ROBINSON_X_INV[band]) * t;
        if (x_scale <= 0.0) return false;
        double lon_delta = x / (INV_R_EARTH * 0.8487 * x_scale) * INV_RAD2DEG;
        if (fabs(lon_delta) > 180.0 + 1.0e-6) return false;
        *lat_out = copysign(lat_abs, y);
        *lon_out = inv_normalize_projection_lon(p0 + lon_delta);
        return true;
    }

    return false;
}

__device__ inline bool inv_grid_x_for_axis_lon(
    double lon,
    int nx,
    double lon0,
    double lon_step,
    int periodic_lon,
    double period_points,
    double* gx_out
) {
    if (periodic_lon) {
        *gx_out = inv_rem_euclid((lon - lon0) / lon_step, period_points);
        return true;
    }

    double adjusted = inv_normalize_grid_lon(lon);
    double axis_center = lon0 + lon_step * (double)(nx - 1) / 2.0;
    while (adjusted - axis_center > 180.0) adjusted -= 360.0;
    while (adjusted - axis_center < -180.0) adjusted += 360.0;

    double gx = (adjusted - lon0) / lon_step;
    if (gx < 0.0 || gx > (double)(nx - 1)) return false;
    *gx_out = gx;
    return true;
}

// q: regular axes[6], central meridian, extent[4], clip-present + bounds[4].
extern "C" __global__ __launch_bounds__(256)
void rasterize_robinson(
    const double* data, int ny, int nx,
    const double* levels, int nl, const unsigned int* colors, int nc,
    int hu, unsigned int under, int ho, unsigned int over, int hm, double mask,
    unsigned int* out, int w, int h, const double* q, int nearest
) {
    int pixel = blockIdx.x * blockDim.x + threadIdx.x;
    if (pixel >= w*h) return;
    out[pixel] = 0u;
    int px = pixel % w, py = pixel / w;
    double x = q[7] + ((double)px / ((w > 1) ? w-1 : 1)) * (q[8]-q[7]);
    double y = q[10] - ((double)py / ((h > 1) ? h-1 : 1)) * (q[10]-q[9]);
    double lat, lon;
    if (!inv_unproject(1, q[6], 0, 0, 0, 0, 0, x, y, &lat, &lon)) return;
    if (!inv_clip_contains((int)q[11], q[12], q[13], q[14], q[15], lat, lon)) return;
    double gy = (lat-q[0])/q[1], gx;
    if (gy < 0 || gy > ny-1) return;
    if (!inv_grid_x_for_axis_lon(lon, nx, q[2], q[3], (int)q[4], q[5], &gx)) return;
    double value;
    if (nearest) {
        int i = (q[4] != 0) ? (int)inv_rem_euclid(round(gx), q[5]) : min((int)round(gx),nx-1);
        int j = min((int)round(gy),ny-1);
        value = data[j*nx+i];
    } else {
        int i0 = min((int)floor(gx),nx-1), j0=(int)floor(gy);
        int i1 = (q[4] != 0) ? (int)inv_rem_euclid((double)(i0+1),q[5]) : min(i0+1,nx-1);
        int j1 = min(j0+1,ny-1);
        value = bilinear_sample(data[j0*nx+i0],data[j0*nx+i1],data[j1*nx+i0],data[j1*nx+i1],gx-i0,gy-j0);
    }
    out[pixel] = colormap_lookup(value,levels,nl,colors,nc,hu,under,ho,over,hm,mask);
}
