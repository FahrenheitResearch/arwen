// WRF v4.7.1 phys/module_fr_fire_phys.F:1540-1670, with documented
// corrections for undefined chaparral base ROS and the mutated cap denominator.
#ifndef GPUWM_SFIRE_PHYS_CUH
#define GPUWM_SFIRE_PHYS_CUH
__device__ __forceinline__ int sfire_category(int code, int fallback)
{
    if (code == 0) code = fallback;
    if (code >= 1 && code <= 13) return code;
    if (code >= 101 && code <= 109) return code - 86;
    if (code >= 121 && code <= 124) return code - 97;
    if (code >= 141 && code <= 149) return code - 113;
    if (code >= 161 && code <= 165) return code - 124;
    if (code >= 181 && code <= 189) return code - 139;
    if (code >= 201 && code <= 204) return code - 150;
    return 14;
}

__device__ __forceinline__ void sfire_ros_device(
    float &ros_base, float &ros_wind, float &ros_slope,
    float propx, float propy, float vx, float vy, float dzdx, float dzdy,
    float bbb, float betafl, float phiwc, float r0, float ischap,
    int advection)
{
    float speed, tanphi, cor_wind, cor_slope;
    if (advection != 0) {
        speed = sqrtf(vx*vx + vy*vy) + 1.1754943508222875e-38f;
        tanphi = sqrtf(dzdx*dzdx + dzdy*dzdy) + 1.1754943508222875e-38f;
        cor_wind = fmaxf(0.0f, __fdiv_rn(vx*propx + vy*propy, speed));
        cor_slope = fmaxf(0.0f, __fdiv_rn(dzdx*propx + dzdy*propy, tanphi));
    } else {
        speed = vx*propx + vy*propy;
        tanphi = dzdx*propx + dzdy*propy;
        cor_wind = 1.0f;
        cor_slope = 1.0f;
    }
    if (!(ischap > 0.0f)) {
        float spdms = fmaxf(speed, 0.0f);
        float umidm = fminf(spdms, 30.0f);
        float umid = umidm * 196.850f;
        float phiw = gfk_pow(umid, bbb) * phiwc;
        float phis = 0.0f;
        if (tanphi > 0.0f)
            phis = 5.275f * gfk_pow(betafl, -0.3f) * (tanphi*tanphi);
        ros_base = r0 * 0.00508f;
        ros_wind = ros_base * phiw;
        ros_slope = ros_base * phis;
    } else {
        ros_base = 0.0f;
        ros_wind = fmaxf(1.2974f * gfk_pow(fmaxf(speed, 0.0f), 1.41f), 0.03333f);
        ros_slope = 0.0f;
    }
    ros_wind = ros_wind * cor_wind;
    ros_slope = ros_slope * cor_slope;
    float excess = ros_base + ros_wind + ros_slope - 6.0f;
    if (excess > 0.0f) {
        float denominator = ros_wind + ros_slope;
        if (denominator > 0.0f) {
            // Allocate the remaining budget without cancelling large rates.
            // The complementary term keeps both contributions nonnegative.
            ros_base=fminf(ros_base,6.0f);
            float available=fmaxf(6.0f-ros_base,0.0f);
            ros_wind=available*__fdiv_rn(ros_wind,denominator);
            ros_slope=available-ros_wind;
        } else {
            ros_base = 6.0f;
        }
    }
}
#endif
