// Flat C ABI over fpv::Simulator so any language (ctypes, C#, Rust...) can drive the core.
#include "fpv/simulator.hpp"

#if defined(_WIN32)
#define FPV_API extern "C" __declspec(dllexport)
#else
#define FPV_API extern "C" __attribute__((visibility("default")))
#endif

using fpv::Simulator;

FPV_API int fpv_abi_version() { return 2; }
FPV_API int fpv_config_size() { return (int)sizeof(fpv::FpvConfig); }
FPV_API int fpv_state_size() { return (int)sizeof(fpv::FpvState); }
FPV_API int fpv_obstacle_size() { return (int)sizeof(fpv::FpvObstacle); }
FPV_API void fpv_default_config(fpv::FpvConfig* out) { *out = fpv::default_config(); }

FPV_API void* fpv_create(const fpv::FpvConfig* cfg) { return new Simulator(*cfg); }
FPV_API void fpv_destroy(void* h) { delete static_cast<Simulator*>(h); }
FPV_API void fpv_reset(void* h) { static_cast<Simulator*>(h)->reset(); }
FPV_API void fpv_set_pose(void* h, double x, double y, double z, double yaw) {
    static_cast<Simulator*>(h)->set_pose(x, y, z, yaw);
}
FPV_API void fpv_set_config(void* h, const fpv::FpvConfig* cfg) { static_cast<Simulator*>(h)->set_config(*cfg); }
FPV_API void fpv_get_config(void* h, fpv::FpvConfig* out) { *out = static_cast<Simulator*>(h)->config(); }
FPV_API void fpv_set_sticks(void* h, double r, double p, double y, double t) {
    static_cast<Simulator*>(h)->set_sticks(r, p, y, t);
}
FPV_API void fpv_set_armed(void* h, int a) { static_cast<Simulator*>(h)->set_armed(a != 0); }
FPV_API void fpv_set_mode(void* h, int m) { static_cast<Simulator*>(h)->set_mode(m); }
FPV_API void fpv_set_obstacles(void* h, const fpv::FpvObstacle* o, int n) {
    static_cast<Simulator*>(h)->set_obstacles(o, n);
}
FPV_API int fpv_advance(void* h, double dt) { return static_cast<Simulator*>(h)->advance(dt); }
FPV_API void fpv_get_state(void* h, fpv::FpvState* out) { static_cast<Simulator*>(h)->fill_state(*out); }
FPV_API void fpv_set_terrain(void* h, const float* data, int n, double cell, double ox, double oy) {
    static_cast<Simulator*>(h)->set_terrain(data, n, cell, ox, oy);
}
FPV_API double fpv_ground_height(void* h, double x, double y) {
    return static_cast<Simulator*>(h)->ground_height(x, y);
}
