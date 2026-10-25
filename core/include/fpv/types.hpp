#pragma once
// Math primitives and POD structs shared with the C ABI (see c_api.cpp).
#include <cmath>
#include <cstdint>

namespace fpv {

constexpr double kPi = 3.14159265358979323846;
constexpr double kG = 9.80665;
constexpr double kDeg = kPi / 180.0;
constexpr double kGroundZ = 0.04;  // centre-of-mass height when resting on landing gear

struct Vec3 { double x = 0, y = 0, z = 0; };
inline Vec3 operator+(Vec3 a, Vec3 b) { return {a.x + b.x, a.y + b.y, a.z + b.z}; }
inline Vec3 operator-(Vec3 a, Vec3 b) { return {a.x - b.x, a.y - b.y, a.z - b.z}; }
inline Vec3 operator*(double s, Vec3 a) { return {s * a.x, s * a.y, s * a.z}; }
inline Vec3 operator/(Vec3 a, double s) { return {a.x / s, a.y / s, a.z / s}; }
inline double dot(Vec3 a, Vec3 b) { return a.x * b.x + a.y * b.y + a.z * b.z; }
inline Vec3 cross(Vec3 a, Vec3 b) {
    return {a.y * b.z - a.z * b.y, a.z * b.x - a.x * b.z, a.x * b.y - a.y * b.x};
}
inline double norm(Vec3 a) { return std::sqrt(dot(a, a)); }

// Body frame is FLU (x forward, y left, z up). Quaternion rotates body -> world.
struct Quat { double w = 1, x = 0, y = 0, z = 0; };
inline Quat operator*(Quat a, Quat b) {
    return {a.w * b.w - a.x * b.x - a.y * b.y - a.z * b.z,
            a.w * b.x + a.x * b.w + a.y * b.z - a.z * b.y,
            a.w * b.y - a.x * b.z + a.y * b.w + a.z * b.x,
            a.w * b.z + a.x * b.y - a.y * b.x + a.z * b.w};
}
inline Quat conj(Quat q) { return {q.w, -q.x, -q.y, -q.z}; }
inline Quat normalized(Quat q) {
    double n = std::sqrt(q.w * q.w + q.x * q.x + q.y * q.y + q.z * q.z);
    return n > 0 ? Quat{q.w / n, q.x / n, q.y / n, q.z / n} : Quat{};
}
inline Vec3 rotate(Quat q, Vec3 v) {
    Vec3 u{q.x, q.y, q.z};
    Vec3 t = 2.0 * cross(u, v);
    return v + q.w * t + cross(u, t);
}
inline Vec3 rotate_inv(Quat q, Vec3 v) { return rotate(conj(q), v); }
inline Quat from_yaw(double yaw) { return {std::cos(yaw / 2), 0, 0, std::sin(yaw / 2)}; }

// ZYX Euler. Positive pitch = rotation about +y = NOSE DOWN (right-hand rule in FLU).
struct Euler { double roll, pitch, yaw; };
inline Euler to_euler(Quat q) {
    double s = 2.0 * (q.w * q.y - q.z * q.x);
    s = s > 1 ? 1 : (s < -1 ? -1 : s);
    return {std::atan2(2.0 * (q.w * q.x + q.y * q.z), 1.0 - 2.0 * (q.x * q.x + q.y * q.y)),
            std::asin(s),
            std::atan2(2.0 * (q.w * q.z + q.x * q.y), 1.0 - 2.0 * (q.y * q.y + q.z * q.z))};
}

template <class T> inline T clampv(T v, T lo, T hi) { return v < lo ? lo : (v > hi ? hi : v); }

enum FlightMode : int { MODE_ACRO = 0, MODE_ANGLE = 1 };

// ---- C-ABI POD structs: every field is double so Python can mirror them trivially ----
struct FpvConfig {
    // airframe
    double mass, ixx, iyy, izz, arm_x, arm_y;
    // propulsion
    double max_thrust, max_omega, tau_up, tau_down, c_yaw;
    // aerodynamics
    double drag_lin, drag_quad_xy, drag_quad_z, ang_drag;
    // rates (per axis: roll, pitch, yaw)
    double rc_rate[3], super_rate[3], expo[3];
    // rate-loop PID(F) (per axis)
    double kp[3], ki[3], kd[3], kff[3];
    // misc controller
    double angle_kp, angle_max_deg, throttle_expo, gyro_noise_dps, d_cut_hz, i_limit, idle;
    // environment
    double wind_speed, wind_dir_deg, gust;
    // battery
    double cells, capacity_mah, r_int;
    // solver
    double physics_hz, control_div, crash_speed, seed;
};

struct FpvState {
    double time;
    double pos[3], vel[3], quat[4];
    double euler[3];     // roll, pitch (nose-UP positive), yaw  [rad]
    double gyro[3];      // true body rates [rad/s]
    double rate_sp[3];   // commanded body rates [rad/s]
    double motors[4];    // normalised motor command 0..1
    double throttle, voltage, current, mah, battery_pct, speed, accel_g, vscale;
    int32_t armed, crashed, on_ground, mode;
};

struct FpvObstacle {  // yaw-rotated box
    double cx, cy, cz, hx, hy, hz, yaw;
};

inline FpvConfig default_config() {
    FpvConfig c{};
    c.mass = 0.65; c.ixx = 0.0033; c.iyy = 0.0035; c.izz = 0.0058;
    c.arm_x = 0.078; c.arm_y = 0.078;
    c.max_thrust = 12.0; c.max_omega = 3000.0; c.tau_up = 0.014; c.tau_down = 0.028; c.c_yaw = 0.016;
    c.drag_lin = 0.02; c.drag_quad_xy = 0.011; c.drag_quad_z = 0.030; c.ang_drag = 2.0e-4;
    for (int i = 0; i < 3; ++i) { c.rc_rate[i] = 1.2; c.super_rate[i] = 0.72; c.expo[i] = 0.0; }
    c.rc_rate[2] = 1.1; c.super_rate[2] = 0.65;
    for (int i = 0; i < 2; ++i) { c.kp[i] = 0.030; c.ki[i] = 0.35; c.kd[i] = 0.00055; c.kff[i] = 0.0; }
    c.kp[2] = 0.14; c.ki[2] = 0.9; c.kd[2] = 0.0; c.kff[2] = 0.0;
    c.angle_kp = 9.0; c.angle_max_deg = 55.0; c.throttle_expo = 0.6; c.gyro_noise_dps = 0.25;
    c.d_cut_hz = 110.0; c.i_limit = 0.25; c.idle = 0.04;
    c.wind_speed = 0.0; c.wind_dir_deg = 0.0; c.gust = 0.0;
    c.cells = 6; c.capacity_mah = 1300; c.r_int = 0.022;
    c.physics_hz = 2000; c.control_div = 2; c.crash_speed = 7.0; c.seed = 1;
    return c;
}

}  // namespace fpv
