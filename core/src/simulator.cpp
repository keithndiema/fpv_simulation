#include "fpv/simulator.hpp"
#include <algorithm>
#include <cstring>

namespace fpv {

namespace {
// X-quad motor layout in FLU: FR, RR, RL, FL  (sign of each motor's x / y position)
constexpr double kSignX[4] = {1, -1, -1, 1};
constexpr double kSignY[4] = {-1, -1, 1, 1};
constexpr double kSpin[4] = {-1, 1, -1, 1};  // +1 = CCW seen from above
}  // namespace

Simulator::Simulator(const FpvConfig& cfg) : cfg_(cfg) { reset(); }

void Simulator::reset() {
    std::memset(x_, 0, sizeof(x_));
    x_[2] = ground_height(0, 0) + kGroundZ;
    x_[6] = 1.0;
    gust_ = {};
    time_ = accumulator_ = mah_ = current_ = 0;
    vscale_ = 1.0; voltage_ = cfg_.cells * 4.2; accel_g_ = 1.0;
    ctrl_count_ = 0;
    armed_ = crashed_ = false; on_ground_ = true;
    rng_ = 0x9E3779B97F4A7C15ULL ^ ((uint64_t)cfg_.seed * 0xD1B54A32D192ED03ULL);
    if (rng_ == 0) rng_ = 1;
    fc_.reset();
    ctl_ = ControlOut{};
    last_vel_ = {};
}

void Simulator::set_pose(double x, double y, double z, double yaw) {
    reset();
    x_[0] = x; x_[1] = y; x_[2] = z;
    Quat q = from_yaw(yaw);
    x_[6] = q.w; x_[7] = q.x; x_[8] = q.y; x_[9] = q.z;
}

void Simulator::set_terrain(const float* h, int n, double cell, double ox, double oy) {
    if (!h || n < 2 || cell <= 0) { terrain_.clear(); terrain_n_ = 0; return; }
    terrain_.assign(h, h + (size_t)n * (size_t)n);
    terrain_n_ = n; terrain_cell_ = cell; terrain_ox_ = ox; terrain_oy_ = oy;
}

double Simulator::ground_height(double x, double y) const {
    if (terrain_n_ < 2) return 0.0;
    const int n = terrain_n_;
    const double u = clampv((x - terrain_ox_) / terrain_cell_, 0.0, n - 1.000001);
    const double v = clampv((y - terrain_oy_) / terrain_cell_, 0.0, n - 1.000001);
    const int i = (int)u, j = (int)v;
    const double fu = u - i, fv = v - j;
    const float* r0 = &terrain_[(size_t)j * n + i];
    const float* r1 = r0 + n;
    return (r0[0] * (1 - fu) + r0[1] * fu) * (1 - fv) + (r1[0] * (1 - fu) + r1[1] * fu) * fv;
}

void Simulator::ground_gradient(double x, double y, double& gx, double& gy) const {
    const double e = terrain_cell_ > 0 ? terrain_cell_ * 0.5 : 0.5;
    gx = (ground_height(x + e, y) - ground_height(x - e, y)) / (2 * e);
    gy = (ground_height(x, y + e) - ground_height(x, y - e)) / (2 * e);
}

void Simulator::set_armed(bool a) {
    if (a && crashed_) return;
    armed_ = a;
    if (!a) fc_.reset();
}

double Simulator::uniform() {  // xorshift64*
    rng_ ^= rng_ >> 12; rng_ ^= rng_ << 25; rng_ ^= rng_ >> 27;
    return ((rng_ * 0x2545F4914F6CDD1DULL) >> 11) * (1.0 / 9007199254740992.0);
}
double Simulator::gauss() {
    double u1 = std::max(uniform(), 1e-12), u2 = uniform();
    return std::sqrt(-2.0 * std::log(u1)) * std::cos(2.0 * kPi * u2);
}

void Simulator::derive(const double* x, const double* wcmd, const Vec3& wind, double* dx) const {
    const Vec3 v{x[3], x[4], x[5]};
    const Quat q = normalized({x[6], x[7], x[8], x[9]});
    const Vec3 w{x[10], x[11], x[12]};
    const double kT = cfg_.max_thrust / (cfg_.max_omega * cfg_.max_omega);

    double thrust_sum = 0, tx = 0, ty = 0, tz = 0;
    for (int i = 0; i < 4; ++i) {
        const double T = kT * x[13 + i] * x[13 + i];
        thrust_sum += T;
        tx += cfg_.arm_y * kSignY[i] * T;     // roll torque  = sum(y_i * T_i)
        ty += -cfg_.arm_x * kSignX[i] * T;    // pitch torque = -sum(x_i * T_i)
        tz += -kSpin[i] * cfg_.c_yaw * T;     // reaction torque opposes rotor spin
    }
    const Vec3 tau{tx - cfg_.ang_drag * w.x, ty - cfg_.ang_drag * w.y, tz - cfg_.ang_drag * w.z};
    const Vec3 Iw{cfg_.ixx * w.x, cfg_.iyy * w.y, cfg_.izz * w.z};
    const Vec3 gyro_term = cross(w, Iw);
    const Vec3 wdot{(tau.x - gyro_term.x) / cfg_.ixx, (tau.y - gyro_term.y) / cfg_.iyy,
                    (tau.z - gyro_term.z) / cfg_.izz};

    const Vec3 vb = rotate_inv(q, v - wind);
    const Vec3 fd{-(cfg_.drag_lin + cfg_.drag_quad_xy * std::fabs(vb.x)) * vb.x,
                  -(cfg_.drag_lin + cfg_.drag_quad_xy * std::fabs(vb.y)) * vb.y,
                  -(cfg_.drag_lin + cfg_.drag_quad_z * std::fabs(vb.z)) * vb.z};
    const Vec3 a = rotate(q, Vec3{0, 0, thrust_sum} + fd) / cfg_.mass + Vec3{0, 0, -kG};

    dx[0] = v.x; dx[1] = v.y; dx[2] = v.z;
    dx[3] = a.x; dx[4] = a.y; dx[5] = a.z;
    const Quat qd = Quat{x[6], x[7], x[8], x[9]} * Quat{0, w.x, w.y, w.z};
    dx[6] = 0.5 * qd.w; dx[7] = 0.5 * qd.x; dx[8] = 0.5 * qd.y; dx[9] = 0.5 * qd.z;
    dx[10] = wdot.x; dx[11] = wdot.y; dx[12] = wdot.z;
    for (int i = 0; i < 4; ++i) {
        const double tau_m = wcmd[i] > x[13 + i] ? cfg_.tau_up : cfg_.tau_down;
        dx[13 + i] = (wcmd[i] - x[13 + i]) / tau_m;
    }
}

void Simulator::step() {
    const double dt = 1.0 / cfg_.physics_hz;
    const int div = std::max(1, (int)cfg_.control_div);

    // ---- Flight-controller tick (decimated, ZOH between ticks) ----
    if (ctrl_count_++ % div == 0) {
        const Quat q = normalized({x_[6], x_[7], x_[8], x_[9]});
        const Euler e = to_euler(q);
        const double n = cfg_.gyro_noise_dps * kDeg;
        const Vec3 gyro{x_[10] + n * gauss(), x_[11] + n * gauss(), x_[12] + n * gauss()};
        const bool airborne = x_[2] > ground_height(x_[0], x_[1]) + kGroundZ + 0.03;
        const bool active = armed_ && !crashed_ && (sticks_.thr > 0.06 || airborne);
        fc_.update(cfg_, sticks_, gyro, e.roll, e.pitch, dt * div, mode_, active, ctl_);
    }
    const bool live = armed_ && !crashed_;

    // ---- Motor command (thrust scales with pack voltage) ----
    double wcmd[4];
    for (int i = 0; i < 4; ++i)
        wcmd[i] = live ? cfg_.max_omega * vscale_ * std::sqrt(clampv(ctl_.u[i], 0.0, 1.0)) : 0.0;

    // ---- Wind: constant mean + Ornstein-Uhlenbeck gusts (skipped entirely when gust == 0) ----
    if (cfg_.gust > 0.0) {
        const double tau_g = 2.0;
        const double k = std::sqrt(2.0 * dt / tau_g) * cfg_.gust;
        gust_ = gust_ + Vec3{-gust_.x / tau_g * dt + k * gauss(), -gust_.y / tau_g * dt + k * gauss(),
                             (-gust_.z / tau_g * dt + k * gauss()) * 0.3};
    }
    const double wd = cfg_.wind_dir_deg * kDeg;
    const Vec3 wind = Vec3{cfg_.wind_speed * std::cos(wd), cfg_.wind_speed * std::sin(wd), 0} + gust_;

    // ---- RK4 ----
    double k1[17], k2[17], k3[17], k4[17], tmp[17];
    derive(x_, wcmd, wind, k1);
    for (int i = 0; i < 17; ++i) tmp[i] = x_[i] + 0.5 * dt * k1[i];
    derive(tmp, wcmd, wind, k2);
    for (int i = 0; i < 17; ++i) tmp[i] = x_[i] + 0.5 * dt * k2[i];
    derive(tmp, wcmd, wind, k3);
    for (int i = 0; i < 17; ++i) tmp[i] = x_[i] + dt * k3[i];
    derive(tmp, wcmd, wind, k4);
    for (int i = 0; i < 17; ++i) x_[i] += dt / 6.0 * (k1[i] + 2 * k2[i] + 2 * k3[i] + k4[i]);
    const Quat qn = normalized({x_[6], x_[7], x_[8], x_[9]});
    x_[6] = qn.w; x_[7] = qn.x; x_[8] = qn.y; x_[9] = qn.z;

    // ---- Battery: current ~ idle + k*(omega/omega_max)^2.5 per motor, sag through internal resistance ----
    double amps = 0.6;
    for (int i = 0; i < 4; ++i)
        amps += 0.8 + 30.0 * [&] { const double r = std::max(x_[13 + i], 0.0) / cfg_.max_omega; return r * r * std::sqrt(r); }();
    current_ = amps;
    mah_ += amps * dt / 3.6;
    const double frac = clampv(mah_ / cfg_.capacity_mah, 0.0, 1.0);
    const double cell_oc = 4.2 - 0.55 * frac - 0.45 * [&] { const double f2 = frac * frac, f4 = f2 * f2; return f4 * f4; }();
    voltage_ = cfg_.cells * cell_oc - amps * cfg_.r_int;
    vscale_ = clampv(voltage_ / (cfg_.cells * 4.2), 0.4, 1.0);

    post_step(dt);
    // Flush tiny values to zero: exponentially decaying states otherwise become denormals (10-100x slower).
    for (int i = 3; i < 17; ++i)
        if (i < 6 || i > 9) { if (std::fabs(x_[i]) < 1e-12) x_[i] = 0.0; }
    time_ += dt;
}

void Simulator::post_step(double dt) {
    const Quat q{x_[6], x_[7], x_[8], x_[9]};
    on_ground_ = false;

    // Obstacle collision (drone approximated by a 12 cm sphere vs yaw-rotated boxes)
    const double r = 0.12;
    for (const auto& o : obstacles_) {
        const double dxp = x_[0] - o.cx, dyp = x_[1] - o.cy, dzp = x_[2] - o.cz;
        const double reach = std::sqrt(o.hx * o.hx + o.hy * o.hy) + r;   // cheap reject before the trig
        if (dxp * dxp + dyp * dyp > reach * reach || std::fabs(dzp) > o.hz + r) continue;
        const double c = std::cos(o.yaw), s = std::sin(o.yaw);
        const double lx = c * dxp + s * dyp, ly = -s * dxp + c * dyp;
        const double ex = std::max(std::fabs(lx) - o.hx, 0.0), ey = std::max(std::fabs(ly) - o.hy, 0.0),
                     ez = std::max(std::fabs(dzp) - o.hz, 0.0);
        if (ex * ex + ey * ey + ez * ez < r * r) {
            crashed_ = true; armed_ = false;
            x_[3] *= 0.15; x_[4] *= 0.15; x_[5] *= 0.15;
        }
    }

    const double gz = ground_height(x_[0], x_[1]) + kGroundZ;
    if (x_[2] <= gz) {
        x_[2] = gz;
        on_ground_ = true;
        double gx = 0, gy = 0;
        if (terrain_n_ >= 2) ground_gradient(x_[0], x_[1], gx, gy);
        const double inv = 1.0 / std::sqrt(gx * gx + gy * gy + 1.0);
        const Vec3 nrm{-gx * inv, -gy * inv, inv};                      // surface normal (flat: 0,0,1)
        const double tilt = std::acos(clampv(rotate(normalized(q), Vec3{0, 0, 1}).z, -1.0, 1.0));
        const double vnorm = x_[3] * nrm.x + x_[4] * nrm.y + x_[5] * nrm.z;
        if (vnorm < 0) {
            const double impact = -vnorm;
            if (impact > cfg_.crash_speed || (tilt > 1.05 && impact > 2.5)) { crashed_ = true; armed_ = false; }
            const double bounce = impact > 1.5 ? 0.18 : 0.0;           // remove the normal component (+ small bounce)
            x_[3] += nrm.x * impact * (1 + bounce);
            x_[4] += nrm.y * impact * (1 + bounce);
            x_[5] += nrm.z * impact * (1 + bounce);
        }
        if (tilt > 1.3) { crashed_ = true; armed_ = false; }  // flipped on its back
        const double fr = std::exp(-(crashed_ ? 3.5 : 5.0) * dt);
        x_[3] *= fr; x_[4] *= fr;
        const double wd = std::exp(-8.0 * dt);
        x_[10] *= wd; x_[11] *= wd; x_[12] *= wd;
        if (!crashed_) {  // landing gear self-levels roll/pitch
            const Euler e = to_euler(normalized(q));
            const Quat ql = from_yaw(e.yaw);
            const double kk = 1.0 - std::exp(-10.0 * dt);
            const double sgn = (q.w * ql.w + q.x * ql.x + q.y * ql.y + q.z * ql.z) < 0 ? -1.0 : 1.0;
            const Quat qn = normalized({q.w * (1 - kk) + sgn * ql.w * kk, q.x * (1 - kk) + sgn * ql.x * kk,
                                        q.y * (1 - kk) + sgn * ql.y * kk, q.z * (1 - kk) + sgn * ql.z * kk});
            x_[6] = qn.w; x_[7] = qn.x; x_[8] = qn.y; x_[9] = qn.z;
        }
    }
    const Vec3 vn{x_[3], x_[4], x_[5]};
    const Vec3 acc = (vn - last_vel_) / dt;
    accel_g_ += 0.02 * (norm(acc + Vec3{0, 0, kG}) / kG - accel_g_);
    last_vel_ = vn;
}

int Simulator::advance(double dt) {
    accumulator_ = std::min(accumulator_ + dt, 0.1);
    const double h = 1.0 / cfg_.physics_hz;
    int n = 0;
    while (accumulator_ >= h) { step(); accumulator_ -= h; ++n; }
    return n;
}

void Simulator::fill_state(FpvState& s) const {
    s.time = time_;
    for (int i = 0; i < 3; ++i) { s.pos[i] = x_[i]; s.vel[i] = x_[3 + i]; s.gyro[i] = x_[10 + i]; }
    for (int i = 0; i < 4; ++i) s.quat[i] = x_[6 + i];
    const Euler e = to_euler(normalized({x_[6], x_[7], x_[8], x_[9]}));
    s.euler[0] = e.roll; s.euler[1] = -e.pitch; s.euler[2] = e.yaw;
    for (int i = 0; i < 3; ++i) s.rate_sp[i] = ctl_.rate_sp[i];
    for (int i = 0; i < 4; ++i) s.motors[i] = (armed_ && !crashed_) ? ctl_.u[i] : 0.0;
    s.throttle = ctl_.throttle; s.voltage = voltage_; s.current = current_; s.mah = mah_;
    s.battery_pct = clampv(100.0 * (1.0 - mah_ / cfg_.capacity_mah), 0.0, 100.0);
    s.speed = norm({x_[3], x_[4], x_[5]}); s.accel_g = accel_g_; s.vscale = vscale_;
    s.armed = armed_; s.crashed = crashed_; s.on_ground = on_ground_; s.mode = mode_;
}

}  // namespace fpv
