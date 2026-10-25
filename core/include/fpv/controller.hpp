#pragma once
// Flight controller: rate PIDF (acro) + angle P-loop (self-level) + airmode mixer.
#include <algorithm>
#include "types.hpp"

namespace fpv {

class Pid {
public:
    void reset() { i_ = 0; d_f_ = 0; ff_f_ = 0; last_meas_ = 0; last_sp_ = 0; primed_ = false; }
    // Derivative on measurement (no kick), PT1-filtered D and FF, clamped integral with
    // conditional anti-windup (stops integrating while the mixer is saturated).
    double update(double sp, double meas, double dt, double kp, double ki, double kd, double kff,
                  double i_limit, double d_cut_hz, bool integrate) {
        const double err = sp - meas;
        const double alpha = dt / (dt + 1.0 / (2.0 * kPi * d_cut_hz));
        if (integrate) i_ = clampv(i_ + ki * err * dt, -i_limit, i_limit);
        double d_raw = 0, ff_raw = 0;
        if (primed_) {
            d_raw = -kd * (meas - last_meas_) / dt;
            ff_raw = kff * (sp - last_sp_) / dt;
        }
        primed_ = true;
        d_f_ += alpha * (d_raw - d_f_);
        ff_f_ += alpha * (ff_raw - ff_f_);
        last_meas_ = meas; last_sp_ = sp;
        return kp * err + i_ + d_f_ + ff_f_;
    }
private:
    double i_ = 0, d_f_ = 0, ff_f_ = 0, last_meas_ = 0, last_sp_ = 0;
    bool primed_ = false;
};

struct Sticks { double roll = 0, pitch = 0, yaw = 0, thr = 0; };
struct ControlOut { double u[4] = {0, 0, 0, 0}; double rate_sp[3] = {0, 0, 0}; double throttle = 0; };

inline double rate_curve(double x, double rc, double sup, double expo) {
    const double ax = std::fabs(x);
    const double e = x * (ax * ax * ax * expo + (1.0 - expo));
    const double sf = 1.0 / clampv(1.0 - ax * sup, 0.01, 1.0);
    return 200.0 * rc * e * sf * kDeg;  // rad/s
}

class FlightController {
public:
    void reset() { for (auto& p : pid_) p.reset(); sat_ = false; }

    // roll/pitch: attitude [rad], pitch positive = nose DOWN (matches body +y rotation).
    void update(const FpvConfig& c, const Sticks& s, Vec3 gyro, double roll, double pitch,
                double dt, int mode, bool active, ControlOut& out) {
        const double sx[3] = {s.roll, s.pitch, -s.yaw};  // stick right/forward/clockwise -> body axes
        double sp[3];
        for (int a = 0; a < 3; ++a) sp[a] = rate_curve(sx[a], c.rc_rate[a], c.super_rate[a], c.expo[a]);
        if (mode == MODE_ANGLE) {
            const double lim = c.angle_max_deg * kDeg, max_rate = 600.0 * kDeg;
            const double tr = s.roll * lim, tp = s.pitch * lim;
            sp[0] = clampv(c.angle_kp * (tr - roll), -max_rate, max_rate);
            sp[1] = clampv(c.angle_kp * (tp - pitch), -max_rate, max_rate);
        }
        const double w[3] = {gyro.x, gyro.y, gyro.z};
        double pid[3];
        for (int a = 0; a < 3; ++a) {
            out.rate_sp[a] = sp[a];
            pid[a] = pid_[a].update(sp[a], w[a], dt, c.kp[a], c.ki[a], c.kd[a], c.kff[a],
                                    c.i_limit, c.d_cut_hz, active && !sat_);
        }
        const double t = clampv(s.thr, 0.0, 1.0);
        const double ut = (1.0 - c.throttle_expo) * t + c.throttle_expo * t * t * t;
        out.throttle = ut;
        if (!active) {
            for (auto& p : pid_) p.reset();
            for (double& u : out.u) u = c.idle;
            sat_ = false;
            return;
        }
        // Mixer (X quad, FLU): torque_x = sum(y_i*T_i), torque_y = -sum(x_i*T_i), yaw reaction via spin.
        static const double sy[4] = {-1, -1, 1, 1}, sxp[4] = {1, -1, -1, 1}, spin[4] = {-1, 1, -1, 1};
        double mix[4], mx = -1e9, mn = 1e9;
        for (int i = 0; i < 4; ++i) {
            mix[i] = pid[0] * sy[i] - pid[1] * sxp[i] - pid[2] * spin[i];
            mx = std::max(mx, mix[i]); mn = std::min(mn, mix[i]);
        }
        const double range = mx - mn;
        const double scale = range > 1.0 ? 1.0 / range : 1.0;
        mx *= scale; mn *= scale;
        const double te = clampv(ut, -mn, 1.0 - mx);  // airmode: shift collective to keep authority
        sat_ = (range > 1.0) || (te != ut);
        for (int i = 0; i < 4; ++i) out.u[i] = clampv(te + mix[i] * scale, c.idle, 1.0);
    }

private:
    Pid pid_[3];
    bool sat_ = false;
};

}  // namespace fpv
