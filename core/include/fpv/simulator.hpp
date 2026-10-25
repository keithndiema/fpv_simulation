#pragma once
#include <vector>
#include "controller.hpp"
#include "types.hpp"

namespace fpv {

class Simulator {
public:
    explicit Simulator(const FpvConfig& cfg);

    void set_config(const FpvConfig& cfg) { cfg_ = cfg; }
    const FpvConfig& config() const { return cfg_; }
    void reset();
    void set_pose(double x, double y, double z, double yaw);
    void set_sticks(double roll, double pitch, double yaw, double thr) { sticks_ = {roll, pitch, yaw, thr}; }
    void set_armed(bool a);
    void set_mode(int m) { mode_ = m; }
    void set_obstacles(const FpvObstacle* o, int n) { obstacles_.assign(o, o + n); }

    // Heightfield terrain: n x n vertices, row-major [iy * n + ix], world position of vertex (0,0) = (ox, oy).
    // Outside the grid the height is clamped to the border. n <= 1 restores a flat world at z = 0.
    void set_terrain(const float* h, int n, double cell, double ox, double oy);
    double ground_height(double x, double y) const;

    // Advances by wall-clock dt using a fixed-step accumulator. Returns number of physics steps taken.
    int advance(double dt);
    void step();
    void fill_state(FpvState& s) const;

private:
    void derive(const double* x, const double* wcmd, const Vec3& wind, double* dx) const;
    void post_step(double dt);
    double gauss();
    double uniform();
    void ground_gradient(double x, double y, double& gx, double& gy) const;

    FpvConfig cfg_;
    FlightController fc_;
    Sticks sticks_;
    ControlOut ctl_;
    std::vector<FpvObstacle> obstacles_;
    std::vector<float> terrain_;
    int terrain_n_ = 0;
    double terrain_cell_ = 1, terrain_ox_ = 0, terrain_oy_ = 0;

    // x: pos(0-2) vel(3-5) quat(6-9) body rates(10-12) motor omega(13-16)
    double x_[17];
    Vec3 gust_;
    double time_ = 0, accumulator_ = 0;
    double mah_ = 0, current_ = 0, vscale_ = 1, voltage_ = 25.2, accel_g_ = 1;
    int mode_ = MODE_ACRO, ctrl_count_ = 0;
    bool armed_ = false, crashed_ = false, on_ground_ = true;
    uint64_t rng_ = 1;
    Vec3 last_vel_;
};

}  // namespace fpv
