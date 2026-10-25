// Headless tools: step-response CSV (the original demo, now 6-DOF) and a throughput benchmark.
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include "fpv/simulator.hpp"

using namespace fpv;

static void step_response(double target_deg, bool pitch_axis) {
    Simulator sim(default_config());
    sim.set_pose(0, 0, 100.0, 0);
    sim.set_mode(MODE_ANGLE);
    sim.set_armed(true);
    std::puts("Time,Target,Pitch,Roll,M0,M1,M2,M3");
    const double hover = 0.30;
    FpvState s{};
    for (int i = 0; i < 1500; ++i) {  // 1.5 s, logged @100 Hz
        const double cmd = (i > 200 ? target_deg / 55.0 : 0.0);
        sim.set_sticks(pitch_axis ? 0 : cmd, pitch_axis ? cmd : 0, 0, hover);
        sim.advance(0.001);
        if (i % 10 == 0) {
            sim.fill_state(s);
            std::printf("%.3f,%.2f,%.3f,%.3f,%.3f,%.3f,%.3f,%.3f\n", s.time, target_deg * (i > 200),
                        -s.euler[1] / kDeg, s.euler[0] / kDeg, s.motors[0], s.motors[1], s.motors[2], s.motors[3]);
        }
    }
}

static void bench(double seconds) {
    Simulator sim(default_config());
    sim.set_pose(0, 0, 50.0, 0);
    sim.set_armed(true);
    const auto t0 = std::chrono::steady_clock::now();
    FpvState s{};
    long steps = 0;
    for (double t = 0; t < seconds; t += 0.001) {
        sim.set_sticks(0.4 * std::sin(t * 2), 0.4 * std::cos(t * 1.3), 0.2 * std::sin(t), 0.35);
        steps += sim.advance(0.001);
    }
    sim.fill_state(s);
    const double wall = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
    std::printf("simulated %.0f s (%ld physics steps) in %.3f s wall -> %.0fx real-time, %.2f M steps/s\n",
                seconds, steps, wall, seconds / wall, steps / wall / 1e6);
}

int main(int argc, char** argv) {
    if (argc > 1 && !std::strcmp(argv[1], "bench")) { bench(argc > 2 ? std::atof(argv[2]) : 600.0); return 0; }
    if (argc > 1 && !std::strcmp(argv[1], "roll")) { step_response(argc > 2 ? std::atof(argv[2]) : 15.0, false); return 0; }
    step_response(argc > 1 ? std::atof(argv[1]) : 15.0, true);
    return 0;
}
