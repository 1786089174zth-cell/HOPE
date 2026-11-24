// The Planner's racket target for a ball at the strike plane, in bulk: main's RacketTargetPlanner::plan (the
// copy bundled with Supergate, native/planner) called on rows of strike states, without Runner, robot or
// simulator. Built and driven by planner_targets.py.
//
//   planner_targets <return_physics.yaml> <ball_profile.yaml> <land_x> <land_y> <flight_s> < in.bin > out.bin
//
// in: rows of 9 float64, ball position (table frame: x from the robot's edge towards the net, z above the
// table top), velocity and spin (rad/s) at the strike. out: rows of 12 float64: valid, racket velocity (3; the
// face normal is along it), predicted landing (3), return flight time, clears the net, outgoing ball velocity (3).
//
//   planner_targets flight <plane_x> <dt> <steps> < in.bin > out.bin
//
// The ball's flight from a launch to the strike plane, make_launches.flight step for step (Supergate's flight
// and table-bounce model, RK4), one launch per thread. in: rows of 9 float64, launch position and velocity
// (table frame) and spin. out: rows of 17 float64: state at the plane (6; NaN if it never gets there), spin
// there (3), flight time, bounces, bounce point (2), y and z at the net (2), time at the net, time of the bounce.
#include "hope_planner_cpp/racket_target_planner.hpp"
#include "hope_planner_cpp/return_physics_yaml.hpp"

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

namespace {

const double kDrag = 0.12453915600992894, kMagnus = 0.00399279130564425;
const double kFriction = 0.41642144985036184, kNormalShare = 0.218520906499864, kRestitution = 0.9373319241133231;

inline void accel(const double* v, const double* w, double* a) {
  const double speed = std::sqrt(v[0] * v[0] + v[1] * v[1] + v[2] * v[2]);
  a[0] = -kDrag * speed * v[0] + kMagnus * (w[1] * v[2] - w[2] * v[1]);
  a[1] = -kDrag * speed * v[1] + kMagnus * (w[2] * v[0] - w[0] * v[2]);
  a[2] = -kDrag * speed * v[2] + kMagnus * (w[0] * v[1] - w[1] * v[0]) - 9.81;
}

void fly(const double* in, double plane, double dt, int steps, double* out) {
  const double nan = std::nan("");
  double s[6], w[3], t = 0.0;
  for (int k = 0; k < 6; ++k) s[k] = in[k];
  for (int k = 0; k < 3; ++k) w[k] = in[6 + k];
  for (int k = 0; k < 17; ++k) out[k] = nan;
  out[9] = 0.0; out[10] = 0.0; out[11] = out[12] = out[13] = out[14] = -10.0;
  int bounces = 0;
  for (int step = 0; step < steps; ++step) {
    double prev[6], a[3], b[3], c[3], d[3], vb[3], vc[3], vd[3];
    for (int k = 0; k < 6; ++k) prev[k] = s[k];
    const double* v = prev + 3;
    accel(v, w, a);
    for (int k = 0; k < 3; ++k) vb[k] = v[k] + 0.5 * dt * a[k];
    accel(vb, w, b);
    for (int k = 0; k < 3; ++k) vc[k] = v[k] + 0.5 * dt * b[k];
    accel(vc, w, c);
    for (int k = 0; k < 3; ++k) vd[k] = v[k] + dt * c[k];
    accel(vd, w, d);
    for (int k = 0; k < 3; ++k) {
      s[k] = prev[k] + dt * (v[k] + 2 * vb[k] + 2 * vc[k] + vd[k]) / 6;
      s[3 + k] = v[k] + dt * (a[k] + 2 * b[k] + 2 * c[k] + d[k]) / 6;
    }
    t += dt;
    if (prev[0] >= 1.37 && s[0] < 1.37) {
      const double f = (prev[0] - 1.37) / (prev[0] - s[0] + 1e-12);
      out[13] = prev[1] + f * (s[1] - prev[1]);
      out[14] = prev[2] + f * (s[2] - prev[2]);
      out[15] = t - dt * (1 - f);
    }
    if (s[2] <= 0.0201 && prev[2] > 0.0201 && s[5] < 0 && s[0] >= 0 && s[0] <= 2.74 && s[1] >= -1.525 && s[1] <= 0) {
      // contact point velocity: v + w x (0, 0, -r), its horizontal part
      const double ux = s[3] + (w[1] * -0.02), uy = s[4] - (w[0] * -0.02);
      const double size = std::sqrt(ux * ux + uy * uy);
      const double impulse = -std::min(kFriction * size, kNormalShare * (1 + kRestitution) * std::fabs(s[5]));
      const double dvx = impulse * ux / (size + 1e-12), dvy = impulse * uy / (size + 1e-12);
      s[3] += dvx; s[4] += dvy;
      // w -= (0, 0, 1) x dv / (2/3 r)
      w[0] -= -dvy / ((2.0 / 3.0) * 0.02);
      w[1] -= dvx / ((2.0 / 3.0) * 0.02);
      s[5] = -s[5] * kRestitution;
      s[2] = 0.0201;
      out[11] = s[0]; out[12] = s[1];
      out[16] = t;
      ++bounces;
    }
    if (s[0] <= plane) {
      const double f = (prev[0] - plane) / (prev[0] - s[0] + 1e-12);
      for (int k = 0; k < 6; ++k) out[k] = prev[k] + f * (s[k] - prev[k]);
      for (int k = 0; k < 3; ++k) out[6 + k] = w[k];
      out[9] = t;
      break;
    }
    if (s[2] < -0.8) break;
  }
  out[10] = bounces;
}

std::vector<double> read_rows(int width) {
  std::vector<double> in;
  std::vector<double> buffer(static_cast<size_t>(width) * 4096);
  for (size_t got; (got = std::fread(buffer.data(), sizeof(double), buffer.size(), stdin)) > 0;)
    in.insert(in.end(), buffer.begin(), buffer.begin() + static_cast<long>(got));
  return in;
}

}  // namespace

int main(int argc, char** argv) {
  if (argc == 5 && std::string(argv[1]) == "flight") {
    const double plane = std::atof(argv[2]), dt = std::atof(argv[3]);
    const int steps = std::atoi(argv[4]);
    const std::vector<double> in = read_rows(9);
    const long n = static_cast<long>(in.size() / 9);
    std::vector<double> out(static_cast<size_t>(n) * 17);
#pragma omp parallel for schedule(dynamic, 256)
    for (long i = 0; i < n; ++i) fly(in.data() + 9 * i, plane, dt, steps, out.data() + 17 * i);
    std::fwrite(out.data(), sizeof(double), out.size(), stdout);
    return 0;
  }
  if (argc != 6) { std::fprintf(stderr, "usage: planner_targets return.yaml ball.yaml land_x land_y flight_s\n"); return 2; }
  namespace hp = hope_planner_cpp;
  hp::BallPhysics ball;
  hp::PlannerConfig config;
  hp::load_return_physics_yaml(argv[1], ball, config);
  hp::load_ball_profile_yaml(argv[2], ball);
  config.max_predict_time_s = 2.6;  // as native_batch.cpp
  const hp::Vec3 land(std::atof(argv[3]), std::atof(argv[4]), 0.0);
  const double flight = std::atof(argv[5]);
  std::vector<double> in;
  double buffer[9 * 1024];
  for (size_t got; (got = std::fread(buffer, sizeof(double), 9 * 1024, stdin)) > 0;) in.insert(in.end(), buffer, buffer + got);
  const long n = static_cast<long>(in.size() / 9);
  std::vector<double> out(static_cast<size_t>(n) * 12, 0.0);
  const hp::RacketTargetPlanner planner(ball, config, hp::TableParams{});
#pragma omp parallel for schedule(dynamic, 64)
  for (long i = 0; i < n; ++i) {
    const double* r = in.data() + 9 * i;
    hp::StrikeTarget strike;
    strike.ball_position = hp::Vec3(r[0], r[1], r[2]);
    strike.ball_velocity = hp::Vec3(r[3], r[4], r[5]);
    strike.ball_spin_rad_s = hp::Vec3(r[6], r[7], r[8]);
    strike.valid = true;
    const hp::RacketCommand command = planner.plan(strike, land, flight);
    double* o = out.data() + 12 * i;
    o[0] = command.valid ? 1.0 : 0.0;
    for (int k = 0; k < 3; ++k) {
      o[1 + k] = command.velocity[k];
      o[4 + k] = command.predicted_landing[k];
      o[9 + k] = command.outgoing_ball_velocity[k];
    }
    o[7] = command.return_flight_time_s;
    o[8] = command.clears_net ? 1.0 : 0.0;
  }
  std::fwrite(out.data(), sizeof(double), out.size(), stdout);
  return 0;
}
