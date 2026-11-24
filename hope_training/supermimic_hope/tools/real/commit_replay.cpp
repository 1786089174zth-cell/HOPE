// When and with what prediction main's packetizer would have committed each incoming ball of a recorded ball
// track: IncomingTrajectory + AdaptiveFlightCommit + BatchPhysicsEstimator + FutureContactPredictor (the copies
// bundled with Supergate, native/planner, with the commit gate of tools/supergate/commit_gate.py), fed the
// recorded samples in order. No robot, no simulator. Built and driven by commit_replay.py.
//
//   commit_replay <return_physics.yaml> <ball_profile.yaml> <packetizer.yaml> <planner.yaml> <plane_x> <gate_x|nan> <policy_lead_s> < in.bin > out.bin
//
// in: rows of 4 float64 in time order: time (s), ball position in the table frame (x from the robot's edge towards
// the net, z of the ball's centre above the table top). out: one row of 13 float64 per commit: time of the
// commit (its latest sample), samples and span (s) of the fit, valid, predicted strike time, ball position (3) and
// velocity (3) at the plane, predicted bounces, time the gate (or the net) was passed.
#include "hope_planner_cpp/batch_physics_estimator.hpp"
#include "hope_planner_cpp/future_contact_predictor.hpp"
#include "hope_planner_cpp/incoming_trajectory.hpp"
#include "hope_planner_cpp/return_physics_yaml.hpp"
#include "hope_planner_cpp/spin_estimator.hpp"

#include <yaml-cpp/yaml.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

namespace hp = hope_planner_cpp;

static double val(YAML::Node n, const char* k, double fallback) { return n[k] ? n[k].as<double>() : fallback; }
static bool flag(YAML::Node n, const char* k, bool fallback) { return n[k] ? n[k].as<bool>() : fallback; }

static hp::EstimatorConfig estimator_config(YAML::Node n, const char* window) {
  hp::EstimatorConfig c;
  c.window_s = val(n, window, .18);
  c.max_sample_gap_s = val(n, "batch_max_sample_gap_s", 0.);
  c.single_flight_only = flag(n, "batch_single_flight_only", false);
  c.min_span_s = val(n, "batch_estimator_min_span_s", .08);
  c.min_samples = val(n, "batch_estimator_min_samples", 12);
  c.post_bounce_min_span_s = val(n, "batch_post_bounce_min_span_s", 0.);
  c.post_bounce_exclusion_s = val(n, "batch_post_bounce_exclusion_s", .008);
  c.spin_enabled = flag(n, "future_contact_magnus_enabled", true);
  c.spin_min_coherence = val(n, "future_contact_spin_min_coherence", .4);
  return c;
}

static hp::FutureContactConfig future_config(YAML::Node n) {
  hp::FutureContactConfig c;
  c.spin_enabled = flag(n, "future_contact_spin_enabled", true);
  c.magnus_enabled = flag(n, "future_contact_magnus_enabled", true);
  c.tangential_gain = val(n, "future_contact_spin_tangential_gain", .41642144985036184);
  c.omega_scale = val(n, "future_contact_spin_omega_scale", 1.);
  c.min_coherence = val(n, "future_contact_spin_min_coherence", .4);
  c.require_spin_for_future_contact = flag(n, "future_contact_require_spin", true);
  return c;
}

int main(int argc, char** argv) {
  if (argc != 8) {
    std::fprintf(stderr, "usage: commit_replay <return.yaml> <ball.yaml> <packetizer.yaml> <planner.yaml> <plane_x> <gate_x|nan> <policy_lead_s>\n");
    return 2;
  }
  const double plane = std::atof(argv[5]), gate = std::atof(argv[6]), lead = std::atof(argv[7]);
  hp::BallPhysics ball;
  hp::PlannerConfig pc;
  hp::load_return_physics_yaml(argv[1], ball, pc);
  hp::load_ball_profile_yaml(argv[2], ball);
  pc.max_predict_time_s = 2.6;
  auto pk = YAML::LoadFile(argv[3])["hope_ball_flight_packetizer"]["ros__parameters"];
  auto pn = YAML::LoadFile(argv[4])["hope_planner"]["ros__parameters"];
  hp::EstimatorConfig ec = estimator_config(pn, "batch_estimator_window_s");
  hp::SpinEstimatorConfig sc;
  sc.window_s = val(pn, "spin_window_s", .18);
  hp::FutureContactPredictor predictor(ball, pc, hp::TableParams{}, future_config(pn));

  hp::IncomingTrajectoryConfig ic;  // as flight_packetizer_node.cpp reads it
  ic.net_x = val(pk, "net_x", 1.37);
  ic.commit_gate_x = gate;
  ic.estimator_window_s = val(pk, "flight_window_s", .18);
  ic.estimator_max_sample_gap_s = val(pk, "batch_max_sample_gap_s", 0.);
  ic.commit_delay_s = val(pk, "post_net_commit_delay_s", .05);
  ic.opponent_side_margin_m = val(pk, "incoming_opponent_side_margin_m", .05);
  ic.incoming_speed_threshold_mps = val(pk, "incoming_speed_threshold_mps", .25);
  ic.outgoing_speed_threshold_mps = val(pk, "outgoing_speed_threshold_mps", .25);
  ic.source_gap_reset_s = val(pk, "incoming_source_gap_reset_s", .25);
  ic.direction_fit_samples = std::max(3, int(val(pk, "incoming_direction_fit_samples", 4)));
  ic.direction_confirmations = std::max(1, int(val(pk, "incoming_direction_confirmations", 2)));
  ic.pre_roll_samples = std::max(6, int(val(pk, "incoming_pre_roll_samples", 24)));
  ic.adaptive.enabled = flag(pk, "adaptive_commit_enabled", true);
  ic.adaptive.policy_lead_s = lead;
  ic.adaptive.transport_budget_s = val(pk, "adaptive_transport_budget_s", .03);
  ic.adaptive.solve_budget_s = val(pk, "adaptive_solve_budget_s", .005);
  ic.adaptive.policy_tick_s = val(pk, "adaptive_policy_tick_s", .02);
  ic.adaptive.max_sample_age_s = val(pk, "adaptive_max_sample_age_s", .05);
  ic.adaptive.evaluation_period_s = val(pk, "adaptive_evaluation_period_s", .01);
  hp::CommitPredictionConfig cp;
  cp.observed_physics = ball;
  cp.future_physics = ball;
  cp.estimator = estimator_config(pk, "flight_window_s");
  cp.spin_estimator = sc;
  cp.future_contact = future_config(pk);
  cp.planner = pc;
  cp.planner.x_hit = plane;
  hp::IncomingTrajectory incoming(ic, cp);
  incoming.set_prediction_x_hit(plane);

  std::vector<double> out;
  double row[4];
  std::uint64_t sequence = 0;
  while (std::fread(row, sizeof(double), 4, stdin) == 4) {
    hp::BallSample sample;
    sample.source_time_s = row[0];
    sample.source_time_ns = std::llround(row[0] * 1e9);
    sample.position = hp::Vec3(row[1], row[2], row[3]);
    sample.sequence = ++sequence;
    auto update = incoming.observe(sample);
    if (!update.snapshot_ready) continue;
    hp::BatchPhysicsEstimator est(ball, ec, sc);
    hp::SpinEstimator spin(sc);
    for (std::size_t i = 0; i < update.snapshot.sample_count; i++) {
      est.push(update.snapshot.samples[i]);
      spin.push(update.snapshot.samples[i]);
    }
    const auto state = est.estimate();
    hp::StrikeTarget strike;
    if (state.valid) strike = predictor.predict(state, plane, spin.estimate());
    const double nan = std::nan("");
    const bool ok = state.valid && strike.valid;
    out.insert(out.end(), {row[0], double(state.sample_count), state.sample_span_s, ok ? 1. : 0.,
                           ok ? strike.strike_source_time_s : nan,
                           ok ? strike.ball_position.x() : nan, ok ? strike.ball_position.y() : nan, ok ? strike.ball_position.z() : nan,
                           ok ? strike.ball_velocity.x() : nan, ok ? strike.ball_velocity.y() : nan, ok ? strike.ball_velocity.z() : nan,
                           ok ? double(strike.predicted_bounces) : nan, update.snapshot.one_shot.net_cross_source_time_s});
  }
  std::fwrite(out.data(), sizeof(double), out.size(), stdout);
  return 0;
}
