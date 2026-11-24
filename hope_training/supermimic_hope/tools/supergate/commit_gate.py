"""Let Supergate's Planner commit a stroke before the ball is over the net (stdlib only; called by assemble.sh).

    python3 tools/supergate/commit_gate.py <tree>/a3_deploy/a3_deploy_example/scripts/supergate/native/native_batch.cpp

Main's packetizer (``incoming_trajectory.cpp``) considers a commit only after the ball has crossed ``net_x`` (1.37 m),
and sends the stroke at the predicted strike minus 0.405 s. A fast ball needs only 0.22-0.30 s from the net to the
strike plane, so that is all the notice the policy gets for it. This edit keeps the default and reads

``SUPERGATE_COMMIT_X``    the x the ball must be inside before a commit is considered, instead of the net (a new
                          ``commit_gate_x`` in the harness's copy of ``IncomingTrajectoryConfig``; ``net_x`` keeps
                          deciding what the opponent's side is). 2.0 = once the ball is 0.63 m short of the net;
                          9 = as soon as the incoming flight is confirmed and its fit is ready.
``SUPERGATE_POLICY_LEAD`` ``adaptive_policy_lead_s`` (0.35): the notice asked for is this + 0.055 s of budgets.
``SUPERGATE_BALL_NOISE``  standard deviation (m) of white noise added to every ball position the Planner sees
                          (250 Hz in the harness): an earlier commit fits fewer samples and extrapolates further,
                          so its cost only shows with measurement noise. Deterministic per shard and sample time.
``SUPERGATE_PREDICTION_ERROR`` ``st,sy,sz`` (s, m, m): what the Planner believes about the strike is off by Gaussian
                          errors in arrival time, y and z, one draw per shot; the ball flies as it does. The
                          numbers are the errors at 0.40 s of notice, scaled by 0.3 + 0.7 * notice / 0.40. Measured
                          on the real sessions of 2026-11-21 (tools/real/commit_replay.py): 0.030,0.033,0.035
                          (the harness's Planner predicts its own simulator almost exactly, the robot's does not).
"""

import sys
from pathlib import Path

path = Path(sys.argv[1])
text = path.read_text()
config = "hp::IncomingTrajectoryConfig ic;"
planner = path.parent / "planner"
header, source = planner / "include/hope_planner_cpp/incoming_trajectory.hpp", planner / "src/incoming_trajectory.cpp"
field = "  double net_x = 1.37;\n"
crossing = ("  const bool crossing = history_count_ > 0 &&\n"
            "      history_[history_count_ - 1].position.x() > config_.net_x &&\n"
            "      sample.position.x() <= config_.net_x;\n")
fraction = "              (previous.position.x() - config_.net_x) / denominator,\n"
h, s = header.read_text(), source.read_text()
if "commit_gate_x" not in h:
    if h.count(field) != 1 or s.count(crossing) != 1 or s.count(fraction) != 1:
        raise SystemExit("incoming_trajectory: anchors not found")
    header.write_text(h.replace(field, field + "  // Commit gate other than the net (NaN = net_x): a ball already inside it passes at once.\n"
                                               "  double commit_gate_x = std::numeric_limits<double>::quiet_NaN();\n"))
    s = s.replace(crossing, (
        "  const bool gated = std::isfinite(config_.commit_gate_x);\n"
        "  const double gate_x = gated ? config_.commit_gate_x : config_.net_x;\n"
        "  const bool crossing = history_count_ > 0 && sample.position.x() <= gate_x &&\n"
        "      (gated || history_[history_count_ - 1].position.x() > gate_x);\n"))
    s = s.replace(fraction, "              (previous.position.x() - gate_x) / denominator,\n")
    if "#include <cmath>" not in s:
        s = "#include <cmath>\n" + s
    source.write_text(s)
sample = "sample.position=hp::Vec3(d->qpos[q],d->qpos[q+1],d->qpos[q+2]-.76);"
lead = 'ic.adaptive.policy_lead_s=val(pk,"adaptive_policy_lead_s",.35);'
if "sg_commit_x(" not in text:
    for anchor in (config, sample, lead):
        if text.count(anchor) != 1:
            raise SystemExit(f"native_batch.cpp: anchor not found: {anchor}")
    helper = ('#include <cmath>\n#include <cstdint>\n#include <cstdlib>\n'
              'static double sg_env(const char*name,double fallback){const char*v=std::getenv(name);if(!v||!*v)return fallback;return std::strtod(v,nullptr);}\n'
              'static double sg_commit_x(){return sg_env("SUPERGATE_COMMIT_X",std::nan(""));}\n'
              'static double sg_unit(std::uint64_t x){x+=0x9e3779b97f4a7c15ULL;x=(x^(x>>30))*0xbf58476d1ce4e5b9ULL;x=(x^(x>>27))*0x94d049bb133111ebULL;x^=x>>31;'
              'return (double(x>>11)+.5)/9007199254740992.;}\n'
              'static double sg_ball_noise(int shard,long long time_ns,int axis){static const double sigma=sg_env("SUPERGATE_BALL_NOISE",0.);if(sigma<=0.)return 0.;'
              'std::uint64_t key=std::uint64_t(time_ns)*6364136223846793005ULL+std::uint64_t(shard)*1442695040888963407ULL+std::uint64_t(axis)*0x2545f4914f6cdd1dULL;'
              'return sigma*std::sqrt(-2.*std::log(sg_unit(key)))*std::cos(6.283185307179586*sg_unit(key^0x5851f42d4c957f2dULL));}\n')
    first = text.index("#include")
    text = text[:first] + helper + text[first:]
    text = text.replace(config, config + "ic.commit_gate_x=sg_commit_x();")
    text = text.replace(lead, lead + 'ic.adaptive.policy_lead_s=sg_env("SUPERGATE_POLICY_LEAD",ic.adaptive.policy_lead_s);')
    text = text.replace(sample, sample + "for(int k=0;k<3;k++)sample.position[k]+=sg_ball_noise(id,sample.source_time_ns,k);")
    path.write_text(text)
text = path.read_text()
predict = "auto strike=predictor->predict(bs,home.x+home.plane_x,spin.estimate());r.reason=strike.reason;if(!strike.valid)return;"
space = "namespace hp=hope_planner_cpp;\n"
if "sg_prediction_error(" not in text:
    if text.count(predict) != 1 or text.count(space) != 1:
        raise SystemExit("native_batch.cpp: prediction anchors not found")
    helper = ('static void sg_prediction_error(int shard,int shot,double horizon,double&t,hp::Vec3&p){static const std::vector<double> s=[]{std::vector<double> v;'
              'const char*e=std::getenv("SUPERGATE_PREDICTION_ERROR");double a,b,c;if(e&&std::sscanf(e,"%lf,%lf,%lf",&a,&b,&c)==3)v={a,b,c};return v;}();if(s.empty())return;'
              'const double f=.3+.7*std::clamp(horizon,0.,.6)/.4;auto g=[&](int axis){std::uint64_t key=std::uint64_t(shot+1)*6364136223846793005ULL'
              '+std::uint64_t(shard)*1442695040888963407ULL+std::uint64_t(axis+7)*0x2545f4914f6cdd1dULL;'
              'return std::sqrt(-2.*std::log(sg_unit(key)))*std::cos(6.283185307179586*sg_unit(key^0x5851f42d4c957f2dULL));};'
              't+=f*s[0]*g(0);p.y()+=f*s[1]*g(1);p.z()+=f*s[2]*g(2);}\n')
    text = text.replace(space, space + helper)
    text = text.replace(predict, predict + "sg_prediction_error(id,current,strike.strike_source_time_s-a3_pingpong::PpNowWallSec(),strike.strike_source_time_s,strike.ball_position);")
    path.write_text(text)
