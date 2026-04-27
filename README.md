# SUMO-CAV Game-Theoretic Lane Change Simulation

Full-CAV (Connected Automated Vehicle) traffic simulation in SUMO under an accident-induced lane closure scenario, featuring game-theoretic lane change decision-making with Level-k cognitive hierarchy, Stackelberg game, and queue coordination.

## Overview

A three-lane highway section (E0) with a mid-block accident blocking the middle lane (lane 1) at ~3000m, forcing a bidirectional merge. All vehicles are CAVs equipped with V2X communication, perception noise models, and multi-stage lane change decision logic.

### Accident Phases

| Phase | Timing | Behavior |
|-------|--------|----------|
| Normal | t < 90s | Free-flow cruising with CACC-like car-following |
| Sudden | 90-100s | Accident occurs; local V2X only; high urgency |
| Informed | 100s+ | Global V2X broadcast active; orderly evacuation |

## Key Features

### 1. Level-k Cognitive Hierarchy (`assign_level_k`)
Vehicles are assigned cognitive levels (0/1/2) that govern how they model opponent behavior. Level-1 vehicles sharpen their predictions using the knowledge that others are Level-0; Level-2 vehicles account for recursive reasoning.

### 2. Stackelberg Game (`compute_stackelberg_payoff`)
Sequential leader-follower game: the lane-changing vehicle (leader) commits to a decision, and the target-lane follower responds optimally. The leader's payoff is evaluated under the follower's best response.

### 3. Queue Coordination (`build_lc_queue`)
When queue coordination is enabled, vehicles on the blocked lane are sorted by distance from the obstacle. Up to `MAX_QUEUE_ALLOWED` vehicles (default: 3) may attempt lane changes per step, preventing single-vehicle deadlocks.

### 4. Emergency Braking Coverage
A safety overlay on the blocked lane: vehicles approaching the obstacle are speed-capped by a safe-speed envelope and force-braked within the final 95m. Lane changes are prohibited within `EMERGENCY_NO_LC_DIST` (90m) of the obstacle.

### 5. Cooperative Yielding
Target-lane followers may receive a cooperative deceleration request to open a gap. Once the gap reaches `COOP_MIN_GAP`, the lane change proceeds, and the supporter resumes normal control.

## Configuration

All parameters are in `config.py`, organized into sections:

- **Simulation**: step length, total steps
- **Accident**: location, lane, timing
- **Communication**: V2X range, packet loss, perception noise
- **Safety**: TTC thresholds, minimum gaps
- **Emergency braking**: reaction time, deceleration, coverage zones
- **Lane change game**: gain thresholds, cost, cooldown
- **Cooperative yielding**: headway thresholds, deceleration magnitude
- **Dynamics**: lateral acceleration limits, lane change duration
- **Level-k**: max cognitive level, population distribution
- **Parameter profiles**: balanced, balanced_plus, conservative, aggressive

### Parameter Profiles

| Profile | Headway (s) | Sudden Gain Threshold | LC Cost | Description |
|---------|-------------|----------------------|---------|-------------|
| balanced | 1.00 | 0.030 | 0.060 | Default, balanced efficiency and safety |
| balanced_plus | 1.18 | 0.060 | 0.100 | More conservative cruising |
| conservative | 1.25 | 0.050 | 0.090 | Larger headways, higher LC threshold |
| aggressive | 0.85 | 0.020 | 0.040 | Tighter headways, more aggressive merging |

## Usage

### Prerequisites

- SUMO 1.x installed (default path: `C:\Program Files (x86)\Eclipse\Sumo`)
- Python 3.10+ with `traci`, `sumolib`, `numpy`, `pandas`, `matplotlib`

### Run Full Simulation

```bash
python game_lane_change.py
```

Select parameter profile(s) when prompted, or pipe input:

```bash
# Single profile
echo "b" | PYTHONIOENCODING=utf-8 python game_lane_change.py

# All profiles comparison
echo "all" | PYTHONIOENCODING=utf-8 python game_lane_change.py
```

### Run Quick LKSQ Test

```bash
python test_lksq_fix.py
```

Runs both Original Game and LKSQ modes at 1200 pcu/h and reports lane change counts.

### Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `SIM_STEPS` | 3600 | Total simulation steps |
| `SKIP_GUI_DEMO` | 0 | Set to `1` to skip interactive GUI demo |
| `PYTHONIOENCODING` | - | Set to `utf-8` to fix GBK encoding errors on Windows |

## Output

Results are saved to `results/<timestamp>/`:

- `results_<timestamp>.csv` — aggregated metrics per scenario
- `lanechange_dynamics_<timestamp>.csv` — per-vehicle lane change logs
- `comparison_<timestamp>.png` — 4-panel bar chart (throughput, travel time, delay, queue)
- `queue_timeseries_<timestamp>.png` — queue length over time
- `speed_timeseries_<timestamp>.png` — average speed over time
- `phase_lanechange_<timestamp>.png` — sudden vs. informed phase lane change counts
- `coop_metrics_<timestamp>.png` — cooperative behavior statistics
- `robustness_metrics_<timestamp>.png` — jerk/acc violations, energy, comm load

### Multi-profile Comparison

When running with `all` profiles, additional outputs include:
- `profile_comparison_<timestamp>.png` — cross-profile bar charts
- `profile_delay_by_scenario_<timestamp>.png` — delay breakdown by scenario

## Project Structure

```
SUMO-1/
├── game_lane_change.py      # Main simulation & lane change logic
├── config.py                # Centralized parameters (import-safe)
├── metrics.py               # Comfort & fairness evaluation
├── test_lksq_fix.py         # Quick LKSQ validation script
├── baseline_comparison.py   # Baseline model comparison
├── plot_baseline_results.py # Baseline result visualization
├── run_baseline_stepwise.py # Stepwise baseline runner
├── test_baselines_quick.py  # Quick baseline validation
├── accident_highway.net.xml # SUMO network
├── accident_highway.rou.xml # Route definition
├── accident_highway.sumocfg # SUMO configuration
├── viewsettings.xml         # GUI view settings
└── results/                 # Simulation output directory
```
