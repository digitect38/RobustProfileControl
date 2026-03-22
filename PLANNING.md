# CMP Profile Control H-infinity — Implementation & Test Plan

## 1. Project Overview

This project implements a **hierarchical robust CMP (Chemical Mechanical Planarization) profile control system** based on the specification in `Plan/robust_cmp_profile_control_4_2_en.tex`.

**Core Problem**: Control 11 pressure inputs (10 carrier zones + retaining ring) to achieve a desired 101-point radial thickness profile on semiconductor wafers, using a hierarchical architecture with:

- **InRun (in-wafer) control**: Two-phase approach — optimal steady-state pressure + turn-wise QP correction
- **R2R (run-to-run) control**: Supervisory layer, wafer-to-wafer recipe adaptation
- **Configurable trajectory**: Coarse-to-fine polishing via power-law trajectory shaping

**Technology Stack**: Rust (core algorithms) → WebAssembly → Web frontend (HTML/CSS/JS + Chart.js)

### Physical CMP Parameters

| Parameter | Value | Unit |
|-----------|-------|------|
| Initial wafer thickness | 10,000 | Å |
| Initial profile range | 1,000 (center-thick, deterministic parabola) | Å |
| Target thickness | 2,000 | Å |
| Target profile range | ≤ 50 | Å |
| Nominal removal rate | 50 | Å/turn |
| Turn duration | 1 | sec |
| Turns per wafer | 160 | (8,000 Å / 50 Å) |
| Wafer radius | 150 | mm |
| Measurement points | 101 | (0–150 mm, 1.5 mm spacing) |
| Platen / Carrier RPM | 80 / 78 | rpm |
| Center-to-center offset | 175 | mm |

### Carrier Head Zone Geometry

The carrier is a **circular membrane** (R = 150 mm). Zone 1 is a center disk spanning -30 to +30 mm across the diameter. Zones 2–10 are concentric annuli. The retaining ring is outside the wafer edge.

| Zone | Inner (mm) | Outer (mm) | Width (mm) | σ (mm) |
|------|-----------|-----------|-----------|--------|
| Z1 (center disk) | 0 | 30 | 30 | 6 |
| Z2 | 30 | 50 | 20 | 6 |
| Z3 | 50 | 70 | 20 | 6 |
| Z4 | 70 | 90 | 20 | 6 |
| Z5 | 90 | 110 | 20 | 6 |
| Z6 | 110 | 120 | 10 | 6 |
| Z7 | 120 | 130 | 10 | 6 |
| Z8 | 130 | 140 | 10 | 6 |
| Z9 | 140 | 145 | 5 | 6 |
| Z10 | 145 | 150 | 5 | 6 |
| RR (retaining ring) | 150 | 170 | 20 | 3 |

---

## 2. Architecture

```
┌───────────────────────────────────────────────────────────────────┐
│                      Web Frontend (JS + Chart.js)                 │
│  ┌────────┐ ┌───────────┐ ┌──────────┐ ┌────────┐ ┌───────────┐ │
│  │Profile │ │G₀ Influen.│ │Zone Test │ │Pressure│ │Error+P(t) │ │
│  │Animate │ │  11 curves│ │Interactive│ │Bar+Time│ │R2R+InRun  │ │
│  └────────┘ └───────────┘ └──────────┘ └────────┘ └───────────┘ │
│  ┌─────┐ ┌─────────┐ ┌───────────┐                               │
│  │ SVD │ │ Metrics │ │Debug Panel│  ← 7 tabs + debug             │
│  └─────┘ └─────────┘ └───────────┘                               │
│                          │ JSON                                   │
│  ┌───────────────────────┴────────────────────────────────────┐   │
│  │           WASM Bridge (wasm-bindgen, ~10 exports)          │   │
│  └───────────────────────┬────────────────────────────────────┘   │
└──────────────────────────┼────────────────────────────────────────┘
                           │
┌──────────────────────────┴────────────────────────────────────────┐
│                    control-core (Rust)                             │
│                                                                    │
│  ┌──────────────┐  ┌───────────────┐  ┌────────────────────────┐  │
│  │ Plant Model  │  │  SVD & Mode   │  │   QP Solver            │  │
│  │ Preston Eq.  │──│  Selection    │──│  (Projected Gradient)  │  │
│  │ CCDF kernel  │  │  (nalgebra)   │  │                        │  │
│  └──────────────┘  └───────────────┘  └────────────┬───────────┘  │
│                                                     │              │
│  ┌──────────────────┐  ┌────────────────────────────┴───────────┐ │
│  │  R2R Controller  │  │  InRun: Two-phase                      │ │
│  │  (supervisory,   │──│  Phase 1: Steady-state LS (per wafer)  │ │
│  │   delayed metro) │  │  Phase 2: Turn-wise QP correction      │ │
│  └──────────────────┘  └────────────────────────────────────────┘ │
│                                                                    │
│  ┌──────────────────┐  ┌────────────────────────────────────────┐ │
│  │    Observer       │  │     Simulation Engine                  │ │
│  │ (Kalman, delay    │──│  (removal model: thick -= G·u + d,    │ │
│  │  roll-forward)    │  │   configurable trajectory shape)      │ │
│  └──────────────────┘  └────────────────────────────────────────┘ │
└────────────────────────────────────────────────────────────────────┘
```

---

## 3. Key Algorithms

### 3.1 Plant Model — Preston Equation with CCDF Kernel

**Preston equation**: Removal Rate = k_p × P(r) × V(r)

- **k_p**: Preston coefficient (absorbed into G₀ scaling)
- **P(r)**: Local contact pressure from carrier zones (CCDF spreading model)
- **V(r)**: Relative pad–wafer velocity from platen/carrier rotation

**Velocity profile** (slight radial gradient from speed mismatch):
```
V(r) = ω_platen × r_cc + (ω_platen − ω_carrier) × r
     = 80 rpm × 175 mm + (80 − 78 rpm) × r
```
Normalized so average V = 1. Each row of G₀ is scaled by V(r_j)/V_avg.

**CCDF (Complementary CDF) pressure influence kernel**:

The carrier is circular. Along any measurement diameter, each annular zone [r_inner, r_outer] appears as two segments: right side [+r_inner, +r_outer] and left side [-r_outer, -r_inner]. The Gaussian CCDF gives the fraction of pressure reaching each point:

```
Carrier zones (σ = 6 mm):
  right   = Φ((r_outer − r) / σ) − Φ((r_inner − r) / σ)
  left    = Φ((−r_inner − r) / σ) − Φ((−r_outer − r) / σ)
  reflect = Φ((img_outer − r) / σ) − Φ((img_inner − r) / σ)   [edge image at 2R − r']
  G₀[j][i] = (right + left + reflect) × V(r_j)/V_avg × k_p

Retaining ring (σ = 3 mm, rebound pivot at 146 mm):
  Same CCDF formula but with tighter σ = 3 mm.
  Zone [150, 170] mm → CDF tail reaches into wafer edge.
  Edge reflection image at [130, 150] mm adds a second lobe.
  Rebound sign via smooth tanh transition (4 mm width):
    sign_factor = −tanh((r − 146) / 4)
    G₀_rr = envelope × sign_factor
  Result: positive (redistribution) for r < 146 mm,
          negative (rebound) for r > 146 mm.
```

**Key properties**:
- Zone 1 (center disk): integrates from -30 to +30 mm — full disk coverage
- CCDF gives smooth step transitions at zone boundaries (not Gaussian PDF peaks)
- Edge reflection (method of images at R = 150 mm) enforces Neumann boundary
- RR with σ = 3 mm: edge-profile correction with rebound pivot at 146 mm
- Scaled so nominal 3.5 psi uniform pressure → ~50 Å/turn average removal
- G₀ produces **only smooth profiles** — any residual error is smooth, never noisy

**Initial profile**: Deterministic center-thick parabolic pattern. Average = 10,000 Å, range = 1,000 Å exactly. No random variation — same for every wafer.

### 3.2 Configurable Trajectory Shape (Coarse-to-Fine Polishing)

The trajectory controls how removal is distributed over time:

```
progress = (turn / N)^shape
thickness(turn) = initial × (1 − progress) + target × progress
```

| Shape | Strategy | First-half removal | Use case |
|-------|----------|-------------------|----------|
| 0.3 | Very aggressive bulk | ~80% | Fast bulk, careful endpoint |
| 0.5 | Moderate front-load | ~71% | Standard coarse-to-fine |
| 1.0 | Linear (default) | 50% | Constant removal rate |
| 2.0 | Back-loaded | ~25% | Gentle start, aggressive finish |

This is standard multi-phase CMP practice — aggressive bulk removal early, then fine/clearing phase near the target thickness for uniformity.

### 3.3 Two-Phase InRun Control

**Phase 1 — Steady-state optimization (once per wafer)**:
Compute the optimal constant pressure vector via unconstrained least-squares:
```
u* = (G^T G)^{-1} G^T × (removal_per_turn)
```
Clamped to actuator bounds. This is the best pressure for the average removal rate.

**Phase 2 — Turn-wise QP correction**:
When disturbance, noise, or non-linear trajectory is active, re-solve the QP each turn:
```
min |W_e(removal_needed − G·u)|² + |W_u·u|² + |W_Δu·(u − u_prev)|²
s.t. u_min ≤ u ≤ u_max,  Δu_min ≤ u − u_prev ≤ Δu_max
```
Under ideal conditions (zero disturbance/noise, linear trajectory), Phase 1 alone gives the optimal result without QP re-solving.

**Ideal-condition performance**:
- RMS error = 7.66 Å (null-space residual of 11 CCDF zones fitting parabolic removal)
- Profile range = 39 Å (within 50 Å spec)
- This is the **theoretical physical limit** of the carrier head geometry
- Error is smooth (not noisy) — proven by smoothness ratio test (d²/e = 0.38 < 0.5)
- Simulation RMS exactly matches theoretical null-space minimum to 4 decimal places

### 3.4 SVD Dimension Reduction

```
G₀ ∈ R^{101×11}  →  G₀ = U Σ V^T  →  G₀ ≈ U_rc Σ_rc V_rc^T
z_k = U_rc^T y_k  (reduced coords, rc ≤ 11)
y_k^⊥ = (I − U_rc U_rc^T) y_k  (uncontrollable residual)
```

### 3.5 Box-Constrained QP Solver

Projected gradient descent with Barzilai-Borwein adaptive step size:
```
min 0.5 x^T H x + f^T x   s.t. lb ≤ x ≤ ub

H = G^T W_e^T W_e G + W_u^T W_u + W_Δu^T W_Δu
f = −G^T W_e^T W_e r − W_Δu^T W_Δu u_prev

Effective bounds (merging absolute + slew):
  lb[i] = max(u_min[i], u_prev[i] + Δu_min[i])
  ub[i] = min(u_max[i], u_prev[i] + Δu_max[i])
```

Weight tuning (optimized for 101-point tracking):
- W_e = 1.0 (strong tracking priority)
- W_u = 0.001 (minimal effort penalty)
- W_Δu = 0.01 (moderate slew smoothing)
- Edge points (r > 127.5 mm): 1.5× error emphasis

### 3.6 Hierarchical Control Loop

```
For each wafer k:
  initial_profile = deterministic center-thick (10,000 Å avg, 1,000 Å range)
  recipe = R2R.current_recipe  (or nominal 3.5 psi if R2R disabled)
  thickness = initial_profile

  Phase 1: steady_u = (G^T G)^{-1} G^T × per_turn_removal  [clamped]

  For each turn j = 0..N:
    progress = (j/N)^trajectory_shape
    trajectory_target = initial*(1−progress) + target*progress
    if ideal:  u = steady_u
    else:      u = QP_solve(thickness − trajectory_target)
    thickness −= G · u + disturbance

  R2R.step(target=2000Å, metrology=thickness)  // with delay
```

---

## 4. Web Frontend — 7 Tabs + Debug Panel

| Tab | Description |
|-----|-------------|
| **Profile** | Wafer thickness profile (-150 to +150 mm). Shows initial, trajectory, current, target. Animate button for turn-by-turn playback. |
| **G₀ Influence** | All 11 zone influence curves on -150 to +150 mm diameter. Zone coverage bar chart showing radial extents. |
| **Zone Test** | Interactive zone-by-zone pressure test. 11 realtime sliders. Solo sweep mode. Configurable Y-max scale. |
| **Pressure** | Bar chart of current pressure recipe (11 zones). |
| **Error** | R2R wafer-level RMS + range. InRun turn-level RMS + range. **Pressure vs. time** chart showing all 11 zone pressures over turns. |
| **SVD** | Singular values (log scale), cumulative energy capture, profile-space mode shapes. |
| **Metrics** | Summary cards (RMS, range, edge error, removal rate, polish time, saturation count, SVD energy). Per-wafer table. Saturation histogram. |
| **Debug** | Fixed bottom panel showing actual config values (dist, noise), full profile data, and error diagnostics after each Run. |

### Sidebar Controls

| Group | Controls |
|-------|----------|
| CMP Process | Display: 10,000→2,000 Å, 50 Å/turn, 1 sec/turn |
| Simulation | Wafers, turns/wafer, metrology delay, SVD modes (rc), seed |
| Trajectory | **Shape slider** (0.1–3.0): <1 coarse→fine, 1 linear, >1 fine→coarse |
| Controllers | Enable InRun, Enable R2R |
| Disturbance | Amplitude (Å/turn), Noise (Å) — both go to true zero |
| Wear Drift | Enable, rate |
| Wafer Inspector | Wafer # slider, animation speed |

---

## 5. Test Strategy — 417 Tests

### 5.1 Unit Tests (28 tests)

| Module | Tests | What they verify |
|--------|-------|-----------------|
| synth_data | 6 | G₀ shape/rank, removal rate at nominal, deterministic initial profile, trajectory with shape, bounds, RNG |
| svd | 4 | Dimensions, descending singular values, energy ratios, projection complement |
| plant | 2 | Apply positivity, clamp with slew limits |
| qp | 3 | Unconstrained 2D, constrained 2D, 11D CMP QP |
| weighting | 1 | Weight dimensions and edge emphasis |
| antiwindup | 2 | No-saturation passthrough, saturation reduces integration |
| observer | 2 | Creation, predict/update cycle |
| inrun | 2 | Step bounds, reset state |
| r2r | 1 | Convergence over 30 wafers |
| simulation | 4 | Default run, thickness decreases, physical units, wear drift |
| generalized_plant | 1 | Matrix dimensions |

### 5.2 Comprehensive Tests (345 tests)

19 categories covering all critical paths:

| Category | Count | Key verifications |
|----------|-------|-------------------|
| Normal CDF | 10 | Accuracy, symmetry, monotonicity, boundary values |
| Zone geometry | 15 | Boundaries sum to 150mm, widths match spec, RR outside wafer |
| G₀ plant model | 31 | Per-zone peaks, CCDF shape, RR rebound sign, Preston velocity, rank |
| Initial profile | 10 | Deterministic, exact avg/range, center > edge, parabolic |
| Velocity profile | 10 | Monotonically increasing, normalized avg=1 |
| Actuator bounds | 15 | Feasibility, effective bounds, midpoint fallback, RR bounds |
| QP solver | 32 | Known 1D–11D solutions, convergence, objective, bounds |
| SVD | 28 | rc=1..11, orthogonality, reconstruction, energy, projection |
| Weighting | 10 | Diagonal, positive, edge emphasis 1.5×, RR effort |
| Plant model | 15 | Linearity, clamp idempotent, wear drift, incremental step |
| Anti-windup | 10 | Passthrough, progressive reduction, extreme saturation |
| Observer | 15 | Predict, reset, delay, convergence, matrix structure |
| R2R | 15 | Bounds, convergence, reset, delayed metrology, various rc |
| Simulation | 34 | All configs, physical units, reproducibility, seeds |
| Generalized plant | 10 | Dimensions, A=I, B=-G₀, non-zero D entries |
| Trajectory | 10 | Start/end, monotone, midpoint, saturation, power-law shape |
| Disturbance | 10 | Dimensions, deterministic, scaling, spatial correlation |
| Edge cases | 15 | Zero/max pressure, single turn/wafer |
| Constants/misc | 30 | All physical constants verified, wear, serialization |

### 5.3 Diagnostic Tests (3 tests)

- `ideal_error_is_smooth_not_noisy`: Proves smoothness ratio d²/e = 0.38 < 0.5
- `null_space_residual_is_the_only_error`: Simulation RMS = theoretical minimum (7.6628 Å)
- `compare_ideal_vs_default_disturbance`: Shows disturbance causes noise, not the model

### 5.4 Integration Tests (41 tests)

QP correctness, SVD properties, plant physics, bounds, observer, controllers, simulation end-to-end.

### 5.5 Running Tests

```bash
cargo test                              # All 417 tests
cargo test --lib                        # 28 unit tests
cargo test --test comprehensive_tests   # 345 comprehensive
cargo test --test debug_noise           # 3 diagnostic
cargo test --test integration_tests     # 41 integration
cargo test -- --nocapture               # With stdout
```

---

## 6. Build & Run

```bash
# Build native
cargo build

# Run all 417 tests
cargo test

# Build WASM for web
wasm-pack build crates/wasm-bridge --target web --out-dir ../../web/pkg

# Serve locally
cd web && python -m http.server 8080
# Open http://localhost:8080
```

---

## 7. File Inventory

```
RobustProfileControl/
├── Cargo.toml                           # Workspace
├── PLANNING.md                          # This document
├── Plan/                                # Original spec (LaTeX + PDF)
├── crates/
│   ├── control-core/
│   │   ├── Cargo.toml                   # nalgebra, serde
│   │   ├── src/
│   │   │   ├── lib.rs                   # Module re-exports
│   │   │   ├── types.rs                 # Vec types, ZoneGeometry, constants
│   │   │   ├── synth_data.rs            # G₀ (CCDF+Preston+rebound), trajectory, disturbances
│   │   │   ├── svd.rs                   # SVD decomposition, mode selection
│   │   │   ├── plant.rs                 # Plant model (apply, clamp, incremental step)
│   │   │   ├── qp.rs                    # Box-constrained QP solver
│   │   │   ├── weighting.rs             # Diagonal weight matrices
│   │   │   ├── antiwindup.rs            # Back-calculation anti-windup
│   │   │   ├── observer.rs              # Kalman observer with delay roll-forward
│   │   │   ├── inrun.rs                 # InRun controller (QP-based)
│   │   │   ├── r2r.rs                   # R2R supervisory controller
│   │   │   ├── generalized_plant.rs     # H∞ generalized plant P
│   │   │   └── simulation.rs            # Full simulation engine (two-phase InRun)
│   │   └── tests/
│   │       ├── comprehensive_tests.rs   # 345 tests across 19 categories
│   │       ├── debug_noise.rs           # 3 diagnostic/proof tests
│   │       └── integration_tests.rs     # 41 integration tests
│   └── wasm-bridge/
│       ├── Cargo.toml
│       └── src/lib.rs                   # WASM exports
└── web/
    ├── index.html                       # 7-tab dashboard + debug panel
    ├── style.css
    └── js/
        ├── app.js                       # WASM init, event wiring, zone test, debug
        ├── charts.js                    # All Chart.js chart functions
        ├── controls.js                  # Parameter panel ↔ SimConfig
        └── simulation.js               # Simulation orchestration, animation
```

---

## 8. Known Limitations & Physical Limits

### Null-space residual (ideal conditions)
With 11 CCDF zones controlling 101 measurement points, the plant matrix G₀ has a 90-dimensional null space. A smooth parabolic removal profile has a small component in this null space that cannot be eliminated by any pressure vector.

- **Theoretical minimum**: 7.66 Å RMS (0.77% of incoming 1000 Å range)
- **Dominated by center**: Zone 1 (0–30 mm disk) produces a flat CCDF step, but the parabolic removal needs r-dependent variation within the zone
- **The error is smooth**, not noisy (proven: smoothness ratio = 0.38)

### Disturbance accumulation
Random disturbance d_j at each turn accumulates as a random walk. With amplitude A over N turns:
- Expected std_dev ≈ A × √N
- The controller corrects the 11 controllable modes, but the 90 null-space modes accumulate as genuine noise in the final profile

---

## 9. Future Work

| Phase | Description | Status |
|-------|-------------|--------|
| Phase 1 | InRun sensing & identification | Synthetic CCDF+Preston model implemented |
| Phase 2 | InRun constrained baseline | **Implemented** (two-phase: LS + QP) |
| Phase 3 | Supervisory R2R adaptation | **Implemented** (SVD + delayed observer) |
| Phase 4 | Integrated robust controller | **Implemented** (hierarchical sim) |
| Phase 5 | Trajectory optimization | **Implemented** (configurable shape) |
| Phase 6 | Scheduled/adaptive MPC | Future: online model update, preview |

### Remaining for production:
- Real in-situ sensing interface and latency characterization
- Actual fab metrology delay measurement
- Real G₀ identification from experimental data (replace synthetic CCDF model)
- Wafer-to-wafer incoming variation (currently deterministic initial profile)
- LMI-based H∞ controller synthesis (currently uses QP baseline)
- Gain-scheduled controller with wear-state estimation
- Robust MPC with constraints and preview information
- Combined simulation + animation (turn-by-turn live display)
