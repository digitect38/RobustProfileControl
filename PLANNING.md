# CMP Profile Control H-infinity — Implementation & Test Plan

## 1. Project Overview

This project implements a **hierarchical robust CMP (Chemical Mechanical Planarization) profile control system** based on the specification in `Plan/robust_cmp_profile_control_4_2_en.tex`.

**Core Problem**: Control 11 pressure inputs (10 carrier zones + retaining ring) to achieve a desired 101-point radial thickness profile on semiconductor wafers, using a hierarchical architecture with:

- **InRun (in-wafer) control**: Fast loop, turn-wise direct QP solve within a single wafer
- **R2R (run-to-run) control**: Supervisory layer, wafer-to-wafer recipe adaptation

**Technology Stack**: Rust (core algorithms) → WebAssembly → Web frontend (HTML/CSS/JS + Chart.js)

### Physical CMP Parameters

| Parameter | Value | Unit |
|-----------|-------|------|
| Initial wafer thickness | 10,000 | Å |
| Initial profile range | 1,000 (center-thick, deterministic) | Å |
| Target thickness | 2,000 | Å |
| Target profile range | ≤ 50 | Å |
| Nominal removal rate | 50 | Å/turn |
| Turn duration | 1 | sec |
| Turns per wafer | 160 | (8,000 Å / 50 Å) |
| Wafer radius | 150 | mm |
| Measurement points | 101 | (0–150 mm, 1.5 mm spacing) |

### Carrier Head Zone Geometry

The carrier is a **circular membrane** (R = 150 mm). Zone 1 is a center disk spanning -30 to +30 mm across the diameter. Zones 2–10 are concentric annuli. The retaining ring is outside the wafer edge.

| Zone | Inner (mm) | Outer (mm) | Width (mm) |
|------|-----------|-----------|-----------|
| Z1 (center disk) | 0 | 30 | 30 |
| Z2 | 30 | 50 | 20 |
| Z3 | 50 | 70 | 20 |
| Z4 | 70 | 90 | 20 |
| Z5 | 90 | 110 | 20 |
| Z6 | 110 | 120 | 10 |
| Z7 | 120 | 130 | 10 |
| Z8 | 130 | 140 | 10 |
| Z9 | 140 | 145 | 5 |
| Z10 | 145 | 150 | 5 |
| RR (retaining ring) | 150 | 170 | 20 |

---

## 2. Architecture

```
┌───────────────────────────────────────────────────────────────────┐
│                      Web Frontend (JS + Chart.js)                 │
│  ┌────────┐ ┌───────────┐ ┌──────────┐ ┌────────┐ ┌───────────┐ │
│  │Profile │ │G₀ Influen.│ │Zone Test │ │Pressure│ │Error+P(t) │ │
│  │Animate │ │  11 curves│ │Interactive│ │Bar+Time│ │R2R+InRun  │ │
│  └────────┘ └───────────┘ └──────────┘ └────────┘ └───────────┘ │
│  ┌─────┐ ┌─────────┐                                             │
│  │ SVD │ │ Metrics │  ← 7 tabs total                             │
│  └─────┘ └─────────┘                                             │
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
│  │  R2R Controller  │  │  InRun: Direct QP per turn             │ │
│  │  (supervisory,   │──│  (no integral — trajectory provides    │ │
│  │   delayed metro) │  │   reference; stable pressure output)   │ │
│  └──────────────────┘  └────────────────────────────────────────┘ │
│                                                                    │
│  ┌──────────────────┐  ┌────────────────────────────────────────┐ │
│  │    Observer       │  │     Simulation Engine                  │ │
│  │ (Kalman, delay    │──│  (removal model: thick -= G·u + d,    │ │
│  │  roll-forward)    │  │   deterministic initial profile)      │ │
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
  Sign flipped at pivot point (146 mm) using smooth tanh transition:
    sign_factor = −tanh((r − 146) / 2)
    G₀_rr = envelope × sign_factor
  Result: positive (redistribution) for r < 146 mm,
          negative (rebound) for r > 146 mm.
```

**Key properties**:
- Zone 1 (center disk): integrates from -30 to +30 mm — full disk coverage
- CCDF gives smooth step transitions at zone boundaries (not Gaussian PDF peaks)
- Edge reflection (method of images at R = 150 mm) enforces Neumann boundary
- RR with σ = 3 mm: tightly localized edge-profile correction actuator
- Scaled so nominal 3.5 psi uniform pressure → ~50 Å/turn average removal

**Initial profile**: Deterministic center-thick parabolic pattern. Average = 10,000 Å, range = 1,000 Å exactly. No random variation — same for every wafer.

### 3.2 SVD Dimension Reduction

```
G₀ ∈ R^{101×11}  →  G₀ = U Σ V^T  →  G₀ ≈ U_rc Σ_rc V_rc^T
z_k = U_rc^T y_k  (reduced coords, rc ≤ 11)
y_k^⊥ = (I − U_rc U_rc^T) y_k  (uncontrollable residual)
```

### 3.3 Box-Constrained QP Solver

Projected gradient descent with Barzilai-Borwein adaptive step size:
```
min 0.5 x^T H x + f^T x   s.t. lb ≤ x ≤ ub

H = G^T W_e^T W_e G + W_u^T W_u + W_Δu^T W_Δu
f = −G^T W_e^T W_e r − W_Δu^T W_Δu u_prev

Effective bounds (merging absolute + slew):
  lb[i] = max(u_min[i], u_prev[i] + Δu_min[i])
  ub[i] = min(u_max[i], u_prev[i] + Δu_max[i])
```

### 3.4 InRun Control — Direct QP per Turn (No Integral Action)

At each turn j within wafer k:
1. Compute trajectory target: `r_{k,j+1} = initial × (1 − j/N) + target × (j/N)`
2. Desired removal: `removal_needed = measured_thickness − r_{k,j+1}`
3. Solve QP: `min |W_e(removal_needed − G·u)|² + |W_u·u|² + |W_Δu·(u − u_prev)|²`
4. Apply: `thickness -= G · u + disturbance`

No integral state accumulation — the trajectory reference already encodes the desired removal schedule. This produces **stable, smooth pressure trajectories** without windup.

### 3.5 Hierarchical Control Loop

```
For each wafer k:
  initial_profile = deterministic center-thick (10,000 Å avg, 1,000 Å range)
  recipe = R2R.current_recipe  (or nominal 3.5 psi if R2R disabled)
  thickness = initial_profile

  For each turn j = 0..160:
    trajectory_target = linear interpolation(initial → 2000 Å)
    removal_needed = thickness − trajectory_target
    u = QP_solve(removal_needed)           // direct QP, no integral
    thickness −= G · u + disturbance       // Preston removal model

  R2R.step(target=2000Å, metrology=thickness)  // with delay
```

---

## 4. Web Frontend — 7 Tabs

| Tab | Description |
|-----|-------------|
| **Profile** | Wafer thickness profile (-150 to +150 mm). Shows initial, trajectory, current, target. Animate button for turn-by-turn playback. |
| **G₀ Influence** | All 11 zone influence curves on -150 to +150 mm diameter. Zone coverage bar chart showing radial extents. |
| **Zone Test** | Interactive zone-by-zone pressure test. 11 sliders (realtime plot on drag). Solo sweep mode. Configurable Y-max scale. |
| **Pressure** | Bar chart of current pressure recipe (11 zones). |
| **Error** | R2R wafer-level RMS + range. InRun turn-level RMS + range. **Pressure vs. time** chart showing all 11 zone pressures over turns. |
| **SVD** | Singular values (log scale), cumulative energy capture, profile-space mode shapes. |
| **Metrics** | Summary cards (RMS, range, edge error, removal rate, polish time, saturation count, SVD energy). Per-wafer table. Saturation histogram. |

---

## 5. Test Strategy — 69 Tests

### 5.1 Unit Tests (28 tests)

| Module | Tests | What they verify |
|--------|-------|-----------------|
| synth_data | 6 | G₀ shape/rank, removal rate at nominal (~50 Å/turn), deterministic initial profile (exact avg & range), thickness trajectory, bounds feasibility, RNG determinism |
| svd | 4 | Dimensions, descending singular values, monotone energy ratios, projection complement |
| plant | 2 | Apply positivity, clamp with slew limits |
| qp | 3 | Unconstrained 2D, constrained 2D, 11D CMP QP build+solve |
| weighting | 1 | Default weight dimensions and edge weighting |
| antiwindup | 2 | No-saturation passthrough, saturation reduces integration |
| observer | 2 | Creation, predict/update cycle |
| inrun | 2 | Step output within bounds, reset state |
| r2r | 1 | Convergence over 30 wafers (final < initial error) |
| simulation | 4 | Default run, thickness decreases, physical units correct, wear drift runs |
| generalized_plant | 1 | Matrix dimensions consistency |

### 5.2 Integration Tests (41 tests)

**QP Solver — Numerical Correctness (5 tests)**
- `qp_known_solution_2d`, `qp_active_lower_bound`, `qp_active_upper_and_lower`
- `qp_11d_full_cmp_problem`: Full 11-variable CMP problem, convergence + bounds
- `qp_tight_bounds_feasibility`: Near-degenerate bounds

**SVD — Mathematical Properties (5 tests)**
- `svd_orthogonality_u_rc`, `svd_orthogonality_v_rc`: Orthonormality to machine precision
- `svd_reconstruction_error`: Full-rank ||G₀ − UΣV^T|| < 1e-10
- `svd_truncation_captures_dominant_energy`: 6 modes capture >90%
- `svd_residual_orthogonal_to_subspace`: U_rc^T × residual = 0

**Plant Model (3 tests)**
- `plant_g0_physical_properties`: Carrier zone peaks near centers, RR strongest at edge & near-zero at center
- `plant_clamp_respects_bounds`, `plant_incremental_step_consistency`

**Effective Bounds (2 tests)**
- `effective_bounds_merge_absolute_and_slew`: Correct merge with du_max = ±0.5 psi/s
- `effective_bounds_near_lower_limit`

**Observer (2), Anti-Windup (2), InRun (2), R2R (2)**: Convergence, bounds, stability

**Simulation Engine — End-to-End (11 tests)**
- Default config, InRun-only, R2R-only, no controllers, wear drift
- Hierarchical ≤ R2R-only error, zero-disturbance convergence
- High disturbance bounded, seed reproducibility, metrology delay

**Weighting, Synthetic Data, Generalized Plant (7 tests)**

### 5.3 Running Tests

```bash
cargo test                              # All 69 tests
cargo test --lib                        # 28 unit tests only
cargo test --test integration_tests     # 41 integration tests only
cargo test -- --nocapture               # With stdout output
```

---

## 6. Build & Run

```bash
# Build native
cargo build

# Run all 69 tests
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
│   │   │   ├── types.rs                 # Vec types, ZoneGeometry, PRESSURE_SIGMA, constants
│   │   │   ├── synth_data.rs            # G₀ (CCDF+Preston), initial profile, disturbances
│   │   │   ├── svd.rs                   # SVD decomposition, mode selection
│   │   │   ├── plant.rs                 # Plant model (apply, clamp, incremental step)
│   │   │   ├── qp.rs                    # Box-constrained QP solver
│   │   │   ├── weighting.rs             # Diagonal weight matrices
│   │   │   ├── antiwindup.rs            # Back-calculation anti-windup
│   │   │   ├── observer.rs              # Kalman observer with delay roll-forward
│   │   │   ├── inrun.rs                 # InRun controller (QP-based)
│   │   │   ├── r2r.rs                   # R2R supervisory controller
│   │   │   ├── generalized_plant.rs     # H∞ generalized plant P
│   │   │   └── simulation.rs            # Full simulation engine
│   │   └── tests/
│   │       └── integration_tests.rs     # 41 integration tests
│   └── wasm-bridge/
│       ├── Cargo.toml
│       └── src/lib.rs                   # WASM exports
└── web/
    ├── index.html                       # 7-tab dashboard
    ├── style.css
    └── js/
        ├── app.js                       # WASM init, event wiring, zone test setup
        ├── charts.js                    # All Chart.js chart functions
        ├── controls.js                  # Parameter panel ↔ SimConfig
        └── simulation.js               # Simulation orchestration, animation
```

---

## 8. Future Work

| Phase | Description | Status |
|-------|-------------|--------|
| Phase 1 | InRun sensing & identification | Synthetic CCDF+Preston model implemented |
| Phase 2 | InRun constrained baseline | **Implemented** (direct QP per turn) |
| Phase 3 | Supervisory R2R adaptation | **Implemented** (SVD + delayed observer) |
| Phase 4 | Integrated robust controller | **Implemented** (hierarchical sim) |
| Phase 5 | Scheduled/adaptive MPC | Future: online model update, preview |

### Remaining for production:
- Real in-situ sensing interface and latency characterization
- Actual fab metrology delay measurement
- Real G₀ identification from experimental data (replace synthetic CCDF model)
- LMI-based H∞ controller synthesis (currently uses QP baseline)
- Gain-scheduled controller with wear-state estimation
- Robust MPC with constraints and preview information
