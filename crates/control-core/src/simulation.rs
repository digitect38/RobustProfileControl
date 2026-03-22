use serde::{Deserialize, Serialize};

use nalgebra::{DMatrix, DVector};

use crate::plant::Plant;
use crate::qp::QpSolver;
use crate::r2r::R2RController;
use crate::svd::{SvdDecomposition, SvdInfo};
use crate::synth_data::*;
use crate::types::*;
use crate::weighting::WeightConfig;

/// Configuration for a simulation run
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct SimConfig {
    pub n_wafers: usize,
    pub turns_per_wafer: usize,
    pub metrology_delay: usize,
    pub rc: usize,
    pub enable_inrun: bool,
    pub enable_r2r: bool,
    pub enable_wear_drift: bool,
    pub wear_rate: f64,
    /// Disturbance amplitude in Å/turn
    pub disturbance_amplitude: f64,
    /// Metrology noise in Å
    pub noise_amplitude: f64,
    pub seed: u64,
    /// Record turn-level detail for every Nth wafer (0 = none)
    pub turn_detail_every_n: usize,
}

impl Default for SimConfig {
    fn default() -> Self {
        SimConfig {
            n_wafers: 30,
            turns_per_wafer: DEFAULT_TURNS_PER_WAFER, // 160 turns (8000Å / 50Å)
            metrology_delay: 1,
            rc: 8,
            enable_inrun: true,
            enable_r2r: true,
            enable_wear_drift: false,
            wear_rate: 0.02,
            disturbance_amplitude: 3.0, // ±3 Å/turn disturbance
            noise_amplitude: 5.0,       // ±5 Å metrology noise
            seed: 42,
            turn_detail_every_n: 5,
        }
    }
}

/// Snapshot of one turn within a wafer (all values in Å)
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct TurnSnapshot {
    pub wafer: usize,
    pub turn: usize,
    /// Current thickness profile (21 values, Å)
    pub profile: Vec<f64>,
    /// Error from target trajectory (Å)
    pub error: Vec<f64>,
    /// Pressure command (11 values, psi)
    pub pressure: Vec<f64>,
    /// RMS thickness error (Å)
    pub rms_error: f64,
    /// Profile range: max - min (Å)
    pub profile_range: f64,
    /// Time in seconds from wafer start
    pub time_sec: f64,
}

/// Snapshot at wafer level (all values in Å)
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct WaferSnapshot {
    pub wafer: usize,
    /// Final thickness profile (Å)
    pub final_profile: Vec<f64>,
    /// Target thickness profile (Å)
    pub target_profile: Vec<f64>,
    /// Error = target - final (Å)
    pub final_error: Vec<f64>,
    /// R2R baseline recipe (psi)
    pub recipe: Vec<f64>,
    /// RMS thickness error (Å)
    pub rms_error: f64,
    /// Edge error: RMS of edge points (Å)
    pub edge_error: f64,
    /// Final profile range: max - min (Å)
    pub profile_range: f64,
    /// Reduced SVD coordinates
    pub reduced_coords: Vec<f64>,
    /// Residual energy (Å²)
    pub residual_energy: f64,
    /// Saturation event count
    pub saturation_count: usize,
    /// Total polishing time (sec)
    pub polishing_time_sec: f64,
    /// Average removal rate (Å/turn)
    pub avg_removal_rate: f64,
    /// Initial thickness profile (Å)
    pub initial_profile: Vec<f64>,
}

/// Full simulation result
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct SimResult {
    pub config: SimConfig,
    pub wafer_snapshots: Vec<WaferSnapshot>,
    pub turn_snapshots: Vec<TurnSnapshot>,
    pub svd_info: SvdInfo,
}

/// Run the full hierarchical CMP simulation with physical units.
///
/// Model:
///   thickness_{j+1} = thickness_j - G * u_j - d_j
///
/// where G is in Å/(psi·turn), u in psi, d in Å/turn.
/// The controller targets a thickness trajectory from initial to final.
pub fn run_simulation(config: &SimConfig) -> SimResult {
    let g0 = generate_synthetic_g0();
    let bounds = default_actuator_bounds();
    let weights = WeightConfig::default_cmp();
    let svd = SvdDecomposition::from_plant(&g0, config.rc);
    let svd_info = svd.to_info();

    let mut plant = Plant::new(g0, bounds.clone());

    // Direct QP solver for InRun turn-wise removal-rate tracking.
    // No integral action — the trajectory provides the reference directly.
    let qp = QpSolver::default();
    let g_dyn = DMatrix::from_fn(NY, NU, |r, c| g0[(r, c)]);
    let w_e = weights.build_we();
    let w_u = weights.build_wu();
    let w_du = weights.build_wdu();

    let mut r2r = R2RController::new(
        &g0,
        svd.clone(),
        bounds.clone(),
        &weights,
        config.metrology_delay,
    );

    let target = generate_target_profile(); // 2000 Å flat

    // Generate disturbances (in Å/turn)
    let disturbances = generate_disturbance_sequence(
        config.n_wafers,
        config.turns_per_wafer,
        config.disturbance_amplitude,
        config.seed,
    );

    let mut rng = SimpleRng::new(config.seed.wrapping_add(999));

    let mut wafer_snapshots = Vec::with_capacity(config.n_wafers);
    let mut turn_snapshots = Vec::new();

    for k in 0..config.n_wafers {
        // Apply wear drift if enabled
        if config.enable_wear_drift {
            let wear_frac = config.wear_rate * k as f64;
            let pad_delta = generate_pad_wear_perturbation(wear_frac);
            let rr_delta = generate_rr_wear_perturbation(wear_frac * 0.5);
            plant.set_plant_matrix(g0 + pad_delta + rr_delta);
        }

        // Generate incoming wafer thickness profile
        let initial_profile = generate_initial_profile();
        let mut thickness = initial_profile;

        // R2R provides baseline recipe
        let recipe = if config.enable_r2r {
            r2r.current_recipe
        } else {
            Vec11::from_element(NOMINAL_PRESSURE)
        };

        let mut saturation_count: usize = 0;

        let record_turns = config.turn_detail_every_n > 0
            && k % config.turn_detail_every_n == 0;

        // Phase 1: Compute the optimal steady-state pressure for this wafer.
        // Solve constrained least-squares: min |r - G·u|² s.t. u_min ≤ u ≤ u_max
        // No slew limits, no effort/slew penalty — pure tracking.
        let per_turn_removal = (initial_profile - target) / (config.turns_per_wafer as f64);
        let steady_u = if config.enable_inrun {
            let removal_dyn = DVector::from_fn(NY, |i, _| per_turn_removal[i]);

            // Unconstrained least-squares: u = (G^T G)^{-1} G^T r
            // Then clamp to actuator bounds.
            let gtg = g_dyn.transpose() * &g_dyn;
            let gtr = g_dyn.transpose() * &removal_dyn;
            let u_ls = gtg.cholesky()
                .expect("G^T G should be positive definite")
                .solve(&gtr);

            // Clamp to absolute bounds
            Vec11::from_fn(|i, _| u_ls[i].clamp(bounds.u_min[i], bounds.u_max[i]))
        } else {
            recipe
        };

        let mut u_prev = steady_u;

        // Phase 2: Turn-by-turn loop.
        // Start from the optimal steady-state. Only re-solve the QP when
        // there's actual disturbance or noise to correct for.
        let has_corrections = config.disturbance_amplitude > 0.0
            || config.noise_amplitude > 0.0
            || config.enable_wear_drift;

        for j in 0..config.turns_per_wafer {
            let u = if config.enable_inrun {
                if has_corrections {
                    // Real-time correction: re-solve QP with measured thickness
                    let trajectory_target = generate_thickness_trajectory(
                        &initial_profile, &target,
                        config.turns_per_wafer, j + 1,
                    );
                    let noise = rng.random_vec21(config.noise_amplitude);
                    let measured_thickness = thickness + noise;
                    let removal_needed = measured_thickness - trajectory_target;

                    let removal_dyn = DVector::from_fn(NY, |i, _| removal_needed[i]);
                    let u_prev_dyn = DVector::from_fn(NU, |i, _| u_prev[i]);
                    let prob = QpSolver::build_cmp_qp(
                        &g_dyn, &removal_dyn, &u_prev_dyn,
                        &w_e, &w_u, &w_du, &bounds,
                    );
                    let sol = qp.solve(&prob);
                    let (lb, ub) = bounds.effective_bounds(&u_prev);
                    for i in 0..NU {
                        if (sol.x[i] - lb[i]).abs() < 1e-6 || (sol.x[i] - ub[i]).abs() < 1e-6 {
                            saturation_count += 1;
                            break;
                        }
                    }
                    Vec11::from_fn(|i, _| sol.x[i])
                } else {
                    // Ideal conditions: use optimal steady-state pressure
                    steady_u
                }
            } else {
                recipe
            };

            // Apply removal: thickness -= G * u + disturbance
            let removal = plant.g * u;
            u_prev = u;
            let d_j = if j < disturbances[k].len() {
                disturbances[k][j]
            } else {
                Vec21::zeros()
            };
            thickness = thickness - removal - d_j;

            let current_pressure = u;

            // Record turn snapshot if requested
            if record_turns {
                let error = target - thickness;
                let rms = (error.norm_squared() / NY as f64).sqrt();
                let t_min = thickness.iter().cloned().fold(f64::MAX, f64::min);
                let t_max = thickness.iter().cloned().fold(f64::MIN, f64::max);
                turn_snapshots.push(TurnSnapshot {
                    wafer: k,
                    turn: j,
                    profile: thickness.as_slice().to_vec(),
                    error: error.as_slice().to_vec(),
                    pressure: current_pressure.as_slice().to_vec(),
                    rms_error: rms,
                    profile_range: t_max - t_min,
                    time_sec: (j + 1) as f64 * TURN_DURATION,
                });
            }
        }

        // End of wafer: record wafer snapshot
        let final_error = target - thickness;
        let rms = (final_error.norm_squared() / NY as f64).sqrt();

        // Edge error: RMS of last 4 points (indices 17-20)
        // Edge error: RMS of points with r > 130 mm
        let edge_start = (NY * 130) / 150; // index for r ≈ 130 mm
        let edge_count = NY - edge_start;
        let edge_err: f64 = (edge_start..NY)
            .map(|i| final_error[i] * final_error[i])
            .sum::<f64>()
            / edge_count.max(1) as f64;
        let edge_error = edge_err.sqrt();

        let t_min = thickness.iter().cloned().fold(f64::MAX, f64::min);
        let t_max = thickness.iter().cloned().fold(f64::MIN, f64::max);
        let profile_range = t_max - t_min;

        let total_removed = initial_profile - thickness;
        let avg_removal: f64 = total_removed.iter().sum::<f64>() / NY as f64;
        let avg_removal_rate = avg_removal / config.turns_per_wafer as f64;

        let reduced_coords = svd.project_to_reduced(&final_error);
        let residual_energy = svd.residual_energy(&final_error);

        wafer_snapshots.push(WaferSnapshot {
            wafer: k,
            final_profile: thickness.as_slice().to_vec(),
            target_profile: target.as_slice().to_vec(),
            final_error: final_error.as_slice().to_vec(),
            recipe: recipe.as_slice().to_vec(),
            rms_error: rms,
            edge_error,
            profile_range,
            reduced_coords: reduced_coords.as_slice().to_vec(),
            residual_energy,
            saturation_count,
            polishing_time_sec: config.turns_per_wafer as f64 * TURN_DURATION,
            avg_removal_rate,
            initial_profile: initial_profile.as_slice().to_vec(),
        });

        // R2R update using post-metrology (with delay)
        if config.enable_r2r && k >= config.metrology_delay {
            let metro_wafer = k - config.metrology_delay;
            if metro_wafer < wafer_snapshots.len() {
                let ws = &wafer_snapshots[metro_wafer];
                let metro_profile = Vec21::from_column_slice(&ws.final_profile);
                let metro_noise = rng.random_vec21(config.noise_amplitude * 0.5);
                let noisy_metro = metro_profile + metro_noise;
                // For R2R, the "target" is the desired final thickness
                // The R2R controller adjusts baseline recipe to minimize
                // the final thickness error across wafers
                r2r.step(&target, &noisy_metro);
            }
        }
    }

    SimResult {
        config: config.clone(),
        wafer_snapshots,
        turn_snapshots,
        svd_info,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_simulation_runs() {
        let config = SimConfig {
            n_wafers: 5,
            turns_per_wafer: 80, // fewer turns for speed
            turn_detail_every_n: 2,
            ..Default::default()
        };

        let result = run_simulation(&config);
        assert_eq!(result.wafer_snapshots.len(), 5);
        assert!(!result.turn_snapshots.is_empty());
        assert_eq!(result.svd_info.singular_values.len(), NU);
    }

    #[test]
    fn test_thickness_decreases() {
        let config = SimConfig {
            n_wafers: 3,
            turns_per_wafer: 100,
            disturbance_amplitude: 0.0,
            noise_amplitude: 0.0,
            ..Default::default()
        };

        let result = run_simulation(&config);
        for ws in &result.wafer_snapshots {
            let avg_final: f64 = ws.final_profile.iter().sum::<f64>() / NY as f64;
            let avg_initial: f64 = ws.initial_profile.iter().sum::<f64>() / NY as f64;
            assert!(
                avg_final < avg_initial,
                "Thickness should decrease: initial={:.0}, final={:.0}",
                avg_initial, avg_final
            );
        }
    }

    #[test]
    fn test_physical_units() {
        let config = SimConfig {
            n_wafers: 2,
            turns_per_wafer: DEFAULT_TURNS_PER_WAFER,
            disturbance_amplitude: 1.0,
            noise_amplitude: 2.0,
            ..Default::default()
        };

        let result = run_simulation(&config);
        let ws = &result.wafer_snapshots[0];

        // Final thickness should be in the ballpark of target (2000 Å)
        let avg_final: f64 = ws.final_profile.iter().sum::<f64>() / NY as f64;
        assert!(
            avg_final > 500.0 && avg_final < 5000.0,
            "Final thickness should be near target 2000Å, got {:.0}Å",
            avg_final
        );

        // Removal rate should be near nominal
        assert!(
            ws.avg_removal_rate > 20.0 && ws.avg_removal_rate < 100.0,
            "Removal rate should be ~50 Å/turn, got {:.1}",
            ws.avg_removal_rate
        );

        // Polishing time should be correct
        assert_eq!(ws.polishing_time_sec, DEFAULT_TURNS_PER_WAFER as f64);
    }

    #[test]
    fn test_simulation_with_wear() {
        let config = SimConfig {
            n_wafers: 5,
            turns_per_wafer: 80,
            enable_wear_drift: true,
            wear_rate: 0.05,
            ..Default::default()
        };
        let result = run_simulation(&config);
        assert_eq!(result.wafer_snapshots.len(), 5);
    }
}
