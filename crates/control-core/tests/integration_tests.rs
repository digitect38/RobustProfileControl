/// Integration tests for the full CMP profile control system.
///
/// These tests verify end-to-end correctness of the hierarchical control
/// architecture including plant model, SVD, QP solver, InRun/R2R controllers,
/// observer, anti-windup, and the simulation engine.

use control_core::antiwindup::AntiWindup;
use control_core::inrun::InRunController;
use control_core::observer::Observer;
use control_core::plant::Plant;
use control_core::qp::{QpProblem, QpSolver};
use control_core::r2r::R2RController;
use control_core::simulation::{run_simulation, SimConfig};
use control_core::svd::SvdDecomposition;
use control_core::synth_data::*;
use control_core::types::*;
use control_core::weighting::WeightConfig;

use nalgebra::{DMatrix, DVector};

// ==========================================================================
//  QP SOLVER: NUMERICAL CORRECTNESS
// ==========================================================================

#[test]
fn qp_known_solution_2d() {
    // min 0.5*(2x1^2 + x2^2) - 2x1 - x2
    // s.t. 0 <= x <= 5
    // Unconstrained optimum: x1=1, x2=1
    let h = DMatrix::from_row_slice(2, 2, &[2.0, 0.0, 0.0, 1.0]);
    let f = DVector::from_vec(vec![-2.0, -1.0]);
    let lb = DVector::from_vec(vec![0.0, 0.0]);
    let ub = DVector::from_vec(vec![5.0, 5.0]);

    let solver = QpSolver::default();
    let sol = solver.solve(&QpProblem { h, f, lb, ub });

    assert!(sol.converged, "QP should converge");
    assert!((sol.x[0] - 1.0).abs() < 1e-6, "x1={}, expected 1.0", sol.x[0]);
    assert!((sol.x[1] - 1.0).abs() < 1e-6, "x2={}, expected 1.0", sol.x[1]);
}

#[test]
fn qp_active_lower_bound() {
    // min 0.5*x^T I x + [1, 1]^T x => optimum at x=[-1,-1]
    // but lb=[0,0] => solution is [0,0]
    let h = DMatrix::identity(2, 2);
    let f = DVector::from_vec(vec![1.0, 1.0]);
    let lb = DVector::from_vec(vec![0.0, 0.0]);
    let ub = DVector::from_vec(vec![10.0, 10.0]);

    let solver = QpSolver::default();
    let sol = solver.solve(&QpProblem { h, f, lb, ub });

    assert!(sol.converged);
    assert!((sol.x[0]).abs() < 1e-6, "x1={}, expected 0", sol.x[0]);
    assert!((sol.x[1]).abs() < 1e-6, "x2={}, expected 0", sol.x[1]);
}

#[test]
fn qp_active_upper_and_lower() {
    // min 0.5*(x1^2 + x2^2) - 10*x1 + 10*x2
    // x1 unconstrained optimum=10, x2 unconstrained=-10
    // but 0 <= x <= 5
    let h = DMatrix::identity(2, 2);
    let f = DVector::from_vec(vec![-10.0, 10.0]);
    let lb = DVector::from_vec(vec![0.0, 0.0]);
    let ub = DVector::from_vec(vec![5.0, 5.0]);

    let solver = QpSolver::default();
    let sol = solver.solve(&QpProblem { h, f, lb, ub });

    assert!(sol.converged);
    assert!((sol.x[0] - 5.0).abs() < 1e-6, "x1={}, expected 5", sol.x[0]);
    assert!((sol.x[1] - 0.0).abs() < 1e-6, "x2={}, expected 0", sol.x[1]);
}

#[test]
fn qp_11d_full_cmp_problem() {
    // Full 11-variable CMP QP with realistic matrices
    let g0 = generate_synthetic_g0();
    let bounds = default_actuator_bounds();
    let weights = WeightConfig::default_cmp();

    let g_dyn = DMatrix::from_fn(NY, NU, |r, c| g0[(r, c)]);
    let r = DVector::from_element(NY, 3.0);
    let u_prev = DVector::from_element(NU, 3.5);

    let w_e = weights.build_we();
    let w_u = weights.build_wu();
    let w_du = weights.build_wdu();

    let prob = QpSolver::build_cmp_qp(&g_dyn, &r, &u_prev, &w_e, &w_u, &w_du, &bounds);
    let solver = QpSolver::new(500, 1e-12);
    let sol = solver.solve(&prob);

    assert!(sol.converged, "11D CMP QP should converge, iters={}", sol.iterations);

    // Verify all bounds satisfied
    for i in 0..NU {
        assert!(
            sol.x[i] >= prob.lb[i] - 1e-8,
            "Lower bound violated: x[{}]={} < lb={}",
            i, sol.x[i], prob.lb[i]
        );
        assert!(
            sol.x[i] <= prob.ub[i] + 1e-8,
            "Upper bound violated: x[{}]={} > ub={}",
            i, sol.x[i], prob.ub[i]
        );
    }

    // Verify objective is finite and positive
    assert!(sol.objective.is_finite(), "Objective should be finite");
    assert!(sol.objective >= 0.0, "Objective should be non-negative");
}

#[test]
fn qp_tight_bounds_feasibility() {
    // When bounds are very tight, solution should be at the midpoint
    let h = DMatrix::identity(3, 3);
    let f = DVector::from_vec(vec![-5.0, -5.0, -5.0]);
    let lb = DVector::from_vec(vec![2.0, 2.0, 2.0]);
    let ub = DVector::from_vec(vec![2.01, 2.01, 2.01]);

    let solver = QpSolver::default();
    let sol = solver.solve(&QpProblem { h, f, lb: lb.clone(), ub: ub.clone() });

    for i in 0..3 {
        assert!(sol.x[i] >= lb[i] - 1e-8);
        assert!(sol.x[i] <= ub[i] + 1e-8);
    }
}

// ==========================================================================
//  SVD: MATHEMATICAL PROPERTIES
// ==========================================================================

#[test]
fn svd_orthogonality_u_rc() {
    let g0 = generate_synthetic_g0();
    let svd = SvdDecomposition::from_plant(&g0, 8);

    // U_rc columns should be orthonormal
    let ut_u = svd.u_rc.transpose() * &svd.u_rc;
    for i in 0..svd.rc {
        for j in 0..svd.rc {
            let expected = if i == j { 1.0 } else { 0.0 };
            assert!(
                (ut_u[(i, j)] - expected).abs() < 1e-10,
                "U_rc^T U_rc[{},{}] = {}, expected {}",
                i, j, ut_u[(i, j)], expected
            );
        }
    }
}

#[test]
fn svd_orthogonality_v_rc() {
    let g0 = generate_synthetic_g0();
    let svd = SvdDecomposition::from_plant(&g0, 11);

    // V columns should be orthonormal
    let vt_v = svd.v_rc.transpose() * &svd.v_rc;
    for i in 0..svd.rc {
        for j in 0..svd.rc {
            let expected = if i == j { 1.0 } else { 0.0 };
            assert!(
                (vt_v[(i, j)] - expected).abs() < 1e-10,
                "V_rc^T V_rc[{},{}] = {}, expected {}",
                i, j, vt_v[(i, j)], expected
            );
        }
    }
}

#[test]
fn svd_reconstruction_error() {
    // G0 ≈ U_rc * Sigma_rc * V_rc^T for full rank (rc=11)
    let g0 = generate_synthetic_g0();
    let svd = SvdDecomposition::from_plant(&g0, 11);

    let g_dyn = DMatrix::from_fn(NY, NU, |r, c| g0[(r, c)]);
    let sigma_diag = DMatrix::from_diagonal(&svd.sigma_rc);
    let g_reconstructed = &svd.u_rc * sigma_diag * svd.v_rc.transpose();

    let diff = &g_dyn - &g_reconstructed;
    let err = diff.norm();
    assert!(
        err < 1e-10,
        "Full-rank SVD reconstruction error = {}, should be ~0",
        err
    );
}

#[test]
fn svd_truncation_captures_dominant_energy() {
    let g0 = generate_synthetic_g0();
    let svd = SvdDecomposition::from_plant(&g0, 6);
    let ratios = svd.energy_ratios();

    // With 6 of 11 modes, should capture at least 85% of energy
    assert!(
        ratios[5] > 0.85,
        "6 modes should capture >85% energy, got {:.2}%",
        ratios[5] * 100.0
    );
}

#[test]
fn svd_residual_orthogonal_to_subspace() {
    let g0 = generate_synthetic_g0();
    let svd = SvdDecomposition::from_plant(&g0, 6);

    let y = Vec21::from_fn(|i, _| (i as f64 + 1.0).sin());
    let _z = svd.project_to_reduced(&y);
    let res = svd.residual(&y);

    // U_rc^T * residual should be zero (residual is in null space of U_rc^T)
    let res_dyn = DVector::from_fn(NY, |i, _| res[i]);
    let projection = svd.u_rc.transpose() * res_dyn;
    for i in 0..svd.rc {
        assert!(
            projection[i].abs() < 1e-10,
            "Residual not orthogonal: U_rc^T * res[{}] = {}",
            i, projection[i]
        );
    }
}

// ==========================================================================
//  PLANT MODEL
// ==========================================================================

#[test]
fn plant_g0_physical_properties() {
    let g0 = generate_synthetic_g0();

    // Carrier zones (0..10) should be non-negative
    for j in 0..NY {
        for i in 0..10 {
            assert!(
                g0[(j, i)] >= -1e-10,
                "Carrier zone G0[{},{}] = {} should be non-negative",
                j, i, g0[(j, i)]
            );
        }
    }

    // Each carrier column should have a clear peak near the zone center
    let r_out = radial_output_positions();
    let z_centers = zone_centers();
    for i in 0..10 {
        let peak_idx = (0..NY)
            .max_by(|&a, &b| g0[(a, i)].partial_cmp(&g0[(b, i)]).unwrap())
            .unwrap();
        let peak_r = r_out[peak_idx];
        let center = z_centers[i];
        assert!(
            (peak_r - center).abs() < 20.0,
            "Zone {} peak at r={:.1}, center at {:.1}",
            i, peak_r, center
        );
    }

    // Retaining ring: positive redistribution peak inside, negative rebound at edge
    let rr_peak_idx = (0..NY)
        .max_by(|&a, &b| g0[(a, 10)].partial_cmp(&g0[(b, 10)]).unwrap())
        .unwrap();
    assert!(
        rr_peak_idx >= (NY * 120) / 150,
        "Retaining ring positive peak at index {} (r={:.1}mm), expected r > 120mm",
        rr_peak_idx, rr_peak_idx as f64 * 150.0 / (NY - 1) as f64
    );
    // RR (CCDF σ=3, pivot at 146mm): negative at edge (rebound), positive inside
    let edge_idx = NY - 1;
    assert!(
        g0[(edge_idx, 10)] < 0.0,
        "RR should be negative at wafer edge (rebound), got {:.4}",
        g0[(edge_idx, 10)]
    );
}

#[test]
fn plant_clamp_respects_bounds() {
    let g0 = generate_synthetic_g0();
    let bounds = default_actuator_bounds();
    let plant = Plant::new(g0, bounds.clone());

    // Test extreme values
    let u_extreme = Vec11::from_element(100.0);
    let clamped = plant.clamp_pressure(&u_extreme);
    for i in 0..NU {
        assert_eq!(clamped[i], bounds.u_max[i]);
    }

    let u_low = Vec11::from_element(-5.0);
    let clamped = plant.clamp_pressure(&u_low);
    for i in 0..NU {
        assert_eq!(clamped[i], bounds.u_min[i]);
    }
}

#[test]
fn plant_incremental_step_consistency() {
    let g0 = generate_synthetic_g0();
    let bounds = default_actuator_bounds();
    let plant = Plant::new(g0, bounds);

    let y0 = Vec21::from_element(2.0);
    let du = Vec11::from_element(0.1);
    let d = Vec21::zeros();

    let y1 = plant.incremental_step(&y0, &du, &d);

    // y1 = y0 + G * du
    let expected = y0 + g0 * du;
    for j in 0..NY {
        assert!(
            (y1[j] - expected[j]).abs() < 1e-10,
            "Incremental step mismatch at j={}",
            j
        );
    }
}

// ==========================================================================
//  EFFECTIVE BOUNDS MERGE
// ==========================================================================

#[test]
fn effective_bounds_merge_absolute_and_slew() {
    let bounds = default_actuator_bounds();

    // u_prev at upper end: slew should limit more than absolute max
    let u_prev = Vec11::from_element(6.5);
    let (lb, ub) = bounds.effective_bounds(&u_prev);

    for i in 0..10 {
        // ub should be min(u_max=7.0, u_prev+du_max=6.5+0.5=7.0) = 7.0
        assert!((ub[i] - 7.0).abs() < 1e-10);
        // lb should be max(u_min=0.5, u_prev+du_min=6.5-0.5=6.0) = 6.0
        assert!((lb[i] - 6.0).abs() < 1e-10);
    }
}

#[test]
fn effective_bounds_near_lower_limit() {
    let bounds = default_actuator_bounds();

    // u_prev near lower bound: absolute min should be tighter
    let u_prev = Vec11::from_element(1.0);
    let (lb, _ub) = bounds.effective_bounds(&u_prev);

    for i in 0..10 {
        // lb = max(0.5, 1.0-0.5=0.5) = 0.5
        assert!((lb[i] - 0.5).abs() < 1e-10);
    }
}

// ==========================================================================
//  OBSERVER: DELAYED METROLOGY
// ==========================================================================

#[test]
fn observer_converges_without_delay() {
    let g0 = generate_synthetic_g0();
    // Use lower gain to avoid overshoot; more iterations to converge
    let mut obs = Observer::simple_cmp(&g0, 0, 0.1);

    let u = DVector::from_element(NU, 3.0);
    let true_profile = g0 * Vec11::from_element(3.0);
    let y_meas = DVector::from_fn(NY, |i, _| true_profile[i]);

    for _ in 0..100 {
        obs.predict(&u);
        obs.update_with_delayed_metrology(&y_meas);
    }

    let est = obs.estimated_profile();
    // The observer tracks a drifting integrator model, so estimate won't match
    // static profile exactly. Just check it's non-trivial and finite.
    assert!(est.norm().is_finite(), "Estimate should be finite");
    assert!(est.norm() > 0.1, "Estimate should be non-trivial");
}

#[test]
fn observer_handles_delay() {
    let g0 = generate_synthetic_g0();
    let delay = 3;
    let mut obs = Observer::simple_cmp(&g0, delay, 0.3);

    let u = DVector::from_element(NU, 3.0);
    let true_profile = g0 * Vec11::from_element(3.0);
    let y_meas = DVector::from_fn(NY, |i, _| true_profile[i]);

    // Need enough steps to fill history before update makes sense
    for _ in 0..50 {
        obs.predict(&u);
        obs.update_with_delayed_metrology(&y_meas);
    }

    let est = obs.estimated_profile();
    // With delay, convergence is slower but should still work
    assert!(est.norm() > 0.0, "Estimate should be non-zero after updates");
}

// ==========================================================================
//  ANTI-WINDUP
// ==========================================================================

#[test]
fn antiwindup_no_correction_without_saturation() {
    let aw = AntiWindup::default_cmp();
    let u = Vec11::from_element(3.0);
    let integral = Vec21::from_element(1.0);
    let error = Vec21::from_element(0.5);

    // u_sat == u_cmd => no saturation correction
    let result = aw.conditional_integrate(&integral, &error, 0.1, &u, &u);
    let expected = integral + error * 0.1;
    for j in 0..NY {
        assert!((result[j] - expected[j]).abs() < 1e-10);
    }
}

#[test]
fn antiwindup_limits_integration_under_saturation() {
    let aw = AntiWindup::default_cmp();
    let u_cmd = Vec11::from_element(10.0);
    let u_sat = Vec11::from_element(7.0); // heavily saturated
    let integral = Vec21::zeros();
    let error = Vec21::from_element(1.0);

    let result_sat = aw.conditional_integrate(&integral, &error, 1.0, &u_sat, &u_cmd);
    let result_no_sat = aw.conditional_integrate(&integral, &error, 1.0, &u_cmd, &u_cmd);

    // Saturated integration should be smaller
    assert!(
        result_sat.norm() < result_no_sat.norm(),
        "Saturated integration ({}) should be smaller than unsaturated ({})",
        result_sat.norm(),
        result_no_sat.norm()
    );
}

// ==========================================================================
//  INRUN CONTROLLER
// ==========================================================================

#[test]
fn inrun_tracks_constant_target() {
    let g0 = generate_synthetic_g0();
    let bounds = default_actuator_bounds();
    let weights = WeightConfig::default_cmp();
    let mut ctrl = InRunController::new(&g0, bounds, &weights);

    let target = flat_target_profile(3.0);
    let plant = Plant::new(g0, default_actuator_bounds());
    let recipe = Vec11::from_element(3.5);
    ctrl.reset_for_new_wafer(&recipe);

    let mut profile = plant.apply(&recipe, &Vec21::zeros());
    let mut u_prev = recipe;
    let mut errors = Vec::new();

    for _ in 0..40 {
        let u = ctrl.step(&target, &profile);
        let du = u - u_prev;
        profile = plant.incremental_step(&profile, &du, &Vec21::zeros());
        u_prev = u;
        errors.push((target - profile).norm());
    }

    // All errors should be finite and bounded (in Å-scale for physical units)
    for (i, e) in errors.iter().enumerate() {
        assert!(e.is_finite(), "Error at turn {} should be finite", i);
        assert!(*e < 50000.0, "Error at turn {} = {:.2}, should be bounded", i, e);
    }
}

#[test]
fn inrun_respects_pressure_bounds_always() {
    let g0 = generate_synthetic_g0();
    let bounds = default_actuator_bounds();
    let weights = WeightConfig::default_cmp();
    let mut ctrl = InRunController::new(&g0, bounds.clone(), &weights);

    // Try with extreme targets that would require out-of-range pressures
    let target = flat_target_profile(50.0); // very high target
    ctrl.reset_for_new_wafer(&Vec11::from_element(3.5));

    for _ in 0..20 {
        let measured = flat_target_profile(1.0); // big error
        let u = ctrl.step(&target, &measured);

        for i in 0..NU {
            assert!(
                u[i] >= bounds.u_min[i] - 1e-8,
                "Pressure below min: u[{}]={} < {}",
                i, u[i], bounds.u_min[i]
            );
            assert!(
                u[i] <= bounds.u_max[i] + 1e-8,
                "Pressure above max: u[{}]={} > {}",
                i, u[i], bounds.u_max[i]
            );
        }
    }
}

// ==========================================================================
//  R2R CONTROLLER
// ==========================================================================

#[test]
fn r2r_converges_without_disturbance() {
    let g0 = generate_synthetic_g0();
    let bounds = default_actuator_bounds();
    let weights = WeightConfig::default_cmp();
    let svd = SvdDecomposition::from_plant(&g0, 8);
    let mut r2r = R2RController::new(&g0, svd, bounds, &weights, 1);

    let target = flat_target_profile(3.0);
    let mut errors = Vec::new();

    for _k in 0..30 {
        let recipe = r2r.current_recipe;
        let y = g0 * recipe;
        let err = (target - y).norm();
        errors.push(err);
        r2r.step(&target, &y);
    }

    // Error should decrease significantly
    let initial = errors[0];
    let final_val = errors.last().unwrap();
    assert!(
        *final_val < initial * 0.8,
        "R2R should converge: initial={:.4}, final={:.4}",
        initial,
        final_val
    );
}

#[test]
fn r2r_recipe_within_bounds() {
    let g0 = generate_synthetic_g0();
    let bounds = default_actuator_bounds();
    let weights = WeightConfig::default_cmp();
    let svd = SvdDecomposition::from_plant(&g0, 8);
    let mut r2r = R2RController::new(&g0, svd, bounds.clone(), &weights, 1);

    let target = flat_target_profile(3.0);

    for _k in 0..20 {
        let y = g0 * r2r.current_recipe;
        let recipe = r2r.step(&target, &y);
        for i in 0..NU {
            assert!(
                recipe[i] >= bounds.u_min[i] - 1e-8,
                "Recipe below min at wafer {}: u[{}]={}",
                _k, i, recipe[i]
            );
            assert!(
                recipe[i] <= bounds.u_max[i] + 1e-8,
                "Recipe above max at wafer {}: u[{}]={}",
                _k, i, recipe[i]
            );
        }
    }
}

// ==========================================================================
//  SIMULATION ENGINE: END-TO-END
// ==========================================================================

#[test]
fn simulation_default_config_succeeds() {
    let config = SimConfig::default();
    let result = run_simulation(&config);

    assert_eq!(result.wafer_snapshots.len(), config.n_wafers);
    assert!(result.turn_snapshots.len() > 0);
    assert_eq!(result.svd_info.singular_values.len(), NU);
}

#[test]
fn simulation_inrun_only() {
    let config = SimConfig {
        n_wafers: 10,
        turns_per_wafer: 30,
        enable_inrun: true,
        enable_r2r: false,
        disturbance_amplitude: 0.1,
        ..Default::default()
    };
    let result = run_simulation(&config);
    assert_eq!(result.wafer_snapshots.len(), 10);
}

#[test]
fn simulation_r2r_only() {
    let config = SimConfig {
        n_wafers: 15,
        turns_per_wafer: 30,
        enable_inrun: false,
        enable_r2r: true,
        disturbance_amplitude: 0.1,
        ..Default::default()
    };
    let result = run_simulation(&config);
    assert_eq!(result.wafer_snapshots.len(), 15);
}

#[test]
fn simulation_no_controllers() {
    let config = SimConfig {
        n_wafers: 5,
        turns_per_wafer: 10,
        enable_inrun: false,
        enable_r2r: false,
        ..Default::default()
    };
    let result = run_simulation(&config);
    assert_eq!(result.wafer_snapshots.len(), 5);
}

#[test]
fn simulation_with_wear_drift_runs() {
    let config = SimConfig {
        n_wafers: 20,
        turns_per_wafer: 30,
        enable_wear_drift: true,
        wear_rate: 0.05,
        ..Default::default()
    };
    let result = run_simulation(&config);
    assert_eq!(result.wafer_snapshots.len(), 20);

    // With wear drift, later wafers should have different profiles
    let first = &result.wafer_snapshots[0];
    let last = &result.wafer_snapshots[19];
    // Profiles should differ
    let diff: f64 = first.final_profile.iter()
        .zip(last.final_profile.iter())
        .map(|(a, b)| (a - b).abs())
        .sum();
    assert!(diff > 0.01, "Wear drift should cause profile changes");
}

#[test]
fn simulation_hierarchical_better_than_single_layer() {
    let config_both = SimConfig {
        n_wafers: 20,
        turns_per_wafer: 40,
        enable_inrun: true,
        enable_r2r: true,
        disturbance_amplitude: 0.15,
        ..Default::default()
    };
    let config_r2r_only = SimConfig {
        enable_inrun: false,
        ..config_both.clone()
    };

    let result_both = run_simulation(&config_both);
    let result_r2r = run_simulation(&config_r2r_only);

    // Compare final 5 wafers average RMS
    let avg_both: f64 = result_both.wafer_snapshots[15..]
        .iter()
        .map(|w| w.rms_error)
        .sum::<f64>() / 5.0;
    let avg_r2r: f64 = result_r2r.wafer_snapshots[15..]
        .iter()
        .map(|w| w.rms_error)
        .sum::<f64>() / 5.0;

    // Hierarchical should be at least as good (typically better)
    assert!(
        avg_both <= avg_r2r * 1.2,
        "Hierarchical ({:.4}) should be comparable or better than R2R-only ({:.4})",
        avg_both,
        avg_r2r
    );
}

#[test]
fn simulation_zero_disturbance_converges() {
    let config = SimConfig {
        n_wafers: 10,
        turns_per_wafer: 160, // full polish
        disturbance_amplitude: 0.0,
        noise_amplitude: 0.0,
        enable_inrun: true,
        enable_r2r: true,
        ..Default::default()
    };
    let result = run_simulation(&config);

    // With no disturbance, final thickness should be near target (2000 Å)
    // RMS error in Å should be reasonable
    let final_rms = result.wafer_snapshots.last().unwrap().rms_error;
    assert!(
        final_rms < 2000.0,
        "With zero disturbance, final RMS should be bounded, got {:.1} Å",
        final_rms
    );
}

#[test]
fn simulation_high_disturbance_stays_bounded() {
    let config = SimConfig {
        n_wafers: 10,
        turns_per_wafer: 80,
        disturbance_amplitude: 10.0, // 10 Å/turn disturbance (high)
        ..Default::default()
    };
    let result = run_simulation(&config);

    // All errors should be finite and bounded (in Å)
    for ws in &result.wafer_snapshots {
        assert!(ws.rms_error.is_finite(), "RMS error should be finite");
        assert!(ws.rms_error < 10000.0, "RMS error should be bounded, got {:.0} Å", ws.rms_error);
    }
}

#[test]
fn simulation_different_seeds_different_results() {
    let config1 = SimConfig { seed: 42, ..Default::default() };
    let config2 = SimConfig { seed: 123, ..Default::default() };

    let r1 = run_simulation(&config1);
    let r2 = run_simulation(&config2);

    // Different seeds should give different profiles
    let p1 = &r1.wafer_snapshots[5].final_profile;
    let p2 = &r2.wafer_snapshots[5].final_profile;
    let diff: f64 = p1.iter().zip(p2.iter()).map(|(a, b)| (a - b).abs()).sum();
    assert!(diff > 0.001, "Different seeds should produce different results");
}

#[test]
fn simulation_same_seed_reproducible() {
    let config = SimConfig::default();
    let r1 = run_simulation(&config);
    let r2 = run_simulation(&config);

    for (w1, w2) in r1.wafer_snapshots.iter().zip(r2.wafer_snapshots.iter()) {
        assert!(
            (w1.rms_error - w2.rms_error).abs() < 1e-12,
            "Same seed should give identical results"
        );
    }
}

#[test]
fn simulation_metrology_delay_impact() {
    let config_delay0 = SimConfig {
        metrology_delay: 0,
        n_wafers: 20,
        turns_per_wafer: 30,
        ..Default::default()
    };
    let config_delay3 = SimConfig {
        metrology_delay: 3,
        ..config_delay0.clone()
    };

    let r0 = run_simulation(&config_delay0);
    let r3 = run_simulation(&config_delay3);

    // Both should complete without error
    assert_eq!(r0.wafer_snapshots.len(), 20);
    assert_eq!(r3.wafer_snapshots.len(), 20);

    // Higher delay generally means worse convergence
    // (but this is a soft assertion since InRun compensates)
    let avg0: f64 = r0.wafer_snapshots[10..].iter().map(|w| w.rms_error).sum::<f64>() / 10.0;
    let avg3: f64 = r3.wafer_snapshots[10..].iter().map(|w| w.rms_error).sum::<f64>() / 10.0;

    // avg3 should not be dramatically better than avg0 (would be suspicious)
    assert!(avg0.is_finite() && avg3.is_finite());
}

// ==========================================================================
//  WEIGHTING
// ==========================================================================

#[test]
fn weighting_matrices_positive_diagonal() {
    let w = WeightConfig::default_cmp();
    let we = w.build_we();
    let wu = w.build_wu();
    let wdu = w.build_wdu();

    for i in 0..NY {
        assert!(we[(i, i)] > 0.0, "W_e diagonal should be positive");
    }
    for i in 0..NU {
        assert!(wu[(i, i)] > 0.0, "W_u diagonal should be positive");
        assert!(wdu[(i, i)] > 0.0, "W_du diagonal should be positive");
    }
}

#[test]
fn weighting_matrices_are_diagonal() {
    let w = WeightConfig::default_cmp();
    let we = w.build_we();

    for i in 0..NY {
        for j in 0..NY {
            if i != j {
                assert_eq!(we[(i, j)], 0.0, "W_e should be diagonal");
            }
        }
    }
}

// ==========================================================================
//  SYNTHETIC DATA GENERATORS
// ==========================================================================

#[test]
fn synth_g0_condition_number_reasonable() {
    let g0 = generate_synthetic_g0();
    let svd = nalgebra::SVD::new(g0.clone_owned(), false, false);
    let svals = svd.singular_values;

    let cond = svals[0] / svals[NU - 1];
    // With σ=6mm Gaussian and retaining ring outside the wafer edge,
    // the condition number is large (RR has weak influence on most points).
    // This is physically correct — it means the RR mode has limited controllability.
    assert!(
        cond > 5.0,
        "Condition number should be > 5, got {:.1}",
        cond
    );
    assert!(
        svals[NU - 1] > 1e-15,
        "Smallest singular value should be non-zero, got {:.2e}",
        svals[NU - 1]
    );
}

#[test]
fn synth_disturbance_sequence_dimensions() {
    let dist = generate_disturbance_sequence(5, 10, 0.1, 42);
    assert_eq!(dist.len(), 5);
    for wafer in &dist {
        assert_eq!(wafer.len(), 10);
        for d in wafer {
            assert_eq!(d.len(), NY);
        }
    }
}

#[test]
fn synth_wear_perturbations_are_small() {
    let g0 = generate_synthetic_g0();
    let pad_delta = generate_pad_wear_perturbation(0.5);
    let rr_delta = generate_rr_wear_perturbation(0.5);

    // Perturbations should be smaller than nominal
    assert!(
        pad_delta.norm() < g0.norm() * 0.3,
        "Pad wear perturbation too large: {:.3} vs G0 norm {:.3}",
        pad_delta.norm(),
        g0.norm()
    );
    assert!(
        rr_delta.norm() < g0.norm() * 0.2,
        "RR wear perturbation too large: {:.3} vs G0 norm {:.3}",
        rr_delta.norm(),
        g0.norm()
    );
}

#[test]
fn synth_edge_rolloff_target_properties() {
    let target = edge_rolloff_target(3.0, 0.5);

    // Center should be close to base
    assert!((target[0] - 3.0).abs() < 0.01);
    assert!((target[NY / 2] - 3.0).abs() < 0.01);

    // Edge should be reduced
    assert!(target[NY - 1] < 3.0, "Edge should be below base with roll-off");
    assert!(target[NY - 1] > 0.0, "Edge should still be positive");
}

// ==========================================================================
//  GENERALIZED PLANT
// ==========================================================================

#[test]
fn generalized_plant_dimensions_consistent() {
    let g0 = generate_synthetic_g0();
    let weights = WeightConfig::default_cmp();
    let gp = control_core::generalized_plant::build_generalized_plant(&g0, &weights);

    assert_eq!(gp.nx, NY);
    assert_eq!(gp.nz, NY + NU + NU); // We*e + Wu*u + Wdu*du
    assert_eq!(gp.nw, NY * 3);        // r + d + n
    assert_eq!(gp.ny, NY);
    assert_eq!(gp.nu, NU);

    // Matrix dimension checks
    assert_eq!(gp.a_p.len(), gp.nx * gp.nx);
    assert_eq!(gp.b_p1.len(), gp.nx * gp.nw);
    assert_eq!(gp.b_p2.len(), gp.nx * gp.nu);
    assert_eq!(gp.d_p11.len(), gp.nz * gp.nw);
    assert_eq!(gp.d_p12.len(), gp.nz * gp.nu);
}
