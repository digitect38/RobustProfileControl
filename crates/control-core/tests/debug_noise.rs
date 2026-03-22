/// Diagnostic: trace WHY there's error under ideal conditions
use control_core::simulation::{run_simulation, SimConfig};
use control_core::synth_data::*;
use control_core::types::*;
use control_core::qp::QpSolver;
use control_core::weighting::WeightConfig;

#[test]
fn ideal_conditions_produce_ideal_result() {
    let config = SimConfig {
        n_wafers: 3,
        turns_per_wafer: 160,
        disturbance_amplitude: 0.0,
        noise_amplitude: 0.0,
        enable_inrun: true,
        enable_r2r: true,
        turn_detail_every_n: 1,
        ..Default::default()
    };
    let result = run_simulation(&config);
    let ws = &result.wafer_snapshots[0];

    let r_out = radial_output_positions();
    println!("=== IDEAL CONDITIONS (zero disturbance, zero noise) ===");
    println!("RMS error:     {:.2} Å", ws.rms_error);
    println!("Profile range: {:.2} Å", ws.profile_range);
    println!("Edge error:    {:.2} Å", ws.edge_error);

    println!("\nError profile:");
    for j in (0..NY).step_by(10) {
        println!("  r={:6.1} mm  error={:8.3} Å", r_out[j], ws.final_error[j]);
    }
    println!("  r={:6.1} mm  error={:8.3} Å", r_out[NY-1], ws.final_error[NY-1]);

    // Under ideal conditions, error equals the null-space residual of
    // fitting a parabolic removal to 11 CCDF zone basis functions.
    // The 0.048 Å/turn residual accumulates over 160 turns to ~7.7 Å RMS.
    // This is the physical limit of the 11-zone carrier head geometry.
    assert!(
        ws.rms_error < 10.0,
        "Ideal RMS should be <10 Å (null-space limit), got {:.2} Å", ws.rms_error
    );
    assert!(
        ws.profile_range < 50.0,
        "Ideal range should be <50 Å, got {:.2} Å", ws.profile_range
    );
}
use nalgebra::{DMatrix, DVector};

#[test]
fn diagnose_ideal_single_step() {
    let g0 = generate_synthetic_g0();
    let bounds = default_actuator_bounds();
    let initial = generate_initial_profile();
    let target = generate_target_profile();
    let total_removal = initial - target;  // what we need to remove total

    // Per-turn removal for 160 turns
    let n = DEFAULT_TURNS_PER_WAFER as f64;
    let per_turn_removal = total_removal / n;

    println!("=== PER-TURN REMOVAL NEEDED ===");
    let r_out = radial_output_positions();
    println!("Center (r=0):  {:.2} Å/turn", per_turn_removal[0]);
    println!("Middle (r=75): {:.2} Å/turn", per_turn_removal[NY/2]);
    println!("Edge (r=150):  {:.2} Å/turn", per_turn_removal[NY-1]);

    // Solve unconstrained least-squares: u* = (G^T G)^{-1} G^T r
    let g_dyn = DMatrix::from_fn(NY, NU, |r, c| g0[(r, c)]);
    let r_dyn = DVector::from_fn(NY, |i, _| per_turn_removal[i]);

    let gtg = g_dyn.transpose() * &g_dyn;
    let gtr = g_dyn.transpose() * &r_dyn;

    // Solve via Cholesky
    let u_ls = gtg.clone().cholesky().unwrap().solve(&gtr);
    let removal_ls = &g_dyn * &u_ls;
    let residual_ls = &r_dyn - &removal_ls;
    let rms_ls = (residual_ls.norm_squared() / NY as f64).sqrt();

    println!("\n=== UNCONSTRAINED LEAST-SQUARES ===");
    println!("Pressure: {:?}", u_ls.as_slice().iter().map(|v| format!("{:.3}", v)).collect::<Vec<_>>());
    println!("RMS residual: {:.6} Å/turn", rms_ls);
    println!("After 160 turns accumulated: {:.2} Å", rms_ls * n);

    // Check if any pressure is negative (infeasible)
    let any_negative = u_ls.iter().any(|&v| v < 0.0);
    println!("Any negative pressure: {}", any_negative);

    // Now solve with QP (constrained)
    let weights = WeightConfig::default_cmp();
    let w_e = weights.build_we();
    let w_u = weights.build_wu();
    let w_du = weights.build_wdu();
    let u_prev = DVector::from_fn(NU, |_, _| NOMINAL_PRESSURE);
    let qp = QpSolver::default();
    let prob = QpSolver::build_cmp_qp(&g_dyn, &r_dyn, &u_prev, &w_e, &w_u, &w_du, &bounds);
    let sol = qp.solve(&prob);
    let removal_qp = &g_dyn * &sol.x;
    let residual_qp = &r_dyn - &removal_qp;
    let rms_qp = (residual_qp.norm_squared() / NY as f64).sqrt();

    println!("\n=== CONSTRAINED QP ===");
    println!("Pressure: {:?}", sol.x.as_slice().iter().map(|v| format!("{:.3}", v)).collect::<Vec<_>>());
    println!("Converged: {}, iterations: {}", sol.converged, sol.iterations);
    println!("RMS residual: {:.6} Å/turn", rms_qp);
    println!("After 160 turns: {:.2} Å", rms_qp * n);

    // Bounds check
    for i in 0..NU {
        let at_lb = (sol.x[i] - prob.lb[i]).abs() < 1e-6;
        let at_ub = (sol.x[i] - prob.ub[i]).abs() < 1e-6;
        if at_lb || at_ub {
            println!("Zone {} AT BOUND: {:.3} (lb={:.3}, ub={:.3})",
                i, sol.x[i], prob.lb[i], prob.ub[i]);
        }
    }

    // Show the actual per-turn null-space residual
    println!("\n=== NULL-SPACE RESIDUAL (per turn) ===");
    for j in (0..NY).step_by(10) {
        println!("r={:6.1} mm  needed={:.4}  achieved={:.4}  residual={:.4} Å/turn",
            r_out[j], r_dyn[j], removal_qp[j], residual_qp[j]);
    }
    println!("r={:6.1} mm  needed={:.4}  achieved={:.4}  residual={:.4} Å/turn",
        r_out[NY-1], r_dyn[NY-1], removal_qp[NY-1], residual_qp[NY-1]);
}
