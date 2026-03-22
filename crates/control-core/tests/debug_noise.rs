/// Prove: smooth G₀ can only produce smooth profiles.
/// Any residual error must be smooth, not noisy.
use control_core::simulation::{run_simulation, SimConfig};
use control_core::synth_data::*;
use control_core::types::*;
use nalgebra::{DMatrix, DVector};

#[test]
fn compare_ideal_vs_default_disturbance() {
    // IDEAL: zero disturbance, zero noise
    let ideal = run_simulation(&SimConfig {
        n_wafers: 1, turns_per_wafer: 160,
        disturbance_amplitude: 0.0, noise_amplitude: 0.0,
        enable_inrun: true, enable_r2r: false,
        ..Default::default()
    });
    let ws_ideal = &ideal.wafer_snapshots[0];

    // DEFAULT: disturbance=3.0, noise=5.0 (what the UI shows on first load)
    let noisy = run_simulation(&SimConfig {
        n_wafers: 1, turns_per_wafer: 160,
        disturbance_amplitude: 3.0, noise_amplitude: 5.0,
        enable_inrun: true, enable_r2r: false,
        ..Default::default()
    });
    let ws_noisy = &noisy.wafer_snapshots[0];

    println!("=== WHAT YOU SEE IN THE BROWSER ===\n");
    println!("  Ideal (dist=0, noise=0):  RMS={:.1} Å  range={:.1} Å",
        ws_ideal.rms_error, ws_ideal.profile_range);
    println!("  Default (dist=3, noise=5): RMS={:.1} Å  range={:.1} Å",
        ws_noisy.rms_error, ws_noisy.profile_range);
    println!();
    println!("The DEFAULT config has disturbance_amplitude=3.0 Å/turn.");
    println!("Over 160 turns, disturbance accumulates as a random walk:");
    println!("  expected std_dev ≈ 3.0 × √160 ≈ {:.0} Å", 3.0 * (160.0_f64).sqrt());
    println!();
    println!("G₀ is smooth → removal G·u is smooth.");
    println!("But disturbance d_j is RANDOM → it adds non-smooth noise every turn.");
    println!("The controller corrects 11 controllable modes, but the");
    println!("remaining 90 null-space modes of disturbance ACCUMULATE as noise.");
    println!();

    // The default UI shows the noisy result, not the ideal one
    assert!(ws_noisy.rms_error > ws_ideal.rms_error * 2.0,
        "Disturbance should significantly increase error");
}

#[test]
fn ideal_error_is_smooth_not_noisy() {
    let config = SimConfig {
        n_wafers: 1,
        turns_per_wafer: 160,
        disturbance_amplitude: 0.0,
        noise_amplitude: 0.0,
        enable_inrun: true,
        enable_r2r: false,
        turn_detail_every_n: 1,
        ..Default::default()
    };
    let result = run_simulation(&config);
    let ws = &result.wafer_snapshots[0];
    let r_out = radial_output_positions();

    // The error profile must be SMOOTH because G₀ columns are smooth.
    // Check smoothness: the second derivative (finite differences) should be small
    // relative to the error magnitude.
    let err = &ws.final_error;
    let n = err.len();

    let mut max_d2 = 0.0_f64;
    let mut max_err = 0.0_f64;
    for j in 1..n-1 {
        let d2 = (err[j+1] - 2.0*err[j] + err[j-1]).abs();
        max_d2 = max_d2.max(d2);
        max_err = max_err.max(err[j].abs());
    }
    let smoothness_ratio = max_d2 / max_err.max(1e-10);

    println!("=== SMOOTHNESS PROOF ===");
    println!("Max |error|:             {:.4} Å", max_err);
    println!("Max |d²error/dr²|:       {:.4} Å/pt²", max_d2);
    println!("Smoothness ratio (d2/e): {:.4}", smoothness_ratio);
    println!("(Noisy signal would have ratio >> 1; smooth has ratio << 1)");

    // A smooth signal has small second derivative relative to magnitude.
    // For truly random noise, ratio ≈ 2-3. For smooth curves, ratio < 0.5.
    assert!(
        smoothness_ratio < 0.5,
        "Error profile should be smooth (ratio={:.4}), not noisy",
        smoothness_ratio
    );

    // Print error to show it's a smooth curve
    println!("\nError profile (smooth systematic bias):");
    for j in (0..n).step_by(5) {
        println!("  r={:6.1} mm  error={:8.3} Å", r_out[j], err[j]);
    }
}

#[test]
fn null_space_residual_is_the_only_error() {
    // Under ideal conditions, the ONLY error source is the null-space residual:
    // error = Proj_{null(G^T)}(initial - target)
    // This is independent of the number of turns.

    let g0 = generate_synthetic_g0();
    let initial = generate_initial_profile();
    let target = generate_target_profile();
    let total_removal = initial - target;

    let g_dyn = DMatrix::from_fn(NY, NU, |r, c| g0[(r, c)]);
    let delta = DVector::from_fn(NY, |i, _| total_removal[i]);

    // Compute least-squares solution for total removal
    let gtg = g_dyn.transpose() * &g_dyn;
    let gtr = g_dyn.transpose() * &delta;
    let u_total = gtg.cholesky().unwrap().solve(&gtr);

    // The achievable removal
    let achievable = &g_dyn * &u_total;

    // The null-space residual (what 11 zones physically can't match)
    let null_residual = &delta - &achievable;
    let rms_null = (null_residual.norm_squared() / NY as f64).sqrt();

    println!("=== NULL-SPACE ANALYSIS ===");
    println!("Total removal needed:    {:.1} Å average", delta.mean());
    println!("Null-space residual RMS: {:.4} Å", rms_null);
    println!("As % of incoming range:  {:.2}%", rms_null / INITIAL_RANGE * 100.0);

    // Now run the simulation and verify the final error matches
    let config = SimConfig {
        n_wafers: 1,
        turns_per_wafer: 160,
        disturbance_amplitude: 0.0,
        noise_amplitude: 0.0,
        enable_inrun: true,
        enable_r2r: false,
        ..Default::default()
    };
    let result = run_simulation(&config);
    let sim_rms = result.wafer_snapshots[0].rms_error;

    println!("Simulation RMS error:    {:.4} Å", sim_rms);
    println!("Theoretical minimum:     {:.4} Å", rms_null);

    // The simulation error should equal the null-space residual
    assert!(
        (sim_rms - rms_null).abs() < 1.0,
        "Simulation error ({:.2}) should match null-space limit ({:.2})",
        sim_rms, rms_null
    );

    // The null-space error is independent of turn count (same total removal).
    // Only test with the default 160 turns since changing N changes the
    // per-turn removal magnitude and may hit actuator bounds differently.
    println!("Conclusion: ideal error = null-space residual = {:.2} Å RMS", rms_null);
    println!("This is {:.2}% of the incoming 1000 Å range — the physical limit of 11 zones.", rms_null / 10.0);
}
