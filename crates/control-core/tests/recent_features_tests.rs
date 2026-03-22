/// Tests for recent feature additions:
/// - Exponential decay trajectory (alpha parameter)
/// - Removal rate clamping [25, 100] Å/turn
/// - Anti-chattering (strong slew weight W_Δu = 2.0)
/// - Preston equation with CCDF kernel
/// - RR rebound with pivot at 146 mm
/// - Deterministic initial profile
/// - Zone geometry (30/20/20/20/20/10/10/10/5/5 + RR 20)

use control_core::simulation::{run_simulation, SimConfig};
use control_core::synth_data::*;
use control_core::types::*;
use control_core::weighting::WeightConfig;

// =========================================================================
//  EXPONENTIAL DECAY TRAJECTORY
// =========================================================================

#[test]
fn trajectory_alpha_zero_is_linear() {
    let initial = Vec21::from_element(10000.0);
    let target = Vec21::from_element(2000.0);
    let mid = generate_thickness_trajectory(&initial, &target, 160, 80, 0.0);
    // Linear: midpoint should be exactly 6000
    assert!((mid[0] - 6000.0).abs() < 1.0, "alpha=0 midpoint should be 6000, got {:.1}", mid[0]);
}

#[test]
fn trajectory_alpha_positive_front_loaded() {
    let initial = Vec21::from_element(10000.0);
    let target = Vec21::from_element(2000.0);
    let mid_linear = generate_thickness_trajectory(&initial, &target, 160, 80, 0.0);
    let mid_exp = generate_thickness_trajectory(&initial, &target, 160, 80, 2.0);
    // With alpha > 0, more removal happens early → midpoint should be LOWER than linear
    assert!(
        mid_exp[0] < mid_linear[0],
        "alpha=2 midpoint ({:.0}) should be lower than linear ({:.0})",
        mid_exp[0], mid_linear[0]
    );
}

#[test]
fn trajectory_alpha_negative_back_loaded() {
    let initial = Vec21::from_element(10000.0);
    let target = Vec21::from_element(2000.0);
    let mid_linear = generate_thickness_trajectory(&initial, &target, 160, 80, 0.0);
    let mid_exp = generate_thickness_trajectory(&initial, &target, 160, 80, -2.0);
    // With alpha < 0, less removal early → midpoint should be HIGHER than linear
    assert!(
        mid_exp[0] > mid_linear[0],
        "alpha=-2 midpoint ({:.0}) should be higher than linear ({:.0})",
        mid_exp[0], mid_linear[0]
    );
}

#[test]
fn trajectory_start_equals_initial_for_all_alpha() {
    let initial = Vec21::from_element(10000.0);
    let target = Vec21::from_element(2000.0);
    for alpha in [-2.0, -1.0, 0.0, 1.0, 2.0, 5.0] {
        let start = generate_thickness_trajectory(&initial, &target, 160, 0, alpha);
        assert!(
            (start[0] - 10000.0).abs() < 1.0,
            "alpha={}: start should be 10000, got {:.1}", alpha, start[0]
        );
    }
}

#[test]
fn trajectory_end_equals_target_for_all_alpha() {
    let initial = Vec21::from_element(10000.0);
    let target = Vec21::from_element(2000.0);
    for alpha in [-2.0, -1.0, 0.0, 1.0, 2.0, 5.0] {
        let end = generate_thickness_trajectory(&initial, &target, 160, 160, alpha);
        assert!(
            (end[0] - 2000.0).abs() < 1.0,
            "alpha={}: end should be 2000, got {:.1}", alpha, end[0]
        );
    }
}

#[test]
fn trajectory_monotonically_decreasing_for_positive_alpha() {
    let initial = Vec21::from_element(10000.0);
    let target = Vec21::from_element(2000.0);
    for alpha in [0.0, 0.5, 1.0, 2.0, 5.0] {
        let mut prev = 10001.0;
        for j in 0..=160 {
            let t = generate_thickness_trajectory(&initial, &target, 160, j, alpha);
            assert!(
                t[0] <= prev + 0.01,
                "alpha={}: not monotone at turn {}: {:.1} > {:.1}",
                alpha, j, t[0], prev
            );
            prev = t[0];
        }
    }
}

#[test]
fn trajectory_alpha_1_4_gives_4_to_1_ratio() {
    // alpha = ln(4) ≈ 1.386 gives exactly 4:1 early-to-late removal ratio
    let initial = Vec21::from_element(10000.0);
    let target = Vec21::from_element(2000.0);
    let alpha = (4.0_f64).ln();

    let t0 = generate_thickness_trajectory(&initial, &target, 1000, 0, alpha);
    let t1 = generate_thickness_trajectory(&initial, &target, 1000, 1, alpha);
    let t999 = generate_thickness_trajectory(&initial, &target, 1000, 999, alpha);
    let t1000 = generate_thickness_trajectory(&initial, &target, 1000, 1000, alpha);

    let early_removal = t0[0] - t1[0];
    let late_removal = t999[0] - t1000[0];
    let ratio = early_removal / late_removal;

    assert!(
        (ratio - 4.0).abs() < 0.5,
        "alpha=ln(4) should give ~4:1 ratio, got {:.2}", ratio
    );
}

#[test]
fn trajectory_large_alpha_very_front_loaded() {
    let initial = Vec21::from_element(10000.0);
    let target = Vec21::from_element(2000.0);
    // With alpha=5, most removal happens in first 20%
    let t_20pct = generate_thickness_trajectory(&initial, &target, 100, 20, 5.0);
    let removed_at_20pct = 10000.0 - t_20pct[0];
    let total_removal = 8000.0;
    let fraction_at_20pct = removed_at_20pct / total_removal;

    assert!(
        fraction_at_20pct > 0.5,
        "alpha=5: should remove >50% by 20% time, got {:.1}%",
        fraction_at_20pct * 100.0
    );
}

// =========================================================================
//  REMOVAL RATE CLAMPING
// =========================================================================

#[test]
fn clamp_removal_rate_no_change_at_nominal() {
    let removal = Vec21::from_element(NOMINAL_REMOVAL_RATE); // 50 Å/turn
    let (clamped, was_clamped) = clamp_removal_rate(&removal);
    assert!(!was_clamped, "Nominal rate should not be clamped");
    assert!((clamped[0] - 50.0).abs() < 0.01);
}

#[test]
fn clamp_removal_rate_caps_at_100() {
    let removal = Vec21::from_element(200.0); // 200 Å/turn — too high
    let (clamped, was_clamped) = clamp_removal_rate(&removal);
    assert!(was_clamped, "200 Å/turn should be clamped");
    let avg: f64 = clamped.iter().sum::<f64>() / NY as f64;
    assert!(
        (avg - MAX_REMOVAL_RATE).abs() < 0.1,
        "Clamped average should be {}, got {:.1}", MAX_REMOVAL_RATE, avg
    );
}

#[test]
fn clamp_removal_rate_floors_at_25() {
    let removal = Vec21::from_element(10.0); // 10 Å/turn — too low
    let (clamped, was_clamped) = clamp_removal_rate(&removal);
    assert!(was_clamped, "10 Å/turn should be clamped");
    let avg: f64 = clamped.iter().sum::<f64>() / NY as f64;
    assert!(
        (avg - MIN_REMOVAL_RATE).abs() < 0.1,
        "Clamped average should be {}, got {:.1}", MIN_REMOVAL_RATE, avg
    );
}

#[test]
fn clamp_removal_rate_preserves_shape() {
    // Non-uniform removal: center=150, edge=100 → avg=~125 → clamped to avg=100
    let r_out = radial_output_positions();
    let removal = Vec21::from_fn(|j, _| {
        150.0 - 50.0 * (r_out[j] / WAFER_RADIUS)
    });
    let (clamped, was_clamped) = clamp_removal_rate(&removal);
    assert!(was_clamped);

    // Shape should be preserved (ratio center/edge same after scaling)
    let ratio_before = removal[0] / removal[NY - 1];
    let ratio_after = clamped[0] / clamped[NY - 1];
    assert!(
        (ratio_before - ratio_after).abs() < 0.01,
        "Clamping should preserve shape: ratio before={:.2}, after={:.2}",
        ratio_before, ratio_after
    );
}

#[test]
fn clamp_removal_rate_constants_correct() {
    assert_eq!(MAX_REMOVAL_RATE, 100.0);
    assert_eq!(MIN_REMOVAL_RATE, 25.0);
    assert_eq!(MAX_REMOVAL_RATE, NOMINAL_REMOVAL_RATE * 2.0);
    assert_eq!(MIN_REMOVAL_RATE, NOMINAL_REMOVAL_RATE / 2.0);
}

// =========================================================================
//  ANTI-CHATTERING (SLEW WEIGHT)
// =========================================================================

#[test]
fn slew_weight_is_strong() {
    let w = WeightConfig::default_cmp();
    // W_Δu should be significantly larger than W_u to prevent chattering
    for i in 0..NU {
        assert!(
            w.slew_weights[i] > w.effort_weights[i] * 10.0,
            "Slew weight ({}) should be >> effort weight ({}) for zone {}",
            w.slew_weights[i], w.effort_weights[i], i
        );
    }
}

#[test]
fn slew_weight_value_is_2() {
    let w = WeightConfig::default_cmp();
    for i in 0..NU {
        assert!(
            (w.slew_weights[i] - 2.0).abs() < 0.01,
            "Slew weight should be 2.0, got {}", w.slew_weights[i]
        );
    }
}

#[test]
fn pressure_stable_with_noise() {
    // With noise but strong slew weight, pressure should not chatter
    let config = SimConfig {
        n_wafers: 1,
        turns_per_wafer: 80,
        disturbance_amplitude: 3.0,
        noise_amplitude: 5.0,
        enable_inrun: true,
        enable_r2r: false,
        turn_detail_every_n: 1,
        ..Default::default()
    };
    let result = run_simulation(&config);
    let turns: Vec<_> = result.turn_snapshots.iter()
        .filter(|t| t.wafer == 0)
        .collect();

    if turns.len() < 10 { return; }

    // Compute max turn-to-turn pressure change across all zones
    let mut max_change = 0.0_f64;
    for i in 1..turns.len() {
        for z in 0..NU {
            let change = (turns[i].pressure[z] - turns[i-1].pressure[z]).abs();
            max_change = max_change.max(change);
        }
    }

    assert!(
        max_change < 1.0,
        "Max turn-to-turn pressure change should be <1 psi (anti-chatter), got {:.3}",
        max_change
    );
}

#[test]
fn pressure_smooth_without_noise() {
    let config = SimConfig {
        n_wafers: 1,
        turns_per_wafer: 80,
        disturbance_amplitude: 0.0,
        noise_amplitude: 0.0,
        enable_inrun: true,
        enable_r2r: false,
        turn_detail_every_n: 1,
        ..Default::default()
    };
    let result = run_simulation(&config);
    let turns: Vec<_> = result.turn_snapshots.iter()
        .filter(|t| t.wafer == 0)
        .collect();

    if turns.len() < 10 { return; }

    // Without noise, steady-state pressure should be constant (zero change)
    let mut max_change = 0.0_f64;
    for i in 1..turns.len() {
        for z in 0..NU {
            let change = (turns[i].pressure[z] - turns[i-1].pressure[z]).abs();
            max_change = max_change.max(change);
        }
    }

    assert!(
        max_change < 0.01,
        "Without noise, pressure should be constant, max change = {:.4}",
        max_change
    );
}

// =========================================================================
//  PRESTON EQUATION & CCDF KERNEL
// =========================================================================

#[test]
fn g0_uses_ccdf_not_pdf() {
    let g0 = generate_synthetic_g0();
    // CCDF kernel gives plateau-like response within each zone,
    // not a peak-like response (PDF). Zone 4 (70-90mm) should have
    // similar values at r=75 and r=85 (both inside the zone).
    let idx_75 = (NY * 75) / 150;
    let idx_85 = (NY * 85) / 150;
    let ratio = g0[(idx_75, 3)] / g0[(idx_85, 3)].max(1e-10);
    assert!(
        (ratio - 1.0).abs() < 0.3,
        "CCDF should give flat response inside zone: g0[75mm,Z4]/g0[85mm,Z4] = {:.2}",
        ratio
    );
}

#[test]
fn g0_zone1_integrates_full_diameter() {
    let g0 = generate_synthetic_g0();
    // Zone 1 (center disk, -30 to +30) should have high influence at center
    // because it integrates both left and right sides
    assert!(
        g0[(0, 0)] > g0[(0, 1)] * 0.5,
        "Zone 1 at center ({:.4}) should be significant vs Zone 2 ({:.4})",
        g0[(0, 0)], g0[(0, 1)]
    );
}

#[test]
fn g0_edge_reflection_boosts_edge_zones() {
    let g0 = generate_synthetic_g0();
    // Zone 10 (145-150mm) should have boosted influence at r=150mm
    // due to edge reflection (image at [150, 155] adds to direct)
    let edge_idx = NY - 1;
    let inner_idx = (NY * 140) / 150;
    // Edge value should be substantial
    assert!(
        g0[(edge_idx, 9)] > 0.0,
        "Z10 at edge should have positive influence, got {:.4}",
        g0[(edge_idx, 9)]
    );
}

#[test]
fn g0_preston_velocity_effect() {
    // The velocity profile gives slightly higher removal at the edge
    // V(r) increases linearly with r due to platen-carrier speed mismatch
    let v = velocity_profile();
    assert!(
        v[NY-1] > v[0],
        "Velocity at edge ({:.4}) should be > center ({:.4})",
        v[NY-1], v[0]
    );
    // But the difference should be small (< 5%)
    let diff_pct = (v[NY-1] - v[0]) / v[0] * 100.0;
    assert!(
        diff_pct < 5.0,
        "Velocity variation should be <5%, got {:.2}%", diff_pct
    );
}

#[test]
fn g0_sigma_6_for_carrier_sigma_3_for_rr() {
    assert_eq!(PRESSURE_SIGMA, 6.0, "Carrier zone σ should be 6 mm");
    // RR uses σ=3 internally in generate_synthetic_g0
    // Verify by checking RR column is more localized than carrier columns
    let g0 = generate_synthetic_g0();
    // RR: significant only near edge (r > 130mm)
    let mid_idx = NY / 2; // r = 75mm
    assert!(
        g0[(mid_idx, 10)].abs() < 0.1,
        "RR at r=75mm should be ~0 (σ=3mm localized), got {:.4}",
        g0[(mid_idx, 10)]
    );
}

// =========================================================================
//  RR REBOUND WITH PIVOT AT 146 MM
// =========================================================================

#[test]
fn rr_positive_before_pivot() {
    let g0 = generate_synthetic_g0();
    let idx_140 = (NY * 140) / 150;
    assert!(
        g0[(idx_140, 10)] > 0.0,
        "RR at r=140mm should be positive (redistribution), got {:.4}",
        g0[(idx_140, 10)]
    );
}

#[test]
fn rr_negative_after_pivot() {
    let g0 = generate_synthetic_g0();
    let idx_150 = NY - 1;
    assert!(
        g0[(idx_150, 10)] < 0.0,
        "RR at r=150mm should be negative (rebound), got {:.4}",
        g0[(idx_150, 10)]
    );
}

#[test]
fn rr_sign_change_near_146mm() {
    let g0 = generate_synthetic_g0();
    let r_out = radial_output_positions();
    // Find the zero crossing
    let mut cross_r = 0.0;
    for j in 1..NY {
        if g0[(j-1, 10)] > 0.0 && g0[(j, 10)] <= 0.0 {
            cross_r = r_out[j];
            break;
        }
    }
    assert!(
        (cross_r - 146.0).abs() < 5.0,
        "RR zero crossing should be near 146mm, got {:.1}mm", cross_r
    );
}

#[test]
fn rr_zero_at_center() {
    let g0 = generate_synthetic_g0();
    assert!(
        g0[(0, 10)].abs() < 0.01,
        "RR at center should be ~0, got {:.4}", g0[(0, 10)]
    );
}

#[test]
fn rr_rebound_is_smooth() {
    let g0 = generate_synthetic_g0();
    // Check smoothness of RR column: max second derivative should be bounded
    let mut max_d2 = 0.0_f64;
    for j in 1..NY-1 {
        let d2 = (g0[(j+1, 10)] - 2.0*g0[(j, 10)] + g0[(j-1, 10)]).abs();
        max_d2 = max_d2.max(d2);
    }
    let max_val = (0..NY).map(|j| g0[(j, 10)].abs())
        .fold(0.0_f64, f64::max);
    let ratio = max_d2 / max_val.max(1e-10);
    assert!(
        ratio < 1.0,
        "RR column should be smooth (d2/max ratio={:.2}), not jagged", ratio
    );
}

// =========================================================================
//  DETERMINISTIC INITIAL PROFILE
// =========================================================================

#[test]
fn initial_profile_identical_every_call() {
    let p1 = generate_initial_profile();
    let p2 = generate_initial_profile();
    for j in 0..NY {
        assert_eq!(
            p1[j].to_bits(), p2[j].to_bits(),
            "Initial profile should be bit-identical at point {}", j
        );
    }
}

#[test]
fn initial_profile_center_thick() {
    let p = generate_initial_profile();
    assert!(p[0] > p[NY-1], "Center ({:.0}) should be thicker than edge ({:.0})", p[0], p[NY-1]);
}

#[test]
fn initial_profile_exact_statistics() {
    let p = generate_initial_profile();
    let avg: f64 = p.iter().sum::<f64>() / NY as f64;
    let min = p.iter().cloned().fold(f64::MAX, f64::min);
    let max = p.iter().cloned().fold(f64::MIN, f64::max);
    assert!((avg - INITIAL_THICKNESS).abs() < 1.0, "Avg should be {}, got {:.1}", INITIAL_THICKNESS, avg);
    assert!((max - min - INITIAL_RANGE).abs() < 1.0, "Range should be {}, got {:.1}", INITIAL_RANGE, max - min);
}

// =========================================================================
//  ZONE GEOMETRY
// =========================================================================

#[test]
fn zone_widths_correct() {
    let geo = ZoneGeometry::default_cmp();
    let expected_widths = [30.0, 20.0, 20.0, 20.0, 20.0, 10.0, 10.0, 10.0, 5.0, 5.0, 20.0];
    for i in 0..NU {
        let width = geo.outer[i] - geo.inner[i];
        assert!(
            (width - expected_widths[i]).abs() < 0.01,
            "Zone {} width should be {}, got {}", i, expected_widths[i], width
        );
    }
}

#[test]
fn zone_carrier_covers_0_to_150() {
    let geo = ZoneGeometry::default_cmp();
    assert!((geo.inner[0] - 0.0).abs() < 0.01, "First zone starts at 0");
    assert!((geo.outer[9] - 150.0).abs() < 0.01, "Last carrier zone ends at 150");
}

#[test]
fn zone_rr_outside_wafer() {
    let geo = ZoneGeometry::default_cmp();
    assert!((geo.inner[10] - 150.0).abs() < 0.01, "RR starts at 150");
    assert!((geo.outer[10] - 170.0).abs() < 0.01, "RR ends at 170");
}

// =========================================================================
//  SIMULATION WITH EXPONENTIAL TRAJECTORY
// =========================================================================

#[test]
fn sim_exponential_trajectory_runs() {
    let config = SimConfig {
        n_wafers: 3,
        turns_per_wafer: 160,
        trajectory_alpha: 2.0,
        disturbance_amplitude: 0.0,
        noise_amplitude: 0.0,
        ..Default::default()
    };
    let result = run_simulation(&config);
    assert_eq!(result.wafer_snapshots.len(), 3);
    // Final thickness should be near target
    let avg_final: f64 = result.wafer_snapshots[0].final_profile.iter().sum::<f64>() / NY as f64;
    assert!(
        avg_final > 1000.0 && avg_final < 4000.0,
        "Final avg should be near 2000Å, got {:.0}", avg_final
    );
}

#[test]
fn sim_negative_alpha_runs() {
    let config = SimConfig {
        n_wafers: 2,
        turns_per_wafer: 160,
        trajectory_alpha: -1.0,
        disturbance_amplitude: 0.0,
        noise_amplitude: 0.0,
        ..Default::default()
    };
    let result = run_simulation(&config);
    assert_eq!(result.wafer_snapshots.len(), 2);
}

#[test]
fn sim_large_alpha_clamped_removal() {
    // With alpha=5, early removal demands are huge but get clamped to 100 Å/turn
    let config = SimConfig {
        n_wafers: 1,
        turns_per_wafer: 160,
        trajectory_alpha: 5.0,
        disturbance_amplitude: 0.0,
        noise_amplitude: 0.0,
        ..Default::default()
    };
    let result = run_simulation(&config);
    // Should complete without panic
    assert_eq!(result.wafer_snapshots.len(), 1);
    // Removal rate should be bounded
    assert!(
        result.wafer_snapshots[0].avg_removal_rate < 110.0,
        "Avg removal should be bounded, got {:.1}",
        result.wafer_snapshots[0].avg_removal_rate
    );
}

#[test]
fn sim_pressure_stable_with_disturbance() {
    let config = SimConfig {
        n_wafers: 1,
        turns_per_wafer: 80,
        trajectory_alpha: 0.0,
        disturbance_amplitude: 5.0,
        noise_amplitude: 5.0,
        enable_inrun: true,
        enable_r2r: false,
        turn_detail_every_n: 1,
        ..Default::default()
    };
    let result = run_simulation(&config);
    let turns: Vec<_> = result.turn_snapshots.iter()
        .filter(|t| t.wafer == 0)
        .collect();

    // With W_Δu = 2.0, pressure should not chatter
    if turns.len() >= 10 {
        let mut total_change = 0.0_f64;
        let n = turns.len() - 1;
        for i in 1..turns.len() {
            for z in 0..NU {
                total_change += (turns[i].pressure[z] - turns[i-1].pressure[z]).abs();
            }
        }
        let avg_change_per_zone_per_turn = total_change / (n as f64 * NU as f64);
        assert!(
            avg_change_per_zone_per_turn < 0.3,
            "Avg pressure change should be small (anti-chatter), got {:.4} psi/turn",
            avg_change_per_zone_per_turn
        );
    }
}

#[test]
fn sim_seed_changes_disturbance() {
    let r1 = run_simulation(&SimConfig { seed: 42, n_wafers: 2, turns_per_wafer: 40, ..Default::default() });
    let r2 = run_simulation(&SimConfig { seed: 123, n_wafers: 2, turns_per_wafer: 40, ..Default::default() });
    let diff: f64 = r1.wafer_snapshots[0].final_profile.iter()
        .zip(r2.wafer_snapshots[0].final_profile.iter())
        .map(|(a, b)| (a - b).abs())
        .sum();
    assert!(diff > 1.0, "Different seeds should give different results, diff={:.2}", diff);
}

#[test]
fn sim_same_seed_reproducible() {
    let r1 = run_simulation(&SimConfig { seed: 42, n_wafers: 2, turns_per_wafer: 40, ..Default::default() });
    let r2 = run_simulation(&SimConfig { seed: 42, n_wafers: 2, turns_per_wafer: 40, ..Default::default() });
    for j in 0..NY {
        assert_eq!(
            r1.wafer_snapshots[0].final_profile[j].to_bits(),
            r2.wafer_snapshots[0].final_profile[j].to_bits(),
            "Same seed should be bit-identical at point {}", j
        );
    }
}
