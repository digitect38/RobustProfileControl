/// Diagnostic test: check profile quality with zero disturbance and zero noise
use control_core::simulation::{run_simulation, SimConfig};
use control_core::synth_data::*;
use control_core::types::*;

#[test]
fn diagnose_zero_noise_profile() {
    let config = SimConfig {
        n_wafers: 3,
        turns_per_wafer: 160,
        disturbance_amplitude: 0.0,
        noise_amplitude: 0.0,
        enable_inrun: true,
        enable_r2r: true,
        turn_detail_every_n: 1, // record every wafer
        ..Default::default()
    };
    let result = run_simulation(&config);
    let ws = &result.wafer_snapshots[0];

    println!("=== WAFER 0 FINAL STATE ===");
    println!("RMS error:     {:.2} Å", ws.rms_error);
    println!("Profile range: {:.2} Å", ws.profile_range);
    println!("Edge error:    {:.2} Å", ws.edge_error);
    println!("Avg removal:   {:.2} Å/turn", ws.avg_removal_rate);

    // Check a few profile points
    let target = generate_target_profile();
    println!("\n=== PROFILE SAMPLES (target = {:.0} Å) ===", TARGET_THICKNESS);
    println!("{:>6} {:>10} {:>10} {:>10}", "r(mm)", "thickness", "target", "error");
    let r_out = radial_output_positions();
    for &idx in &[0, 10, 25, 50, 75, 90, 95, 98, 100] {
        if idx < NY {
            println!(
                "{:6.1} {:10.1} {:10.1} {:10.1}",
                r_out[idx],
                ws.final_profile[idx],
                ws.target_profile[idx],
                ws.final_error[idx],
            );
        }
    }

    // Check turn-by-turn pressure stability for wafer 0
    let turns: Vec<_> = result.turn_snapshots.iter()
        .filter(|t| t.wafer == 0)
        .collect();

    if turns.len() >= 5 {
        println!("\n=== PRESSURE AT SELECTED TURNS ===");
        println!("{:>5} {:>8} {:>8} {:>8} {:>8} {:>8}", "turn", "Z1", "Z5", "Z10", "RR", "RMS_err");
        for &turn_idx in &[0, 1, 2, 10, 40, 80, 120, 159.min(turns.len()-1)] {
            if turn_idx < turns.len() {
                let t = &turns[turn_idx];
                println!(
                    "{:5} {:8.3} {:8.3} {:8.3} {:8.3} {:8.2}",
                    t.turn,
                    t.pressure[0], t.pressure[4], t.pressure[9], t.pressure[10],
                    t.rms_error,
                );
            }
        }
    }

    // Check: is the "noise" actually step-function quantization?
    println!("\n=== ERROR PROFILE (should be smooth if no noise) ===");
    let mut max_jump = 0.0_f64;
    for j in 1..NY {
        let jump = (ws.final_error[j] - ws.final_error[j-1]).abs();
        if jump > max_jump {
            max_jump = jump;
        }
    }
    println!("Max adjacent-point error jump: {:.4} Å", max_jump);
    println!("(If this is large, it's zone-boundary quantization, not random noise)");

    // Print the full error profile
    println!("\n=== FULL ERROR PROFILE ===");
    for j in 0..NY {
        if j % 5 == 0 || j == NY - 1 {
            println!("r={:6.1} mm  error={:8.2} Å", r_out[j], ws.final_error[j]);
        }
    }
}
