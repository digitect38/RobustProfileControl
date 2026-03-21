use crate::types::*;

// ============================================================================
// Physical CMP constants (angstroms, psi, seconds)
// ============================================================================

/// Average initial wafer thickness (Å)
pub const INITIAL_THICKNESS: f64 = 10_000.0;
/// Incoming profile range (Å) — peak-to-valley non-uniformity
pub const INITIAL_RANGE: f64 = 1_000.0;
/// Target final thickness (Å)
pub const TARGET_THICKNESS: f64 = 2_000.0;
/// Target profile range (Å) — uniformity requirement
pub const TARGET_RANGE: f64 = 50.0;
/// Nominal removal rate (Å per turn at nominal pressure)
pub const NOMINAL_REMOVAL_RATE: f64 = 50.0;
/// Nominal pressure (psi) — midpoint of operating range
pub const NOMINAL_PRESSURE: f64 = 3.5;
/// Turn duration (seconds)
pub const TURN_DURATION: f64 = 1.0;

/// Removal needed: 10000 - 2000 = 8000 Å
pub const TOTAL_REMOVAL: f64 = INITIAL_THICKNESS - TARGET_THICKNESS;
/// Turns per wafer: 8000 / 50 = 160
pub const DEFAULT_TURNS_PER_WAFER: usize = (TOTAL_REMOVAL / NOMINAL_REMOVAL_RATE) as usize;

// ============================================================================
// Preston equation: RR = k_p × P(r) × V(r)
// ============================================================================

/// Platen rotation speed (rpm)
pub const PLATEN_RPM: f64 = 80.0;
/// Carrier rotation speed (rpm)
pub const CARRIER_RPM: f64 = 78.0;
/// Center-to-center distance between platen axis and carrier axis (mm)
pub const CENTER_OFFSET: f64 = 175.0;

/// Compute the relative pad–wafer velocity at radius r (mm/s).
///
/// For a carrier rotating at ω_c and platen at ω_p with center offset r_cc:
///   V(r) = ω_p × r_cc + (ω_p − ω_c) × r
///
/// This gives nearly uniform velocity when ω_p ≈ ω_c, with a slight
/// radial gradient from the speed mismatch.
pub fn pad_velocity(r: f64) -> f64 {
    let omega_p = PLATEN_RPM * 2.0 * std::f64::consts::PI / 60.0; // rad/s
    let omega_c = CARRIER_RPM * 2.0 * std::f64::consts::PI / 60.0;
    let v = omega_p * CENTER_OFFSET + (omega_p - omega_c) * r;
    v.abs()
}

/// Compute the normalized velocity profile: V(r) / V_avg over the wafer.
/// This factor scales each row of G₀ so that removal follows Preston's law.
pub fn velocity_profile() -> [f64; NY] {
    let r_out = radial_output_positions();
    let mut v = [0.0; NY];
    let mut sum = 0.0;
    for j in 0..NY {
        v[j] = pad_velocity(r_out[j]);
        sum += v[j];
    }
    let avg = sum / NY as f64;
    for j in 0..NY {
        v[j] /= avg; // normalize so average = 1
    }
    v
}

// ============================================================================
// Gaussian CDF (for CCDF kernel)
// ============================================================================

/// Standard normal CDF Φ(x) using the Abramowitz & Stegun approximation.
/// Accurate to ~1.5e-7.
fn normal_cdf(x: f64) -> f64 {
    if x < -8.0 {
        return 0.0;
    }
    if x > 8.0 {
        return 1.0;
    }
    let sign = if x >= 0.0 { 1.0 } else { -1.0 };
    let x = x.abs();
    let t = 1.0 / (1.0 + 0.2316419 * x);
    let d = 0.3989422804014327; // 1/√(2π)
    let p = d * (-x * x / 2.0).exp();
    let c = ((((1.330274429 * t - 1.821255978) * t + 1.781477937) * t
        - 0.356563782) * t + 0.319381530) * t;
    0.5 + sign * (0.5 - p * c)
}

// ============================================================================
// Plant model (G₀ in Å / (psi · turn))
// ============================================================================

/// Generate a physically scaled G₀ ∈ R^{101×11} for a CMP carrier head.
///
/// Based on the **Preston equation**: RR(r) = k_p × P(r) × V(r)
///   - k_p: Preston coefficient (material/slurry dependent)
///   - P(r): local contact pressure from carrier zones (Gaussian spreading)
///   - V(r): relative pad–wafer velocity (from platen/carrier rotation)
///
/// The carrier is a circular membrane (R = 150 mm) with concentric annular
/// pressure zones. Zone 1 is a disk (0–30 mm). Zones 2–10 are annuli.
/// The retaining ring is outside the wafer edge (150–170 mm).
///
/// Pressure model (2D axisymmetric, σ = 6 mm Gaussian spreading):
///   P_influence(r, zone_i) = (1/(2πσ²)) ∫∫_zone exp(-d²/(2σ²)) dA'
///   with edge reflection at R = 150 mm (method of images).
///
/// The velocity factor V(r)/V_avg introduces a slight radial gradient
/// from the platen–carrier speed mismatch (Preston effect).
pub fn generate_synthetic_g0() -> Mat21x11 {
    let r_out = radial_output_positions();
    let geo = ZoneGeometry::default_cmp();
    let sigma = PRESSURE_SIGMA;

    let mut g_shape = Mat21x11::zeros();

    for j in 0..NY {
        let r = r_out[j];

        for i in 0..NU {
            if i == 10 {
                // ---- Retaining Ring: CCDF kernel (σ=3) with rebound pivot ----
                //
                // The RR presses the pad down OUTSIDE the wafer (150–170 mm).
                // Due to pad elasticity there is a pivot/fulcrum at ~146 mm:
                //   r < 146 mm → positive redistribution (pad pushed toward wafer)
                //   r > 146 mm → negative rebound (pad lifts off the wafer edge)
                //
                // Model: compute the CCDF pressure envelope from the RR zone,
                // then apply a sign function centered at the pivot point.
                // The magnitude follows the CCDF shape (σ_rr = 3 mm).
                let sigma_rr = 3.0;
                let pivot = 146.0; // mm — rebound pivot point

                let r_inner = geo.inner[10]; // 150
                let r_outer = geo.outer[10]; // 170

                // CCDF envelope: how much RR pressure reaches this point
                let right = normal_cdf((r_outer - r) / sigma_rr)
                          - normal_cdf((r_inner - r) / sigma_rr);

                let left = normal_cdf((-r_inner - r) / sigma_rr)
                         - normal_cdf((-r_outer - r) / sigma_rr);

                // Edge reflection image at [130, 150]
                let img_inner = 2.0 * WAFER_RADIUS - r_outer; // 130
                let img_outer = 2.0 * WAFER_RADIUS - r_inner; // 150
                let reflect = normal_cdf((img_outer - r) / sigma_rr)
                            - normal_cdf((img_inner - r) / sigma_rr);

                let envelope = right + left + reflect;

                // Apply rebound sign: smooth transition around pivot
                // sign_factor = +1 for r << pivot, -1 for r >> pivot
                // Using tanh for a smooth crossover (~2 mm transition width)
                let sign_factor = -((r - pivot) / 2.0).tanh();

                g_shape[(j, 10)] = envelope * sign_factor;
            } else {
                // ---- Carrier zones 0–9: CCDF (complementary CDF) kernel ----
                //
                // The carrier is circular. Along the measurement diameter,
                // each annular zone [r_inner, r_outer] appears as two segments:
                //   Right side: [+r_inner, +r_outer]
                //   Left  side: [-r_outer, -r_inner]
                //
                // For Zone 1 (center disk, r_inner=0): spans [-30, +30] mm.
                //
                // The Gaussian CCDF kernel at point r:
                //   g(r) = right + left + edge_reflection
                //
                //   right = Φ((r_outer - r)/σ) - Φ((r_inner - r)/σ)
                //   left  = Φ((-r_inner - r)/σ) - Φ((-r_outer - r)/σ)
                //
                // Edge reflection (image at 2R) applies to the right side only
                // (left-side image at 2R+r is >300mm, negligible).
                let r_inner = geo.inner[i];
                let r_outer = geo.outer[i];

                // Right side: [+r_inner, +r_outer]
                let right = normal_cdf((r_outer - r) / sigma)
                          - normal_cdf((r_inner - r) / sigma);

                // Left side: [-r_outer, -r_inner]
                let left = normal_cdf((-r_inner - r) / sigma)
                         - normal_cdf((-r_outer - r) / sigma);

                // Edge reflection of right side: image at [2R-r_outer, 2R-r_inner]
                let img_inner = 2.0 * WAFER_RADIUS - r_outer;
                let img_outer = 2.0 * WAFER_RADIUS - r_inner;
                let reflect = normal_cdf((img_outer - r) / sigma)
                            - normal_cdf((img_inner - r) / sigma);

                g_shape[(j, i)] = right + left + reflect;
            }
        }
    }

    // Apply Preston velocity factor: G₀[j][i] *= V(r_j) / V_avg
    // This encodes the Preston equation RR = k_p × P × V into the plant matrix.
    let v_profile = velocity_profile();
    for j in 0..NY {
        for i in 0..NU {
            g_shape[(j, i)] *= v_profile[j];
        }
    }

    // Scale so that at nominal pressure, average removal ≈ NOMINAL_REMOVAL_RATE
    let u_nom = Vec11::from_element(NOMINAL_PRESSURE);
    let removal_unscaled = g_shape * u_nom;
    let avg_removal: f64 = removal_unscaled.iter().sum::<f64>() / NY as f64;

    let gain = NOMINAL_REMOVAL_RATE / avg_removal;

    g_shape * gain
}

/// Generate the incoming wafer thickness profile.
///
/// Deterministic center-thick pattern from the upstream deposition process:
/// thicker at center, thinner at edge, smooth parabolic variation.
/// Average: INITIAL_THICKNESS (10,000 Å), range: INITIAL_RANGE (1,000 Å).
pub fn generate_initial_profile() -> Vec21 {
    let r_out = radial_output_positions();
    let mut profile = Vec21::zeros();

    // Build parabolic shape first, then shift to exact average and range
    for j in 0..NY {
        let r_norm = r_out[j] / WAFER_RADIUS;
        profile[j] = 1.0 - r_norm * r_norm; // 1 at center, 0 at edge
    }

    // Normalize to exact desired average and range
    let min_val = profile.iter().cloned().fold(f64::MAX, f64::min);
    let max_val = profile.iter().cloned().fold(f64::MIN, f64::max);
    let raw_range = max_val - min_val;
    let raw_avg: f64 = profile.iter().sum::<f64>() / NY as f64;

    for j in 0..NY {
        // Scale to desired range, center on desired average
        let normalized = (profile[j] - raw_avg) / raw_range; // in [-0.5, +0.5] approx
        profile[j] = INITIAL_THICKNESS + normalized * INITIAL_RANGE;
    }
    profile
}

/// Generate the target thickness profile.
///
/// Flat target at TARGET_THICKNESS (2000 Å) with no intentional variation.
pub fn generate_target_profile() -> Vec21 {
    Vec21::from_element(TARGET_THICKNESS)
}

/// Generate a target thickness trajectory for InRun control.
///
/// Linear ramp from initial_profile to target over n_turns.
/// r_{k,j} = initial - (j/n_turns) * (initial - target)
pub fn generate_thickness_trajectory(
    initial_profile: &Vec21,
    target: &Vec21,
    n_turns: usize,
    turn: usize,
) -> Vec21 {
    let t = (turn as f64) / (n_turns as f64).max(1.0);
    let t = t.min(1.0);
    // Linear interpolation: initial * (1-t) + target * t
    initial_profile * (1.0 - t) + target * t
}

// ============================================================================
// Wear perturbations (in Å/(psi·turn) units)
// ============================================================================

/// Pad-wear perturbation: wear reduces removal efficiency at the center,
/// increases relative edge removal. Scaled by the physical G₀.
pub fn generate_pad_wear_perturbation(wear_fraction: f64) -> Mat21x11 {
    let r_out = radial_output_positions();
    let g0 = generate_synthetic_g0();
    let mut delta = Mat21x11::zeros();

    for j in 0..NY {
        let radial_factor = (r_out[j] / WAFER_RADIUS - 0.5) * 2.0;
        for i in 0..NU {
            delta[(j, i)] = wear_fraction * radial_factor * g0[(j, i)] * 0.15;
        }
    }
    delta
}

/// Retaining-ring wear perturbation: reduces edge effectiveness.
pub fn generate_rr_wear_perturbation(wear_fraction: f64) -> Mat21x11 {
    let r_out = radial_output_positions();
    let g0 = generate_synthetic_g0();
    let mut delta = Mat21x11::zeros();
    let sigma = 12.0;
    let center = 155.0; // RR center influence on wafer edge

    for j in 0..NY {
        let dr = r_out[j] - center;
        let edge_weight = (-dr * dr / (2.0 * sigma * sigma)).exp();
        // Retaining ring column loses effectiveness
        delta[(j, 10)] = -wear_fraction * 0.3 * edge_weight * g0[(j, 10)].max(0.01);
        // Outer carrier zones also affected slightly
        // Outer carrier zones affected for r > 120 mm
        let edge_start = (NY * 120) / 150;
        if j >= edge_start {
            for i in 7..10 {
                delta[(j, i)] = -wear_fraction * 0.05 * edge_weight * g0[(j, i)].max(0.01);
            }
        }
    }
    delta
}

// ============================================================================
// Actuator bounds
// ============================================================================

/// Default actuator bounds for a typical CMP head (psi)
pub fn default_actuator_bounds() -> ActuatorBounds {
    let mut u_min = [0.5; NU];
    let mut u_max = [7.0; NU];
    let mut du_min = [-0.5; NU]; // 0.5 psi/sec slew
    let mut du_max = [0.5; NU];

    // Retaining ring has tighter bounds
    u_min[10] = 0.3;
    u_max[10] = 5.0;
    du_min[10] = -0.3;
    du_max[10] = 0.3;

    ActuatorBounds {
        u_min,
        u_max,
        du_min,
        du_max,
    }
}

// ============================================================================
// Legacy API (used by tests that don't care about physical units)
// ============================================================================

/// Flat target profile at given value
pub fn flat_target_profile(value: f64) -> Vec21 {
    Vec21::from_element(value)
}

/// Target with controlled edge roll-off
pub fn edge_rolloff_target(base: f64, edge_amount: f64) -> Vec21 {
    let r_out = radial_output_positions();
    let mut target = Vec21::zeros();
    for j in 0..NY {
        let normalized_r = r_out[j] / WAFER_RADIUS;
        let edge_factor = if normalized_r > 0.85 {
            let t = (normalized_r - 0.85) / 0.15;
            1.0 - edge_amount * (1.0 - (std::f64::consts::PI * t / 2.0).cos())
        } else {
            1.0
        };
        target[j] = base * edge_factor;
    }
    target
}

// ============================================================================
// Random number generator
// ============================================================================

/// Simple pseudo-random number generator (xorshift64) for reproducible disturbances.
pub struct SimpleRng {
    state: u64,
}

impl SimpleRng {
    pub fn new(seed: u64) -> Self {
        Self {
            state: if seed == 0 { 1 } else { seed },
        }
    }

    fn next_u64(&mut self) -> u64 {
        let mut x = self.state;
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        self.state = x;
        x
    }

    /// Uniform in [0, 1)
    pub fn next_f64(&mut self) -> f64 {
        (self.next_u64() >> 11) as f64 / (1u64 << 53) as f64
    }

    /// Approximate normal(0, 1) using Box-Muller
    pub fn next_normal(&mut self) -> f64 {
        let u1 = self.next_f64().max(1e-15);
        let u2 = self.next_f64();
        (-2.0 * u1.ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos()
    }

    /// Random Vec21 with given standard deviation
    pub fn random_vec21(&mut self, std_dev: f64) -> Vec21 {
        let mut v = Vec21::zeros();
        for j in 0..NY {
            v[j] = self.next_normal() * std_dev;
        }
        v
    }
}

// ============================================================================
// Disturbance generation (in Å/turn units)
// ============================================================================

/// Generate a disturbance sequence for simulation.
///
/// Disturbances represent unmodeled removal-rate variations in Å/turn:
/// slurry flow, pad temperature, friction transients, etc.
pub fn generate_disturbance_sequence(
    n_wafers: usize,
    turns_per_wafer: usize,
    amplitude: f64,
    seed: u64,
) -> Vec<Vec<Vec21>> {
    let mut rng = SimpleRng::new(seed);
    let r_out = radial_output_positions();
    let mut all = Vec::with_capacity(n_wafers);

    for _k in 0..n_wafers {
        let mut wafer_dist = Vec::with_capacity(turns_per_wafer);
        // Slowly varying wafer-level disturbance
        let wafer_bias = rng.random_vec21(amplitude * 0.3);

        for _j in 0..turns_per_wafer {
            let mut d = wafer_bias + rng.random_vec21(amplitude * 0.5);
            // Spatially correlated edge disturbance burst
            let edge_burst = rng.next_normal() * amplitude * 0.4;
            let edge_start = (NY * 120) / 150; // r > 120 mm
            for jj in edge_start..NY {
                let r_norm = r_out[jj] / WAFER_RADIUS;
                d[jj] += edge_burst * (r_norm - 0.8) / 0.2;
            }
            wafer_dist.push(d);
        }
        all.push(wafer_dist);
    }
    all
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_g0_shape_and_rank() {
        let g0 = generate_synthetic_g0();
        assert_eq!(g0.nrows(), NY);
        assert_eq!(g0.ncols(), NU);
        // Carrier zones (0..10) should be non-negative
        for j in 0..NY {
            for i in 0..10 {
                assert!(g0[(j, i)] >= 0.0, "G0[{},{}] = {} < 0", j, i, g0[(j, i)]);
            }
        }
        // Retaining ring (col 10): CCDF σ=3 with rebound pivot at 146 mm.
        // Positive for r < 146 mm (redistribution), negative for r > 146 mm (rebound).
        let pivot_idx = (NY * 146) / 150; // index for r ≈ 146 mm
        let edge_idx = NY - 1;            // r = 150 mm

        // Should be positive just inside pivot (~140 mm)
        let inside_idx = (NY * 140) / 150;
        assert!(
            g0[(inside_idx, 10)] > 0.0,
            "RR at r=140mm should be positive (redistribution), got {:.4}",
            g0[(inside_idx, 10)]
        );
        // Should be negative at wafer edge (rebound)
        assert!(
            g0[(edge_idx, 10)] < 0.0,
            "RR at r=150mm should be negative (rebound), got {:.4}",
            g0[(edge_idx, 10)]
        );
        // Should be near zero far from edge
        assert!(
            g0[(0, 10)].abs() < 0.01,
            "RR at wafer center should be ~0, got {:.4}",
            g0[(0, 10)]
        );
        // Check full rank via SVD
        let svd = nalgebra::SVD::new(g0.clone_owned(), false, false);
        let svals = svd.singular_values;
        assert!(svals[NU - 1] > 1e-10, "G0 should be full rank, smallest sv = {}", svals[NU - 1]);
    }

    #[test]
    fn test_g0_removal_rate_at_nominal() {
        let g0 = generate_synthetic_g0();
        let u_nom = Vec11::from_element(NOMINAL_PRESSURE);
        let removal = g0 * u_nom;

        // Average removal should be close to NOMINAL_REMOVAL_RATE (50 Å/turn)
        let avg: f64 = removal.iter().sum::<f64>() / NY as f64;
        assert!(
            (avg - NOMINAL_REMOVAL_RATE).abs() < 5.0,
            "Average removal at nominal pressure = {:.1} Å/turn, expected ~{:.0}",
            avg,
            NOMINAL_REMOVAL_RATE
        );
    }

    #[test]
    fn test_initial_profile() {
        let profile = generate_initial_profile();

        let avg: f64 = profile.iter().sum::<f64>() / NY as f64;
        let min = profile.iter().cloned().fold(f64::MAX, f64::min);
        let max = profile.iter().cloned().fold(f64::MIN, f64::max);
        let range = max - min;

        // Deterministic: average exactly INITIAL_THICKNESS, range exactly INITIAL_RANGE
        assert!(
            (avg - INITIAL_THICKNESS).abs() < 1.0,
            "Average should be {}, got {:.1}",
            INITIAL_THICKNESS, avg
        );
        assert!(
            (range - INITIAL_RANGE).abs() < 1.0,
            "Range should be {}, got {:.1}",
            INITIAL_RANGE, range
        );
        // Center should be thicker than edge
        assert!(profile[0] > profile[NY - 1]);
    }

    #[test]
    fn test_thickness_trajectory() {
        let initial = Vec21::from_element(10000.0);
        let target = Vec21::from_element(2000.0);

        let mid = generate_thickness_trajectory(&initial, &target, 160, 80);
        // At halfway, should be ~6000
        assert!(
            (mid[0] - 6000.0).abs() < 1.0,
            "Midpoint trajectory should be ~6000, got {:.0}",
            mid[0]
        );

        let end = generate_thickness_trajectory(&initial, &target, 160, 160);
        assert!(
            (end[0] - 2000.0).abs() < 1.0,
            "End trajectory should be ~2000, got {:.0}",
            end[0]
        );
    }

    #[test]
    fn test_default_bounds_feasible() {
        let b = default_actuator_bounds();
        for i in 0..NU {
            assert!(b.u_min[i] < b.u_max[i]);
            assert!(b.du_min[i] < b.du_max[i]);
            assert!(b.u_min[i] >= 0.0);
        }
    }

    #[test]
    fn test_rng_deterministic() {
        let mut r1 = SimpleRng::new(42);
        let mut r2 = SimpleRng::new(42);
        for _ in 0..100 {
            assert_eq!(r1.next_f64().to_bits(), r2.next_f64().to_bits());
        }
    }
}
