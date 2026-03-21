use nalgebra::DVector;

use crate::types::*;

/// Anti-windup via back-calculation.
///
/// When actuators saturate, the integral state is corrected:
///   x_I_{k+1} = x_I_k + E_I * e_k + K_aw * (u_sat - u_cmd)
///
/// K_aw is typically a diagonal gain that "unwinds" the integrator
/// proportionally to the saturation error.
#[derive(Clone, Debug)]
pub struct AntiWindup {
    /// Anti-windup gain (NU-dimensional diagonal, stored as vector)
    pub k_aw: [f64; NU],
}

impl AntiWindup {
    /// Create with specified gains. Typical range: 0.5 to 1.0 per zone.
    pub fn new(k_aw: [f64; NU]) -> Self {
        Self { k_aw }
    }

    /// Default anti-windup gains
    pub fn default_cmp() -> Self {
        Self {
            k_aw: [0.8; NU],
        }
    }

    /// Correct integral state after saturation.
    ///
    /// integral_new = integral + e_i_gain * error + K_aw * (u_sat - u_cmd)
    ///
    /// `integral` is the current integral state (can be any dimension),
    /// but the back-calculation term is computed in actuator space (NU)
    /// and projected into the integral dimension via `projection`.
    pub fn correct_integral_reduced(
        &self,
        integral: &DVector<f64>,
        error: &DVector<f64>,
        e_i_gain: f64,
        u_saturated: &Vec11,
        u_commanded: &Vec11,
        projection: &nalgebra::DMatrix<f64>,
    ) -> DVector<f64> {
        let n = integral.len();
        let mut new_integral = integral + error * e_i_gain;

        // Back-calculation: Δ_aw = K_aw * (u_sat - u_cmd)
        let mut aw_correction = DVector::zeros(NU);
        for i in 0..NU {
            aw_correction[i] = self.k_aw[i] * (u_saturated[i] - u_commanded[i]);
        }

        // Project into integral space if dimensions differ
        if projection.nrows() == n && projection.ncols() == NU {
            new_integral += projection * aw_correction;
        } else if n == NU {
            new_integral += aw_correction;
        }

        new_integral
    }

    /// Simple direct anti-windup for full-dimensional (21-dim) integral.
    /// Uses conditional integration: freeze integration direction if it deepens saturation.
    pub fn conditional_integrate(
        &self,
        integral: &Vec21,
        error: &Vec21,
        e_i_gain: f64,
        u_saturated: &Vec11,
        u_commanded: &Vec11,
    ) -> Vec21 {
        let sat_diff = u_saturated - u_commanded;
        let sat_magnitude = sat_diff.norm();

        if sat_magnitude < 1e-10 {
            // No saturation: normal integration
            integral + error * e_i_gain
        } else {
            // Reduced integration rate proportional to saturation severity
            let reduction = (1.0 - sat_magnitude * 0.5).max(0.1);
            integral + error * (e_i_gain * reduction)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_no_saturation_no_correction() {
        let aw = AntiWindup::default_cmp();
        let u = Vec11::from_element(3.0);
        let integral = Vec21::from_element(1.0);
        let error = Vec21::from_element(0.1);

        // No saturation: u_sat == u_cmd
        let new_int = aw.conditional_integrate(&integral, &error, 1.0, &u, &u);
        // Should be integral + error
        for j in 0..NY {
            assert!((new_int[j] - 1.1).abs() < 1e-10);
        }
    }

    #[test]
    fn test_saturation_reduces_integration() {
        let aw = AntiWindup::default_cmp();
        let u_cmd = Vec11::from_element(8.0);
        let u_sat = Vec11::from_element(7.0); // saturated
        let integral = Vec21::from_element(1.0);
        let error = Vec21::from_element(0.1);

        let new_int = aw.conditional_integrate(&integral, &error, 1.0, &u_sat, &u_cmd);
        // Integration should be reduced
        for j in 0..NY {
            assert!(new_int[j] < 1.1); // less than full integration
            assert!(new_int[j] > 1.0); // but still some
        }
    }
}
