use nalgebra::{DMatrix, DVector};

use crate::antiwindup::AntiWindup;
use crate::qp::{QpSolver, QpSolution};
use crate::types::*;
use crate::weighting::WeightConfig;

/// InRun (in-wafer) controller.
///
/// Operates at the turn-wise time scale within a single wafer.
/// At each turn j, solves a constrained QP:
///
///   min |W_e(r_{k,j} - G u)|² + |W_u u|² + |W_Δu(u - u_prev)|²
///   s.t. u_min ≤ u ≤ u_max, Δu_min ≤ u - u_prev ≤ Δu_max
///
/// The baseline recipe from R2R is the starting point for each wafer.
#[derive(Clone, Debug)]
pub struct InRunController {
    /// Plant influence matrix for InRun (may be local linearization)
    g: DMatrix<f64>,
    /// Weight matrices
    w_e: DMatrix<f64>,
    w_u: DMatrix<f64>,
    w_du: DMatrix<f64>,
    /// Actuator bounds
    bounds: ActuatorBounds,
    /// Previous pressure command
    u_prev: Vec11,
    /// R2R baseline recipe
    u_r2r_baseline: Vec11,
    /// Integral error state (21-dim)
    integral_state: Vec21,
    /// Integral gain (scalar for simplicity)
    integral_gain: f64,
    /// Anti-windup module
    antiwindup: AntiWindup,
    /// QP solver
    qp: QpSolver,
    /// Last QP solve info
    last_iterations: usize,
    last_converged: bool,
    /// Count of turns where at least one actuator saturated
    pub saturation_count: usize,
}

impl InRunController {
    pub fn new(g0: &Mat21x11, bounds: ActuatorBounds, weights: &WeightConfig) -> Self {
        let g = DMatrix::from_fn(NY, NU, |r, c| g0[(r, c)]);

        InRunController {
            g,
            w_e: weights.build_we(),
            w_u: weights.build_wu(),
            w_du: weights.build_wdu(),
            bounds,
            u_prev: Vec11::from_element(3.5), // mid-range default
            u_r2r_baseline: Vec11::from_element(3.5),
            integral_state: Vec21::zeros(),
            integral_gain: 0.15,
            antiwindup: AntiWindup::default_cmp(),
            qp: QpSolver::default(),
            last_iterations: 0,
            last_converged: true,
            saturation_count: 0,
        }
    }

    /// Reset for a new wafer, receiving the updated R2R baseline recipe.
    pub fn reset_for_new_wafer(&mut self, u_r2r: &Vec11) {
        self.u_r2r_baseline = *u_r2r;
        self.u_prev = *u_r2r;
        self.integral_state = Vec21::zeros();
        self.saturation_count = 0;
    }

    /// Compute one turn-wise control action.
    ///
    /// Given target profile r_{k,j} and measured/estimated profile y_{k,j},
    /// returns the pressure command u_{k,j}.
    pub fn step(&mut self, target: &Vec21, measured: &Vec21) -> Vec11 {
        let error = target - measured;

        // Update integral state with anti-windup
        // (uses the saturation from the previous step)
        self.integral_state = self.antiwindup.conditional_integrate(
            &self.integral_state,
            &error,
            self.integral_gain,
            &self.u_prev, // u_sat (already clamped from previous step)
            &self.u_prev, // approximate: we don't store pre-clamp separately here
        );

        // Augmented reference: original target + integral correction
        let augmented_target_vec: Vec<f64> = (0..NY)
            .map(|i| target[i] + self.integral_state[i])
            .collect();
        let augmented_target = DVector::from_vec(augmented_target_vec);

        let u_prev_dyn = DVector::from_fn(NU, |i, _| self.u_prev[i]);

        // Build and solve the constrained QP
        let prob = QpSolver::build_cmp_qp(
            &self.g,
            &augmented_target,
            &u_prev_dyn,
            &self.w_e,
            &self.w_u,
            &self.w_du,
            &self.bounds,
        );

        let sol: QpSolution = self.qp.solve(&prob);
        self.last_iterations = sol.iterations;
        self.last_converged = sol.converged;

        // Convert solution to Vec11
        let u_cmd = Vec11::from_fn(|i, _| sol.x[i]);

        // Check for saturation (was anything clamped by the QP bounds?)
        let (lb, ub) = self.bounds.effective_bounds(&self.u_prev);
        let mut saturated = false;
        for i in 0..NU {
            if (u_cmd[i] - lb[i]).abs() < 1e-8 || (u_cmd[i] - ub[i]).abs() < 1e-8 {
                saturated = true;
                break;
            }
        }
        if saturated {
            self.saturation_count += 1;
        }

        self.u_prev = u_cmd;
        u_cmd
    }

    /// Get the last QP solve statistics
    pub fn last_solve_info(&self) -> (usize, bool) {
        (self.last_iterations, self.last_converged)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::synth_data::*;

    #[test]
    fn test_inrun_step() {
        let g0 = generate_synthetic_g0();
        let bounds = default_actuator_bounds();
        let weights = WeightConfig::default_cmp();
        let mut ctrl = InRunController::new(&g0, bounds.clone(), &weights);

        let target = flat_target_profile(3.0);
        let measured = flat_target_profile(2.5);

        let u = ctrl.step(&target, &measured);

        // Pressure should be within bounds
        for i in 0..NU {
            assert!(u[i] >= bounds.u_min[i] - 1e-8);
            assert!(u[i] <= bounds.u_max[i] + 1e-8);
        }
    }

    #[test]
    fn test_inrun_reset() {
        let g0 = generate_synthetic_g0();
        let bounds = default_actuator_bounds();
        let weights = WeightConfig::default_cmp();
        let mut ctrl = InRunController::new(&g0, bounds, &weights);

        let baseline = Vec11::from_element(4.0);
        ctrl.reset_for_new_wafer(&baseline);

        assert_eq!(ctrl.u_prev, baseline);
        assert_eq!(ctrl.integral_state, Vec21::zeros());
    }
}
