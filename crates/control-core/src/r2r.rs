use nalgebra::{DMatrix, DVector};

use crate::antiwindup::AntiWindup;
use crate::observer::Observer;
use crate::qp::{QpSolver, QpSolution};
use crate::svd::SvdDecomposition;
use crate::types::*;
use crate::weighting::WeightConfig;

/// Run-to-Run (R2R) supervisory controller.
///
/// Operates at the wafer-to-wafer time scale. Updates the baseline recipe
/// using post-metrology data (potentially delayed). Uses SVD-based dimension
/// reduction for performance shaping while maintaining physical-space constraints.
#[derive(Clone, Debug)]
pub struct R2RController {
    /// SVD decomposition for dimension reduction
    svd: SvdDecomposition,
    /// Plant model (full-space)
    g0_dyn: DMatrix<f64>,
    /// Reduced plant (rc × NU)
    g_reduced: DMatrix<f64>,
    /// Observer for delayed metrology
    observer: Observer,
    /// Weight matrices
    w_e: DMatrix<f64>,
    w_u: DMatrix<f64>,
    w_du: DMatrix<f64>,
    /// Actuator bounds
    bounds: ActuatorBounds,
    /// Current recipe (the baseline that InRun will start from)
    pub current_recipe: Vec11,
    /// Previous recipe (for slew limiting)
    prev_recipe: Vec11,
    /// Mode-selective integral state (rc-dimensional)
    integral_state: DVector<f64>,
    /// Integral gain
    integral_gain: f64,
    /// Anti-windup
    antiwindup: AntiWindup,
    /// QP solver
    qp: QpSolver,
    /// Wafer counter
    wafer_count: usize,
    /// Metrology delay
    delay: usize,
}

impl R2RController {
    pub fn new(
        g0: &Mat21x11,
        svd: SvdDecomposition,
        bounds: ActuatorBounds,
        weights: &WeightConfig,
        metrology_delay: usize,
    ) -> Self {
        let g0_dyn = DMatrix::from_fn(NY, NU, |r, c| g0[(r, c)]);
        let g_reduced = svd.reduced_plant(g0);
        let observer = Observer::simple_cmp(g0, metrology_delay, 0.3);

        // Initial recipe: midpoint of bounds
        let initial_recipe = Vec11::from_fn(|i, _| {
            0.5 * (bounds.u_min[i] + bounds.u_max[i])
        });

        let rc = svd.rc;

        R2RController {
            svd,
            g0_dyn,
            g_reduced,
            observer,
            w_e: weights.build_we(),
            w_u: weights.build_wu(),
            w_du: weights.build_wdu(),
            bounds,
            current_recipe: initial_recipe,
            prev_recipe: initial_recipe,
            integral_state: DVector::zeros(rc),
            integral_gain: 0.2,
            antiwindup: AntiWindup::default_cmp(),
            qp: QpSolver::default(),
            wafer_count: 0,
            delay: metrology_delay,
        }
    }

    /// Process a new wafer's metrology and compute updated recipe.
    ///
    /// `target`: desired profile (21-dim)
    /// `y_measured`: post-metrology profile measurement (may be delayed)
    ///
    /// Returns the updated baseline recipe for the next wafer.
    pub fn step(&mut self, target: &Vec21, y_measured: &Vec21) -> Vec11 {
        self.wafer_count += 1;

        // Update observer with (delayed) measurement
        let y_dyn = DVector::from_fn(NY, |i, _| y_measured[i]);
        let u_dyn = DVector::from_fn(NU, |i, _| self.current_recipe[i]);
        self.observer.predict(&u_dyn);
        self.observer.update_with_delayed_metrology(&y_dyn);

        // Compute error in reduced coordinates
        let error_full = target - y_measured;
        let error_reduced = self.svd.project_to_reduced(&error_full);

        // Update mode-selective integral state with leaky integration for stability
        // Project anti-windup correction through U_rc^T
        let projection = self.svd.u_rc.transpose()
            * DMatrix::from_fn(NY, NU, |r, c| {
                if r == c && r < NU { 1.0 } else { 0.0 }
            });

        // Leaky integrator: decay old state to prevent windup/oscillation
        self.integral_state *= 0.85;

        self.integral_state = self.antiwindup.correct_integral_reduced(
            &self.integral_state,
            &error_reduced,
            self.integral_gain,
            &self.current_recipe,
            &self.current_recipe,
            &projection,
        );

        // Build augmented target: project (target + integral_correction) into reduced space
        let target_dyn = DVector::from_fn(NY, |i, _| target[i]);
        let target_reduced = self.svd.u_rc.transpose() * &target_dyn;
        let augmented_target_reduced = &target_reduced + &self.integral_state;

        // Lift back to full output space for QP
        let augmented_target_full = &self.svd.u_rc * augmented_target_reduced;

        let u_prev_dyn = DVector::from_fn(NU, |i, _| self.prev_recipe[i]);

        // Solve full-space constrained QP
        let prob = QpSolver::build_cmp_qp(
            &self.g0_dyn,
            &augmented_target_full,
            &u_prev_dyn,
            &self.w_e,
            &self.w_u,
            &self.w_du,
            &self.bounds,
        );

        let sol: QpSolution = self.qp.solve(&prob);

        self.prev_recipe = self.current_recipe;
        self.current_recipe = Vec11::from_fn(|i, _| sol.x[i]);
        self.current_recipe
    }

    /// Get the current baseline recipe
    pub fn recipe(&self) -> &Vec11 {
        &self.current_recipe
    }

    /// Get reduced-coordinate error for the last measurement
    pub fn reduced_error(&self, target: &Vec21, y: &Vec21) -> DVector<f64> {
        let error = target - y;
        self.svd.project_to_reduced(&error)
    }

    /// Get residual (uncontrollable) energy
    pub fn residual_energy(&self, y: &Vec21) -> f64 {
        self.svd.residual_energy(y)
    }

    /// Reset the controller state
    pub fn reset(&mut self) {
        self.integral_state = DVector::zeros(self.svd.rc);
        self.observer.reset();
        self.wafer_count = 0;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::synth_data::*;

    #[test]
    fn test_r2r_convergence() {
        let g0 = generate_synthetic_g0();
        let bounds = default_actuator_bounds();
        let weights = WeightConfig::default_cmp();
        let svd = SvdDecomposition::from_plant(&g0, 8);
        let mut ctrl = R2RController::new(&g0, svd, bounds, &weights, 1);

        let target = flat_target_profile(3.0);

        // Simulate several wafers with no disturbance
        let mut errors = Vec::new();
        for _k in 0..30 {
            let recipe = ctrl.current_recipe;
            let y = g0 * recipe; // no disturbance
            let error = (target - y).norm();
            errors.push(error);
            ctrl.step(&target, &y);
        }

        // Final errors should be significantly smaller than initial
        let initial_avg: f64 = errors[..3].iter().sum::<f64>() / 3.0;
        let final_avg: f64 = errors[errors.len()-5..].iter().sum::<f64>() / 5.0;
        assert!(
            final_avg < initial_avg,
            "R2R should converge: initial_avg={}, final_avg={}",
            initial_avg,
            final_avg,
        );
    }
}
