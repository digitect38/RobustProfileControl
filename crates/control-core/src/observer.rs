use std::collections::VecDeque;

use nalgebra::{DMatrix, DVector};

use crate::types::*;

/// Kalman-style observer with delayed metrology handling.
///
/// The observer maintains a state estimate and handles the fact that
/// post-metrology measurements arrive with a delay of δ_u wafers.
///
/// When a delayed measurement y_k^m arrives (corresponding to wafer k - δ_u):
/// 1. Compute innovation: ε = y_k^m - ŷ_{k-δ_u|k-δ_u-1}
/// 2. Correct the delayed state estimate
/// 3. Roll forward to the current time using stored input history
#[derive(Clone, Debug)]
pub struct Observer {
    /// Current state estimate
    pub x_hat: DVector<f64>,
    /// State-space A matrix
    pub a: DMatrix<f64>,
    /// State-space B matrix (input)
    pub b: DMatrix<f64>,
    /// State-space C matrix (output)
    pub c: DMatrix<f64>,
    /// Observer gain L
    pub l_gain: DMatrix<f64>,
    /// Metrology delay in runs (wafers)
    pub delay: usize,
    /// Ring buffer of past inputs for roll-forward
    pub input_history: VecDeque<DVector<f64>>,
    /// Ring buffer of past state predictions at delayed timestamps
    pub prediction_history: VecDeque<DVector<f64>>,
    /// State dimension
    pub nx: usize,
}

impl Observer {
    /// Create a new observer.
    ///
    /// The state-space model is:
    ///   x_{k+1} = A x_k + B u_k
    ///   y_k     = C x_k
    ///
    /// `l_gain` is the observer gain (nx × ny).
    pub fn new(
        a: DMatrix<f64>,
        b: DMatrix<f64>,
        c: DMatrix<f64>,
        l_gain: DMatrix<f64>,
        delay: usize,
    ) -> Self {
        let nx = a.nrows();
        let x_hat = DVector::zeros(nx);

        Observer {
            x_hat,
            a,
            b,
            c,
            l_gain,
            delay,
            input_history: VecDeque::with_capacity(delay + 2),
            prediction_history: VecDeque::with_capacity(delay + 2),
            nx,
        }
    }

    /// Create a simple proportional observer for CMP.
    ///
    /// Uses a diagonal gain structure: L = gain * C^T (simple LQE-like).
    pub fn simple_cmp(g0: &Mat21x11, delay: usize, gain: f64) -> Self {
        // Simple integrator model: x_{k+1} = x_k + G0 * u_k
        // where x is the 21-dim profile state
        let nx = NY;
        let a = DMatrix::identity(nx, nx);
        let b = DMatrix::from_fn(NY, NU, |r, c_| g0[(r, c_)]);
        let c = DMatrix::identity(nx, nx);

        // Observer gain: simple proportional
        let l_gain = DMatrix::identity(nx, nx) * gain;

        Observer::new(a, b, c, l_gain, delay)
    }

    /// Predict step (no measurement): x̂_{k+1|k} = A x̂_{k|k} + B u_k
    pub fn predict(&mut self, u: &DVector<f64>) {
        // Store current prediction and input for delayed correction
        self.prediction_history.push_back(self.x_hat.clone());
        self.input_history.push_back(u.clone());

        // Keep only the last (delay + 1) entries
        while self.prediction_history.len() > self.delay + 2 {
            self.prediction_history.pop_front();
            self.input_history.pop_front();
        }

        // Predict
        self.x_hat = &self.a * &self.x_hat + &self.b * u;
    }

    /// Update with delayed metrology measurement.
    ///
    /// The measurement y_meas corresponds to a state from `delay` steps ago.
    /// We correct the delayed estimate and roll forward.
    pub fn update_with_delayed_metrology(&mut self, y_meas: &DVector<f64>) {
        if self.prediction_history.len() <= self.delay {
            // Not enough history yet; just do a direct correction
            let y_hat = &self.c * &self.x_hat;
            let innovation = y_meas - y_hat;
            self.x_hat += &self.l_gain * innovation;
            return;
        }

        // Get the delayed prediction
        let hist_len = self.prediction_history.len();
        let delayed_idx = if hist_len > self.delay {
            hist_len - 1 - self.delay
        } else {
            0
        };
        let x_delayed = self.prediction_history[delayed_idx].clone();

        // Innovation at the delayed time
        let y_hat_delayed = &self.c * &x_delayed;
        let innovation = y_meas - y_hat_delayed;

        // Correct the delayed state
        let x_corrected = x_delayed + &self.l_gain * &innovation;

        // Roll forward from corrected delayed state using stored inputs
        let mut x_rolled = x_corrected;
        for idx in delayed_idx..self.input_history.len() {
            x_rolled = &self.a * &x_rolled + &self.b * &self.input_history[idx];
        }

        self.x_hat = x_rolled;
    }

    /// Get current estimated profile output: ŷ = C x̂
    pub fn estimated_output(&self) -> DVector<f64> {
        &self.c * &self.x_hat
    }

    /// Get estimated profile as Vec21
    pub fn estimated_profile(&self) -> Vec21 {
        let y = self.estimated_output();
        Vec21::from_fn(|i, _| if i < y.len() { y[i] } else { 0.0 })
    }

    /// Reset observer state (e.g., at start of new lot)
    pub fn reset(&mut self) {
        self.x_hat = DVector::zeros(self.nx);
        self.input_history.clear();
        self.prediction_history.clear();
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::synth_data::generate_synthetic_g0;

    #[test]
    fn test_observer_creation() {
        let g0 = generate_synthetic_g0();
        let obs = Observer::simple_cmp(&g0, 2, 0.3);
        assert_eq!(obs.nx, NY);
        assert_eq!(obs.delay, 2);
    }

    #[test]
    fn test_observer_predict_update() {
        let g0 = generate_synthetic_g0();
        let mut obs = Observer::simple_cmp(&g0, 1, 0.5);

        let u = DVector::from_element(NU, 3.0);

        // Predict a few steps
        obs.predict(&u);
        obs.predict(&u);

        // Now provide a delayed measurement
        let y_meas = DVector::from_element(NY, 10.0);
        obs.update_with_delayed_metrology(&y_meas);

        let est = obs.estimated_profile();
        // After correction, estimate should be non-zero and influenced by measurement
        assert!(est.norm() > 0.0);
    }
}
