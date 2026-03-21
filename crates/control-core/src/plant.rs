use crate::types::*;

/// CMP plant model: y = G * u + d
#[derive(Clone, Debug)]
pub struct Plant {
    /// Current influence matrix (may drift with wear)
    pub g: Mat21x11,
    /// Nominal influence matrix
    pub g0: Mat21x11,
    /// Actuator bounds
    pub bounds: ActuatorBounds,
}

impl Plant {
    pub fn new(g0: Mat21x11, bounds: ActuatorBounds) -> Self {
        Plant {
            g: g0,
            g0,
            bounds,
        }
    }

    /// Apply the plant: y = G * u + d
    pub fn apply(&self, u: &Vec11, d: &Vec21) -> Vec21 {
        self.g * u + d
    }

    /// Apply with explicit uncertainty: y = (G₀ + ΔG) * u + d
    pub fn apply_with_uncertainty(&self, u: &Vec11, d: &Vec21, delta_g: &Mat21x11) -> Vec21 {
        (self.g0 + delta_g) * u + d
    }

    /// Update the plant matrix (e.g., for wear drift)
    pub fn set_plant_matrix(&mut self, g: Mat21x11) {
        self.g = g;
    }

    /// Clamp a pressure vector to absolute bounds
    pub fn clamp_pressure(&self, u: &Vec11) -> Vec11 {
        clamp_vec(u, &self.bounds.u_min_vec(), &self.bounds.u_max_vec())
    }

    /// Clamp to both absolute and slew bounds given previous command
    pub fn clamp_full(&self, u_new: &Vec11, u_prev: &Vec11) -> Vec11 {
        let (lb, ub) = self.bounds.effective_bounds(u_prev);
        clamp_vec(u_new, &lb, &ub)
    }

    /// Compute incremental profile change for one InRun turn:
    /// y_{k,j+1} ≈ y_{k,j} + H * Δu + d
    pub fn incremental_step(
        &self,
        y_current: &Vec21,
        delta_u: &Vec11,
        d: &Vec21,
    ) -> Vec21 {
        y_current + self.g * delta_u + d
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::synth_data::*;

    #[test]
    fn test_plant_apply() {
        let g0 = generate_synthetic_g0();
        let bounds = default_actuator_bounds();
        let plant = Plant::new(g0, bounds);

        let u = Vec11::from_element(3.0);
        let d = Vec21::zeros();
        let y = plant.apply(&u, &d);

        // y should be non-zero and non-negative (G0 >= 0, u > 0)
        for j in 0..NY {
            assert!(y[j] > 0.0, "y[{}] = {} should be positive", j, y[j]);
        }
    }

    #[test]
    fn test_clamp_full() {
        let g0 = generate_synthetic_g0();
        let bounds = default_actuator_bounds();
        let plant = Plant::new(g0, bounds);

        let u_prev = Vec11::from_element(3.0);
        // Try a large jump
        let u_new = Vec11::from_element(10.0);
        let clamped = plant.clamp_full(&u_new, &u_prev);

        for i in 0..NU {
            assert!(clamped[i] <= plant.bounds.u_max[i] + 1e-10);
            assert!(clamped[i] - u_prev[i] <= plant.bounds.du_max[i] + 1e-10);
        }
    }
}
