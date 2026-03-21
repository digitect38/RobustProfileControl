use nalgebra::DMatrix;
use serde::{Deserialize, Serialize};

use crate::types::*;

/// Diagonal weighting configuration for CMP control
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct WeightConfig {
    /// Per-output error weights (NY values). Higher at wafer edges.
    pub error_weights: Vec<f64>,
    /// Per-input effort weights (NU values). Higher for retaining ring.
    pub effort_weights: Vec<f64>,
    /// Per-input slew weights (NU values).
    pub slew_weights: Vec<f64>,
}

impl WeightConfig {
    /// Default physically motivated weighting for CMP profile control.
    ///
    /// Error weights are normalized by target range (50 Å), so that
    /// a 1-unit error weight corresponds to "target range matters".
    /// Edge points (indices 17-20) get 2× error weight.
    /// Effort and slew weights are scaled to balance against error in Å.
    pub fn default_cmp() -> Self {
        let r_out = radial_output_positions();

        // Error weight: 1/target_range so QP cost is in "multiples of spec"
        // This normalizes the error dimension to O(1) when error is at spec
        let base_error_weight = 1.0 / 50.0; // 1 / TARGET_RANGE
        let mut error_weights = vec![base_error_weight; NY];
        for j in 0..NY {
            let r_norm = r_out[j] / WAFER_RADIUS;
            if r_norm > 0.85 {
                error_weights[j] = base_error_weight * 2.0; // edge emphasis
            }
        }

        // Effort weight: penalize deviation from nominal; small relative to error
        let mut effort_weights = vec![0.01; NU];
        effort_weights[10] = 0.02; // retaining ring more sensitive

        // Slew weight: penalize fast pressure changes
        let slew_weights = vec![0.05; NU];

        WeightConfig {
            error_weights,
            effort_weights,
            slew_weights,
        }
    }

    /// Build diagonal error weight matrix W_e (ny × ny)
    pub fn build_we(&self) -> DMatrix<f64> {
        DMatrix::from_diagonal(&nalgebra::DVector::from_column_slice(&self.error_weights))
    }

    /// Build diagonal effort weight matrix W_u (nu × nu)
    pub fn build_wu(&self) -> DMatrix<f64> {
        DMatrix::from_diagonal(&nalgebra::DVector::from_column_slice(&self.effort_weights))
    }

    /// Build diagonal slew weight matrix W_Δu (nu × nu)
    pub fn build_wdu(&self) -> DMatrix<f64> {
        DMatrix::from_diagonal(&nalgebra::DVector::from_column_slice(&self.slew_weights))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_default_weights() {
        let w = WeightConfig::default_cmp();
        let we = w.build_we();
        let wu = w.build_wu();
        let wdu = w.build_wdu();

        assert_eq!(we.nrows(), NY);
        assert_eq!(we.ncols(), NY);
        assert_eq!(wu.nrows(), NU);
        assert_eq!(wdu.nrows(), NU);

        // Edge weights should be higher than center
        let center_idx = NY / 2;
        let edge_idx = NY - 1;
        assert!(w.error_weights[edge_idx] > w.error_weights[center_idx]);
    }
}
