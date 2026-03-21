use nalgebra::{DMatrix, DVector, SVD};
use serde::{Deserialize, Serialize};

use crate::types::*;

/// Result of SVD decomposition of G₀ with mode selection
#[derive(Clone, Debug)]
pub struct SvdDecomposition {
    /// Full singular values (11 values, descending)
    pub singular_values: Vec<f64>,
    /// Full U matrix columns as profile-space vectors (21 x 21)
    pub u_columns: Vec<Vec<f64>>,
    /// Full V matrix columns as pressure-space vectors (11 x 11)
    pub v_columns: Vec<Vec<f64>>,
    /// Number of retained controllable modes
    pub rc: usize,
    /// Truncated U_rc (21 × rc) for projection
    pub u_rc: DMatrix<f64>,
    /// Truncated V_rc (11 × rc)
    pub v_rc: DMatrix<f64>,
    /// Truncated singular values (rc values)
    pub sigma_rc: DVector<f64>,
}

/// Serializable SVD info for the web frontend
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct SvdInfo {
    pub singular_values: Vec<f64>,
    pub energy_ratios: Vec<f64>,
    pub u_modes: Vec<Vec<f64>>,
    pub v_modes: Vec<Vec<f64>>,
    pub rc: usize,
}

impl SvdDecomposition {
    /// Compute SVD of G₀ and retain the top `rc` modes.
    pub fn from_plant(g0: &Mat21x11, rc: usize) -> Self {
        let rc = rc.min(NU).max(1);

        // Convert to DMatrix for SVD (nalgebra SVD works on dynamic matrices)
        let g_dyn = DMatrix::from_fn(NY, NU, |r, c| g0[(r, c)]);
        let svd = SVD::new(g_dyn, true, true);

        let u_full = svd.u.expect("SVD failed to compute U");
        let v_full = svd.v_t.expect("SVD failed to compute V^T").transpose();
        let sigma = &svd.singular_values;

        // Extract singular values
        let singular_values: Vec<f64> = (0..sigma.len()).map(|i| sigma[i]).collect();

        // Extract U columns (first NU columns are meaningful for a 21×11 matrix)
        let u_columns: Vec<Vec<f64>> = (0..u_full.ncols().min(NU))
            .map(|c| (0..NY).map(|r| u_full[(r, c)]).collect())
            .collect();

        // Extract V columns
        let v_columns: Vec<Vec<f64>> = (0..v_full.ncols().min(NU))
            .map(|c| (0..NU).map(|r| v_full[(r, c)]).collect())
            .collect();

        // Truncated matrices
        let u_rc = DMatrix::from_fn(NY, rc, |r, c| u_full[(r, c)]);
        let v_rc = DMatrix::from_fn(NU, rc, |r, c| v_full[(r, c)]);
        let sigma_rc = DVector::from_fn(rc, |i, _| sigma[i]);

        SvdDecomposition {
            singular_values,
            u_columns,
            v_columns,
            rc,
            u_rc,
            v_rc,
            sigma_rc,
        }
    }

    /// Project a 21-vector to reduced coordinates: z = U_rc^T * y
    pub fn project_to_reduced(&self, y: &Vec21) -> DVector<f64> {
        let y_dyn = DVector::from_fn(NY, |i, _| y[i]);
        self.u_rc.transpose() * y_dyn
    }

    /// Compute residual: y_perp = (I - U_rc * U_rc^T) * y
    pub fn residual(&self, y: &Vec21) -> Vec21 {
        let y_dyn = DVector::from_fn(NY, |i, _| y[i]);
        let projected = &self.u_rc * (self.u_rc.transpose() * &y_dyn);
        let res = y_dyn - projected;
        Vec21::from_fn(|i, _| res[i])
    }

    /// Residual energy: ||y_perp||^2
    pub fn residual_energy(&self, y: &Vec21) -> f64 {
        let r = self.residual(y);
        r.norm_squared()
    }

    /// Reduced plant matrix: G_reduced = U_rc^T * G0 (rc × 11)
    pub fn reduced_plant(&self, g0: &Mat21x11) -> DMatrix<f64> {
        let g_dyn = DMatrix::from_fn(NY, NU, |r, c| g0[(r, c)]);
        self.u_rc.transpose() * g_dyn
    }

    /// Cumulative energy ratio per mode:
    /// energy_ratio[k] = sum(sigma[0..=k]^2) / sum(sigma[0..N]^2)
    pub fn energy_ratios(&self) -> Vec<f64> {
        let total: f64 = self.singular_values.iter().map(|s| s * s).sum();
        let mut cumsum = 0.0;
        self.singular_values
            .iter()
            .map(|s| {
                cumsum += s * s;
                cumsum / total
            })
            .collect()
    }

    /// Build serializable info for the frontend
    pub fn to_info(&self) -> SvdInfo {
        SvdInfo {
            singular_values: self.singular_values.clone(),
            energy_ratios: self.energy_ratios(),
            u_modes: self.u_columns.iter().take(self.rc).cloned().collect(),
            v_modes: self.v_columns.iter().take(self.rc).cloned().collect(),
            rc: self.rc,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::synth_data::generate_synthetic_g0;

    #[test]
    fn test_svd_dimensions() {
        let g0 = generate_synthetic_g0();
        let svd = SvdDecomposition::from_plant(&g0, 8);
        assert_eq!(svd.singular_values.len(), NU);
        assert_eq!(svd.u_rc.nrows(), NY);
        assert_eq!(svd.u_rc.ncols(), 8);
        assert_eq!(svd.v_rc.nrows(), NU);
        assert_eq!(svd.v_rc.ncols(), 8);
        assert_eq!(svd.rc, 8);
    }

    #[test]
    fn test_svd_singular_values_descending() {
        let g0 = generate_synthetic_g0();
        let svd = SvdDecomposition::from_plant(&g0, 11);
        for i in 1..svd.singular_values.len() {
            assert!(svd.singular_values[i - 1] >= svd.singular_values[i]);
        }
    }

    #[test]
    fn test_energy_ratios_monotone() {
        let g0 = generate_synthetic_g0();
        let svd = SvdDecomposition::from_plant(&g0, 11);
        let er = svd.energy_ratios();
        for i in 1..er.len() {
            assert!(er[i] >= er[i - 1]);
        }
        assert!((er.last().unwrap() - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_projection_complement() {
        let g0 = generate_synthetic_g0();
        let svd = SvdDecomposition::from_plant(&g0, 6);
        let y = Vec21::from_fn(|i, _| (i as f64 + 1.0) * 0.1);
        let z = svd.project_to_reduced(&y);
        let res = svd.residual(&y);

        // Reconstructed + residual should equal original
        let y_dyn = DVector::from_fn(NY, |i, _| y[i]);
        let reconstructed = &svd.u_rc * z;
        let res_dyn = DVector::from_fn(NY, |i, _| res[i]);
        let sum = reconstructed + res_dyn;
        for i in 0..NY {
            assert!((sum[i] - y_dyn[i]).abs() < 1e-10);
        }
    }
}
