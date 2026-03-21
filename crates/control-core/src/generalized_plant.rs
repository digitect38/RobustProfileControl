use nalgebra::DMatrix;
use serde::{Deserialize, Serialize};

use crate::types::*;
use crate::weighting::WeightConfig;

/// H-infinity generalized plant P for analysis and display.
///
/// The generalized plant partitions the closed-loop as:
///   [z  ]   [P11  P12] [w]
///   [y_m] = [P21  P22] [u]
///
/// where:
///   z = [W_e * e; W_u * u; W_Δu * Δu] (performance output)
///   w = [r; d; n]                       (exogenous inputs)
///   y_m = y + n                         (measured output)
///
/// State-space form:
///   x_{k+1} = A_P x_k + B_P1 w_k + B_P2 u_k
///   z_k     = C_P1 x_k + D_P11 w_k + D_P12 u_k
///   y_m_k   = C_P2 x_k + D_P21 w_k + D_P22 u_k
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct GeneralizedPlantInfo {
    /// State dimension
    pub nx: usize,
    /// Performance output dimension
    pub nz: usize,
    /// Exogenous input dimension
    pub nw: usize,
    /// Measured output dimension
    pub ny: usize,
    /// Control input dimension
    pub nu: usize,
    /// A_P matrix (flattened row-major)
    pub a_p: Vec<f64>,
    /// B_P1 matrix (flattened)
    pub b_p1: Vec<f64>,
    /// B_P2 matrix (flattened)
    pub b_p2: Vec<f64>,
    /// D_P11 matrix (flattened)
    pub d_p11: Vec<f64>,
    /// D_P12 matrix (flattened)
    pub d_p12: Vec<f64>,
}

/// Build the generalized plant matrices for analysis.
///
/// Simple static formulation (no delay augmentation):
///   Plant: y = G₀ u + d
///   Error: e = r - y = r - G₀ u - d
///
/// State: x = [x_I] (integral state, 21-dim)
///   x_{k+1} = x_k + (r_k - G₀ u_k - d_k)
///
/// Performance output:
///   z = [W_e * (r - G₀ u - d); W_u * u; W_Δu * u]  (Δu ≈ u for static analysis)
///
/// Measured output:
///   y_m = G₀ u + d + n
pub fn build_generalized_plant(
    g0: &Mat21x11,
    weights: &WeightConfig,
) -> GeneralizedPlantInfo {
    let g = DMatrix::from_fn(NY, NU, |r, c| g0[(r, c)]);
    let w_e = weights.build_we();
    let w_u = weights.build_wu();
    let w_du = weights.build_wdu();

    // State dimension: 21 (integral state)
    let nx = NY;
    // Performance output: W_e*e (21) + W_u*u (11) + W_Δu*u (11) = 43
    let nz = NY + NU + NU;
    // Exogenous: r (21) + d (21) + n (21) = 63
    let nw = NY + NY + NY;
    let ny = NY; // measured output
    let nu = NU; // control input

    // A_P = I_{21} (integrator dynamics)
    let a_p = DMatrix::identity(nx, nx);

    // B_P1 = [I, -I, 0] (r adds to integral, d subtracts)
    let mut b_p1 = DMatrix::zeros(nx, nw);
    for i in 0..nx {
        b_p1[(i, i)] = 1.0; // r
        b_p1[(i, NY + i)] = -1.0; // -d
    }

    // B_P2 = -G₀ (control input reduces integral error)
    let b_p2 = -&g;

    // D_P11: how exogenous inputs appear in performance output
    // z = [W_e*(r-d); 0; 0] when u=0
    let mut d_p11 = DMatrix::zeros(nz, nw);
    // W_e * r (first block)
    for i in 0..NY {
        for j in 0..NY {
            d_p11[(i, j)] = w_e[(i, j)]; // W_e * I (from r)
        }
    }
    // W_e * (-d) (first block, d part)
    for i in 0..NY {
        for j in 0..NY {
            d_p11[(i, NY + j)] = -w_e[(i, j)]; // -W_e * I (from d)
        }
    }

    // D_P12: how control input appears in performance output
    // z = [-W_e*G₀*u; W_u*u; W_Δu*u]
    let mut d_p12 = DMatrix::zeros(nz, nu);
    let neg_weg = -&w_e * &g;
    for i in 0..NY {
        for j in 0..nu {
            d_p12[(i, j)] = neg_weg[(i, j)];
        }
    }
    for i in 0..nu {
        for j in 0..nu {
            d_p12[(NY + i, j)] = w_u[(i, j)];
            d_p12[(NY + NU + i, j)] = w_du[(i, j)];
        }
    }

    GeneralizedPlantInfo {
        nx,
        nz,
        nw,
        ny,
        nu,
        a_p: a_p.as_slice().to_vec(),
        b_p1: b_p1.as_slice().to_vec(),
        b_p2: b_p2.as_slice().to_vec(),
        d_p11: d_p11.as_slice().to_vec(),
        d_p12: d_p12.as_slice().to_vec(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::synth_data::generate_synthetic_g0;

    #[test]
    fn test_generalized_plant_dimensions() {
        let g0 = generate_synthetic_g0();
        let weights = WeightConfig::default_cmp();
        let gp = build_generalized_plant(&g0, &weights);

        assert_eq!(gp.nx, NY);
        assert_eq!(gp.nz, NY + NU + NU);
        assert_eq!(gp.nw, NY * 3);
        assert_eq!(gp.ny, NY);
        assert_eq!(gp.nu, NU);
        assert_eq!(gp.a_p.len(), NY * NY);
        assert_eq!(gp.b_p2.len(), NY * NU);
    }
}
