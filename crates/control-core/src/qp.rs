use nalgebra::{DMatrix, DVector};

use crate::types::*;

/// Box-constrained QP solver using projected gradient with Barzilai-Borwein step.
///
/// Problem: min 0.5 * x^T H x + f^T x  subject to lb ≤ x ≤ ub
#[derive(Clone, Debug)]
pub struct QpSolver {
    pub max_iter: usize,
    pub tol: f64,
}

/// QP problem data
pub struct QpProblem {
    /// Hessian (n × n, symmetric positive definite)
    pub h: DMatrix<f64>,
    /// Linear cost (n × 1)
    pub f: DVector<f64>,
    /// Lower bounds (n × 1)
    pub lb: DVector<f64>,
    /// Upper bounds (n × 1)
    pub ub: DVector<f64>,
}

/// QP solution
pub struct QpSolution {
    pub x: DVector<f64>,
    pub objective: f64,
    pub iterations: usize,
    pub converged: bool,
}

impl Default for QpSolver {
    fn default() -> Self {
        Self {
            max_iter: 200,
            tol: 1e-10,
        }
    }
}

impl QpSolver {
    pub fn new(max_iter: usize, tol: f64) -> Self {
        Self { max_iter, tol }
    }

    /// Project x onto the box [lb, ub]
    fn project(x: &DVector<f64>, lb: &DVector<f64>, ub: &DVector<f64>) -> DVector<f64> {
        DVector::from_fn(x.len(), |i, _| x[i].clamp(lb[i], ub[i]))
    }

    /// Solve the box-constrained QP using projected gradient with BB step size
    pub fn solve(&self, prob: &QpProblem) -> QpSolution {
        let n = prob.f.len();
        assert_eq!(prob.h.nrows(), n);
        assert_eq!(prob.h.ncols(), n);
        assert_eq!(prob.lb.len(), n);
        assert_eq!(prob.ub.len(), n);

        // Initial point: project midpoint of bounds
        let mut x = Self::project(
            &DVector::from_fn(n, |i, _| 0.5 * (prob.lb[i] + prob.ub[i])),
            &prob.lb,
            &prob.ub,
        );

        let mut grad = &prob.h * &x + &prob.f;
        let mut alpha = 1.0 / prob.h.diagonal().max();

        let mut x_prev = x.clone();
        let mut grad_prev = grad.clone();

        for iter in 0..self.max_iter {
            // Projected gradient step
            let x_new = Self::project(
                &(&x - alpha * &grad),
                &prob.lb,
                &prob.ub,
            );

            // Check convergence: projected gradient residual
            let residual = &x_new - &x;
            let res_norm = residual.norm();
            if res_norm < self.tol {
                let obj = 0.5 * x_new.dot(&(&prob.h * &x_new)) + prob.f.dot(&x_new);
                return QpSolution {
                    x: x_new,
                    objective: obj,
                    iterations: iter + 1,
                    converged: true,
                };
            }

            // Barzilai-Borwein step size
            if iter > 0 {
                let s = &x - &x_prev;
                let y_bb = &grad - &grad_prev;
                let sy = s.dot(&y_bb);
                if sy.abs() > 1e-30 {
                    let ss = s.dot(&s);
                    alpha = (ss / sy).abs().clamp(1e-10, 1e10);
                }
            }

            x_prev = x.clone();
            grad_prev = grad.clone();
            x = x_new;
            grad = &prob.h * &x + &prob.f;
        }

        let obj = 0.5 * x.dot(&(&prob.h * &x)) + prob.f.dot(&x);
        QpSolution {
            x,
            objective: obj,
            iterations: self.max_iter,
            converged: false,
        }
    }

    /// Build a CMP-specific QP from the control formulation:
    ///
    /// min |W_e (r - G u)|² + |W_u u|² + |W_Δu (u - u_prev)|²
    /// s.t. u_min ≤ u ≤ u_max, Δu_min ≤ u - u_prev ≤ Δu_max
    ///
    /// Expands to: min 0.5 u^T H u + f^T u  with merged box constraints
    pub fn build_cmp_qp(
        g: &DMatrix<f64>,
        r: &DVector<f64>,
        u_prev: &DVector<f64>,
        w_e: &DMatrix<f64>,
        w_u: &DMatrix<f64>,
        w_du: &DMatrix<f64>,
        bounds: &ActuatorBounds,
    ) -> QpProblem {
        let nu = g.ncols();

        // W_e * G
        let weg = w_e * g;
        // W_e * r
        let wer = w_e * r;

        // H = G^T W_e^T W_e G + W_u^T W_u + W_Δu^T W_Δu
        let h = weg.transpose() * &weg
            + w_u.transpose() * w_u
            + w_du.transpose() * w_du;

        // f = -G^T W_e^T W_e r + W_Δu^T W_Δu (-u_prev)
        //   = -(G^T W_e^T) (W_e r) - W_Δu^T W_Δu u_prev
        let f = -(weg.transpose() * &wer) - (w_du.transpose() * w_du) * u_prev;

        // Effective bounds: merge absolute and slew constraints
        let u_prev_11 = Vec11::from_fn(|i, _| {
            if i < u_prev.len() { u_prev[i] } else { 0.0 }
        });
        let (eff_lb, eff_ub) = bounds.effective_bounds(&u_prev_11);

        let lb = DVector::from_fn(nu, |i, _| eff_lb[i]);
        let ub = DVector::from_fn(nu, |i, _| eff_ub[i]);

        QpProblem { h, f, lb, ub }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_unconstrained_qp() {
        // min 0.5 * x^T I x + [-1, -2]^T x = min 0.5(x1² + x2²) - x1 - 2x2
        // Solution: x = [1, 2]
        let h = DMatrix::identity(2, 2);
        let f = DVector::from_vec(vec![-1.0, -2.0]);
        let lb = DVector::from_vec(vec![-100.0, -100.0]);
        let ub = DVector::from_vec(vec![100.0, 100.0]);

        let solver = QpSolver::default();
        let sol = solver.solve(&QpProblem { h, f, lb, ub });

        assert!(sol.converged);
        assert!((sol.x[0] - 1.0).abs() < 1e-6);
        assert!((sol.x[1] - 2.0).abs() < 1e-6);
    }

    #[test]
    fn test_constrained_qp() {
        // min 0.5 * x^T I x + [-3, -3]^T x
        // s.t. 0 ≤ x ≤ 2
        // Unconstrained: x = [3, 3], clamped: x = [2, 2]
        let h = DMatrix::identity(2, 2);
        let f = DVector::from_vec(vec![-3.0, -3.0]);
        let lb = DVector::from_vec(vec![0.0, 0.0]);
        let ub = DVector::from_vec(vec![2.0, 2.0]);

        let solver = QpSolver::default();
        let sol = solver.solve(&QpProblem { h, f, lb, ub });

        assert!(sol.converged);
        assert!((sol.x[0] - 2.0).abs() < 1e-6);
        assert!((sol.x[1] - 2.0).abs() < 1e-6);
    }

    #[test]
    fn test_cmp_qp_build_and_solve() {
        use crate::synth_data::*;

        let g0 = generate_synthetic_g0();
        let bounds = default_actuator_bounds();

        let g_dyn = DMatrix::from_fn(NY, NU, |r, c| g0[(r, c)]);
        let r = DVector::from_element(NY, 3.0);
        let u_prev = DVector::from_element(NU, 3.5);

        // Simple identity-like weights
        let w_e = DMatrix::identity(NY, NY) * 1.0;
        let w_u = DMatrix::identity(NU, NU) * 0.1;
        let w_du = DMatrix::identity(NU, NU) * 0.5;

        let prob = QpSolver::build_cmp_qp(&g_dyn, &r, &u_prev, &w_e, &w_u, &w_du, &bounds);
        let solver = QpSolver::default();
        let sol = solver.solve(&prob);

        assert!(sol.converged, "QP should converge");
        // Solution should respect bounds
        for i in 0..NU {
            assert!(sol.x[i] >= prob.lb[i] - 1e-8);
            assert!(sol.x[i] <= prob.ub[i] + 1e-8);
        }
    }
}
