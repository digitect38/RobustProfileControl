use nalgebra::{SMatrix, SVector};
use serde::{Deserialize, Serialize};

/// Profile output dimension (101 radial points, 0–150 mm at 1.5 mm spacing)
pub const NY: usize = 101;
/// Pressure input dimension (10 carrier zones + retaining ring)
pub const NU: usize = 11;

/// Fixed-size profile vector y ∈ R^21
pub type Vec21 = SVector<f64, NY>;
/// Fixed-size pressure vector u ∈ R^11
pub type Vec11 = SVector<f64, NU>;
/// Influence matrix G₀ ∈ R^{21×11}
pub type Mat21x11 = SMatrix<f64, NY, NU>;
/// Profile-space square matrix
pub type Mat21x21 = SMatrix<f64, NY, NY>;
/// Input-space square matrix
pub type Mat11x11 = SMatrix<f64, NU, NU>;

/// Actuator box constraints: pressure bounds and slew limits
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ActuatorBounds {
    /// Minimum pressure per zone (≥ 0)
    pub u_min: [f64; NU],
    /// Maximum pressure per zone
    pub u_max: [f64; NU],
    /// Minimum slew per step (typically negative)
    pub du_min: [f64; NU],
    /// Maximum slew per step (typically positive)
    pub du_max: [f64; NU],
}

impl ActuatorBounds {
    pub fn u_min_vec(&self) -> Vec11 {
        Vec11::from_column_slice(&self.u_min)
    }
    pub fn u_max_vec(&self) -> Vec11 {
        Vec11::from_column_slice(&self.u_max)
    }
    pub fn du_min_vec(&self) -> Vec11 {
        Vec11::from_column_slice(&self.du_min)
    }
    pub fn du_max_vec(&self) -> Vec11 {
        Vec11::from_column_slice(&self.du_max)
    }

    /// Compute effective bounds given previous command:
    /// effective_lb[i] = max(u_min[i], u_prev[i] + du_min[i])
    /// effective_ub[i] = min(u_max[i], u_prev[i] + du_max[i])
    pub fn effective_bounds(&self, u_prev: &Vec11) -> (Vec11, Vec11) {
        let mut lb = Vec11::zeros();
        let mut ub = Vec11::zeros();
        for i in 0..NU {
            lb[i] = self.u_min[i].max(u_prev[i] + self.du_min[i]);
            ub[i] = self.u_max[i].min(u_prev[i] + self.du_max[i]);
            // Ensure feasibility
            if lb[i] > ub[i] {
                let mid = 0.5 * (lb[i] + ub[i]);
                lb[i] = mid;
                ub[i] = mid;
            }
        }
        (lb, ub)
    }
}

/// Clamp each element of v to [lo, hi] componentwise
pub fn clamp_vec(v: &Vec11, lo: &Vec11, hi: &Vec11) -> Vec11 {
    let mut out = *v;
    for i in 0..NU {
        out[i] = out[i].clamp(lo[i], hi[i]);
    }
    out
}

/// Wafer radius in mm
pub const WAFER_RADIUS: f64 = 150.0;

/// Radial positions of the 21 output points (mm), 0 (center) to 150 (edge)
pub fn radial_output_positions() -> [f64; NY] {
    let mut r = [0.0; NY];
    for j in 0..NY {
        r[j] = j as f64 * WAFER_RADIUS / (NY as f64 - 1.0);
    }
    r
}

/// Carrier-head zone geometry.
/// Each zone is defined by [inner_radius, outer_radius] in mm.
/// Widths from center: 30, 20, 20, 20, 20, 10, 10, 10, 5, 5 mm
/// Zone 11 (retaining ring): starts at 150 mm (outside wafer edge), 20 mm wide.
pub struct ZoneGeometry {
    /// Inner radius of each zone (mm)
    pub inner: [f64; NU],
    /// Outer radius of each zone (mm)
    pub outer: [f64; NU],
    /// Center of each zone (mm)
    pub center: [f64; NU],
}

impl ZoneGeometry {
    pub fn default_cmp() -> Self {
        // Zone widths from center to edge: 30, 20, 20, 20, 20, 10, 10, 10, 5, 5
        let widths: [f64; 10] = [30.0, 20.0, 20.0, 20.0, 20.0, 10.0, 10.0, 10.0, 5.0, 5.0];
        let mut inner = [0.0; NU];
        let mut outer = [0.0; NU];
        let mut center = [0.0; NU];

        let mut r = 0.0;
        for i in 0..10 {
            inner[i] = r;
            outer[i] = r + widths[i];
            center[i] = r + widths[i] / 2.0;
            r += widths[i];
        }
        // Retaining ring: outside wafer, 150..170 mm
        inner[10] = 150.0;
        outer[10] = 170.0;
        center[10] = 160.0;

        ZoneGeometry { inner, outer, center }
    }
}

/// Gaussian kernel sigma for pressure distribution (mm).
/// Represents mechanical spreading broadened by wafer hardness.
pub const PRESSURE_SIGMA: f64 = 6.0;

/// Legacy API: zone centers as a 10-element array (used by some tests)
pub fn zone_centers() -> [f64; 10] {
    let geo = ZoneGeometry::default_cmp();
    let mut c = [0.0; 10];
    c.copy_from_slice(&geo.center[..10]);
    c
}
