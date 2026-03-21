use wasm_bindgen::prelude::*;

use control_core::generalized_plant;
use control_core::simulation::{self, SimConfig};
use control_core::svd::SvdDecomposition;
use control_core::synth_data;
use control_core::types::*;
use control_core::weighting::WeightConfig;

/// Initialize the WASM module (call once on page load)
#[wasm_bindgen]
pub fn init() {
    console_error_panic_hook::set_once();
}

/// Run a full simulation with the given config JSON.
/// Returns the simulation result as a JSON string.
#[wasm_bindgen]
pub fn run_simulation(config_json: &str) -> String {
    let config: SimConfig = match serde_json::from_str(config_json) {
        Ok(c) => c,
        Err(e) => return format!("{{\"error\": \"{}\"}}", e),
    };
    let result = simulation::run_simulation(&config);
    serde_json::to_string(&result).unwrap_or_else(|e| format!("{{\"error\": \"{}\"}}", e))
}

/// Compute SVD of the default synthetic plant with `rc` retained modes.
/// Returns SVD info as JSON.
#[wasm_bindgen]
pub fn compute_svd(rc: usize) -> String {
    let g0 = synth_data::generate_synthetic_g0();
    let svd = SvdDecomposition::from_plant(&g0, rc);
    let info = svd.to_info();
    serde_json::to_string(&info).unwrap_or_else(|e| format!("{{\"error\": \"{}\"}}", e))
}

/// Get the default synthetic G0 matrix as JSON (21x11 array-of-arrays).
#[wasm_bindgen]
pub fn get_default_plant() -> String {
    let g0 = synth_data::generate_synthetic_g0();
    let rows: Vec<Vec<f64>> = (0..NY)
        .map(|r| (0..NU).map(|c| g0[(r, c)]).collect())
        .collect();
    serde_json::to_string(&rows).unwrap()
}

/// Get default actuator bounds as JSON.
#[wasm_bindgen]
pub fn get_default_bounds() -> String {
    let bounds = synth_data::default_actuator_bounds();
    serde_json::to_string(&bounds).unwrap()
}

/// Get default weight configuration as JSON.
#[wasm_bindgen]
pub fn get_default_weights() -> String {
    let weights = WeightConfig::default_cmp();
    serde_json::to_string(&weights).unwrap()
}

/// Get default simulation config as JSON.
#[wasm_bindgen]
pub fn get_default_config() -> String {
    let config = SimConfig::default();
    serde_json::to_string(&config).unwrap()
}

/// Compute the generalized plant P matrices for H-inf analysis display.
#[wasm_bindgen]
pub fn compute_generalized_plant() -> String {
    let g0 = synth_data::generate_synthetic_g0();
    let weights = WeightConfig::default_cmp();
    let gp = generalized_plant::build_generalized_plant(&g0, &weights);
    serde_json::to_string(&gp).unwrap_or_else(|e| format!("{{\"error\": \"{}\"}}", e))
}

/// Get radial output positions (21 values in mm) as JSON.
#[wasm_bindgen]
pub fn get_radial_positions() -> String {
    let pos = radial_output_positions();
    serde_json::to_string(&pos.to_vec()).unwrap()
}

/// Get zone geometry as JSON: { inner: [...], outer: [...], center: [...] }
#[wasm_bindgen]
pub fn get_zone_geometry() -> String {
    let geo = ZoneGeometry::default_cmp();
    let result = serde_json::json!({
        "inner": geo.inner.to_vec(),
        "outer": geo.outer.to_vec(),
        "center": geo.center.to_vec(),
        "labels": ["Z1(30)", "Z2(20)", "Z3(20)", "Z4(20)", "Z5(20)",
                    "Z6(10)", "Z7(10)", "Z8(10)", "Z9(5)", "Z10(5)", "RR(20)"],
        "sigma": PRESSURE_SIGMA,
    });
    result.to_string()
}
