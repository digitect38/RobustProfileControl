// Simulation orchestration: calls WASM and distributes results to charts
// Supports realtime turn-by-turn playback of wafer profiles

import * as charts from './charts.js';
import * as controls from './controls.js';

let cachedResult = null;
let radialPos = [];
let bounds = null;
let zoneGeo = null;

// Animation state — single instance enforced by a generation counter
let animTimerId = null;
let animGeneration = 0;  // incremented on every start/stop to kill stale callbacks

export function setRadialPositions(pos) { radialPos = pos; }
export function setBounds(b) { bounds = b; }
export function setZoneGeometry(g) { zoneGeo = g; }
export function getCachedResult() { return cachedResult; }

export function stopAnimation() {
  animGeneration++;          // invalidate any pending callbacks
  if (animTimerId !== null) {
    clearTimeout(animTimerId);
    animTimerId = null;
  }
}

export function processResult(result) {
  stopAnimation();
  cachedResult = result;

  if (!result || result.error) {
    console.error('Simulation error:', result?.error);
    return;
  }

  const ws = result.wafer_snapshots;
  if (!ws || ws.length === 0) return;

  controls.setWaferSliderMax(ws.length - 1);

  const waferIndices = ws.map(w => w.wafer);
  const rmsErrors = ws.map(w => w.rms_error);
  const profileRanges = ws.map(w => w.profile_range);
  charts.updateR2RErrorChart(waferIndices, rmsErrors, profileRanges);
  charts.updateSvdCharts(result.svd_info, radialPos, result.config.rc);
  charts.updateSaturationChart(waferIndices, ws.map(w => w.saturation_count));

  // Metrics
  const last = ws[ws.length - 1];
  document.getElementById('metric-final-rms').textContent = last.rms_error.toFixed(1) + ' Å';
  document.getElementById('metric-final-edge').textContent = last.edge_error.toFixed(1) + ' Å';
  document.getElementById('metric-final-range').textContent = last.profile_range.toFixed(1) + ' Å';
  document.getElementById('metric-removal-rate').textContent = last.avg_removal_rate.toFixed(1) + ' Å/s';
  document.getElementById('metric-polish-time').textContent = last.polishing_time_sec.toFixed(0) + ' sec';
  document.getElementById('metric-saturation').textContent =
    ws.reduce((s, w) => s + w.saturation_count, 0);

  const svdInfo = result.svd_info;
  const rc = result.config.rc;
  if (svdInfo.energy_ratios && svdInfo.energy_ratios.length >= rc) {
    document.getElementById('metric-svd-energy').textContent =
      (svdInfo.energy_ratios[rc - 1] * 100).toFixed(1) + '%';
  }

  const tbody = document.getElementById('metrics-tbody');
  tbody.innerHTML = '';
  for (const w of ws) {
    const tr = document.createElement('tr');
    tr.innerHTML = `
      <td>${w.wafer}</td>
      <td>${w.rms_error.toFixed(1)}</td>
      <td>${w.profile_range.toFixed(1)}</td>
      <td>${w.edge_error.toFixed(1)}</td>
      <td>${w.avg_removal_rate.toFixed(1)}</td>
      <td>${w.saturation_count}</td>
    `;
    tbody.appendChild(tr);
  }

  updateWaferDetail(controls.getSelectedWafer());
}

export function updateWaferDetail(waferIdx) {
  if (!cachedResult) return;
  const ws = cachedResult.wafer_snapshots;
  if (waferIdx >= ws.length) waferIdx = ws.length - 1;
  const w = ws[waferIdx];

  charts.updateProfileChart(radialPos, w.target_profile, w.final_profile, w.initial_profile, null);
  charts.updatePressureChart(w.recipe);

  const turnData = cachedResult.turn_snapshots.filter(t => t.wafer === waferIdx);
  if (turnData.length > 0) {
    charts.updateInRunErrorChart(turnData);
    charts.updatePressureTimeChart(turnData);
  }
}

/// Animate turn-by-turn profile evolution for a specific wafer.
export function animateWafer(waferIdx, speedMs) {
  stopAnimation();                       // kill any previous animation

  if (!cachedResult) return;
  const ws = cachedResult.wafer_snapshots;
  if (waferIdx >= ws.length) return;
  const w = ws[waferIdx];

  const turnData = cachedResult.turn_snapshots.filter(t => t.wafer === waferIdx);
  if (turnData.length === 0) {
    updateWaferDetail(waferIdx);
    return;
  }

  const myGen = animGeneration;          // capture current generation
  let turnIdx = 0;
  const statusEl = document.getElementById('status-text');
  const nTurns = cachedResult.config.turns_per_wafer;

  function step() {
    if (myGen !== animGeneration) return; // stale — another anim started or stopped

    if (turnIdx >= turnData.length) {
      charts.updateProfileChart(radialPos, w.target_profile, w.final_profile, w.initial_profile, null);
      statusEl.textContent = `Wafer ${waferIdx}: Complete — Range: ${w.profile_range.toFixed(1)} Å`;
      animTimerId = null;
      return;
    }

    const t = turnData[turnIdx];
    const frac = (t.turn + 1) / nTurns;
    const trajectory = w.initial_profile.map((init, i) =>
      init * (1 - frac) + w.target_profile[i] * frac
    );

    charts.updateProfileChart(radialPos, w.target_profile, t.profile, w.initial_profile, trajectory);
    charts.updatePressureChart(t.pressure);

    statusEl.textContent = `Wafer ${waferIdx}, Turn ${t.turn}/${nTurns} (${t.time_sec.toFixed(0)}s) — RMS: ${t.rms_error.toFixed(1)} Å, Range: ${t.profile_range.toFixed(1)} Å`;

    turnIdx++;
    animTimerId = setTimeout(step, speedMs);
  }

  step();
}
