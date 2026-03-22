// Application bootstrap: load WASM, set up UI, wire events

import * as charts from './charts.js';
import * as controls from './controls.js';
import * as simulation from './simulation.js';

import init, {
  run_simulation,
  compute_svd,
  get_default_bounds,
  get_default_plant,
  get_radial_positions,
  get_zone_geometry,
} from '../pkg/wasm_bridge.js';

let g0Data = null;  // cached G₀ matrix

async function main() {
  const statusEl = document.getElementById('status-text');
  statusEl.textContent = 'Loading WASM...';

  try {
    await init();
  } catch (e) {
    statusEl.textContent = 'WASM load failed: ' + e.message;
    console.error('Failed to load WASM:', e);
    return;
  }

  statusEl.textContent = 'Ready';

  // Load defaults
  const radialPos = JSON.parse(get_radial_positions());
  const bounds = JSON.parse(get_default_bounds());
  const zoneGeo = JSON.parse(get_zone_geometry());
  simulation.setRadialPositions(radialPos);
  simulation.setBounds(bounds);
  simulation.setZoneGeometry(zoneGeo);
  window._zoneGeo = zoneGeo; // share with createAllCharts

  // Set up UI controls
  controls.setupRangeDisplays();

  // Create all charts
  createAllCharts();

  // Wire up tabs (before SVD init so tab switching works)
  document.querySelectorAll('.tab').forEach(tab => {
    tab.addEventListener('click', () => {
      document.querySelectorAll('.tab').forEach(t => t.classList.remove('active'));
      document.querySelectorAll('.tab-panel').forEach(p => p.classList.remove('active'));
      tab.classList.add('active');
      document.getElementById('panel-' + tab.dataset.tab).classList.add('active');
      // Trigger resize so Chart.js recalculates dimensions
      window.dispatchEvent(new Event('resize'));
    });
  });

  // Initial SVD display (after tabs are wired)
  const rc = parseInt(document.getElementById('cfg-rc').value) || 8;
  try {
    const svdInfo = JSON.parse(compute_svd(rc));
    charts.updateSvdCharts(svdInfo, radialPos, rc);
  } catch (e) {
    console.warn('Initial SVD display deferred:', e);
  }

  // Plot G₀ influence matrix and set up zone test
  try {
    g0Data = JSON.parse(get_default_plant());
    charts.updateInfluenceChart(radialPos, g0Data, zoneGeo.labels);
    charts.updateZoneCoverageChart(zoneGeo);

    // Build zone test sliders
    setupZoneTest(radialPos, g0Data, zoneGeo);
  } catch (e) {
    console.warn('Influence/zone-test init deferred:', e);
  }

  // Wire up Run button
  const btnRun = document.getElementById('btn-run');
  btnRun.addEventListener('click', () => {
    simulation.stopAnimation();
    btnRun.disabled = true;
    btnRun.classList.add('loading');
    statusEl.textContent = 'Simulating...';

    requestAnimationFrame(() => {
      setTimeout(() => {
        try {
          const config = controls.gatherConfig();
          const t0 = performance.now();
          const resultJson = run_simulation(JSON.stringify(config));
          const elapsed = (performance.now() - t0).toFixed(0);
          const result = JSON.parse(resultJson);
          simulation.processResult(result);
          const lastW = result.wafer_snapshots[result.wafer_snapshots.length-1];
          statusEl.textContent = `Done: ${result.wafer_snapshots.length} wafers | dist=${config.disturbance_amplitude} noise=${config.noise_amplitude} | RMS=${lastW.rms_error.toFixed(1)}Å range=${lastW.profile_range.toFixed(1)}Å (${elapsed}ms)`;

          // Debug: dump last wafer profile to console and debug panel
          const r_out = JSON.parse(get_radial_positions());
          let debugLines = [`=== Last Wafer (#${lastW.wafer}) Final Profile ===`];
          debugLines.push(`Config: dist=${config.disturbance_amplitude}, noise=${config.noise_amplitude}, inrun=${config.enable_inrun}, r2r=${config.enable_r2r}`);
          debugLines.push(`RMS=${lastW.rms_error.toFixed(4)}, range=${lastW.profile_range.toFixed(4)}, edge=${lastW.edge_error.toFixed(4)}`);
          debugLines.push('');
          debugLines.push('r(mm)     thickness    target       error');
          for (let j = 0; j < lastW.final_profile.length; j += 5) {
            const r = r_out[j].toFixed(1).padStart(6);
            const t = lastW.final_profile[j].toFixed(2).padStart(11);
            const tgt = lastW.target_profile[j].toFixed(2).padStart(9);
            const e = lastW.final_error[j].toFixed(2).padStart(10);
            debugLines.push(`${r}  ${t}  ${tgt}  ${e}`);
          }
          const debugEl = document.getElementById('debug-output');
          if (debugEl) debugEl.textContent = debugLines.join('\n');
          console.log(debugLines.join('\n'));
        } catch (e) {
          statusEl.textContent = 'Error: ' + e.message;
          console.error('Simulation error:', e);
        } finally {
          btnRun.disabled = false;
          btnRun.classList.remove('loading');
        }
      }, 50);
    });
  });

  // Wire up Animate button — plays turn-by-turn profile evolution
  document.getElementById('btn-animate').addEventListener('click', () => {
    const waferIdx = controls.getSelectedWafer();
    const speed = controls.getAnimSpeed();
    // Switch to Profile tab
    document.querySelectorAll('.tab').forEach(t => t.classList.remove('active'));
    document.querySelectorAll('.tab-panel').forEach(p => p.classList.remove('active'));
    document.querySelector('[data-tab="profile"]').classList.add('active');
    document.getElementById('panel-profile').classList.add('active');
    // Start animation
    simulation.animateWafer(waferIdx, speed);
  });

  // Wire up Stop button
  document.getElementById('btn-stop').addEventListener('click', () => {
    simulation.stopAnimation();
    statusEl.textContent = 'Stopped';
  });

  // Wire up Reset button
  document.getElementById('btn-reset').addEventListener('click', () => {
    simulation.stopAnimation();
    charts.destroyAll();
    createAllCharts();
    statusEl.textContent = 'Reset';
  });

  // Wire up wafer slider
  document.getElementById('wafer-slider').addEventListener('input', (e) => {
    simulation.stopAnimation();
    simulation.updateWaferDetail(parseInt(e.target.value));
  });

  // Wire up SVD mode slider
  document.getElementById('cfg-rc').addEventListener('input', (e) => {
    const newRc = parseInt(e.target.value);
    const info = JSON.parse(compute_svd(newRc));
    charts.updateSvdCharts(info, radialPos, newRc);
  });

  // Auto-run initial simulation
  btnRun.click();
}

function createAllCharts() {
  charts.createProfileChart(document.getElementById('chart-profile'));
  charts.createInfluenceChart(document.getElementById('chart-influence'));
  charts.createZoneCoverageChart(document.getElementById('chart-influence-heatmap'));
  charts.createZoneTestChart(document.getElementById('chart-zone-test'));
  charts.createZoneSoloChart(document.getElementById('chart-zone-solo'));
  const zLabels = window._zoneGeo ? window._zoneGeo.labels : null;
  charts.createPressureChart(document.getElementById('chart-pressure'), zLabels);
  charts.createErrorCharts(
    document.getElementById('chart-error-r2r'),
    document.getElementById('chart-error-inrun'),
  );
  charts.createPressureTimeChart(document.getElementById('chart-pressure-time'));
  charts.createSvdCharts(
    document.getElementById('chart-svd-values'),
    document.getElementById('chart-svd-energy'),
    document.getElementById('chart-svd-modes'),
  );
  charts.createSaturationChart(document.getElementById('chart-saturation'));
}

function setupZoneTest(radialPos, g0, zoneGeo) {
  const nZones = g0[0].length;
  const container = document.getElementById('zone-sliders');
  container.innerHTML = '';

  function getPressureVector() {
    const u = [];
    for (let i = 0; i < nZones; i++) {
      u.push(parseFloat(document.getElementById(`zt-${i}`).value) || 0);
    }
    return u;
  }

  function getYMax() {
    return parseFloat(document.getElementById('zt-ymax').value) || 80;
  }

  function applyAndPlot() {
    const u = getPressureVector();
    const nPts = g0.length;
    const perZone = [];
    const total = new Array(nPts).fill(0);
    for (let i = 0; i < nZones; i++) {
      const col = g0.map(row => row[i] * u[i]);
      perZone.push(col);
      for (let j = 0; j < nPts; j++) total[j] += col[j];
    }
    charts.updateZoneTestChart(radialPos, total, perZone, zoneGeo.labels, getYMax());
  }

  // Y-max slider
  const ymaxSlider = document.getElementById('zt-ymax');
  const ymaxVal = document.getElementById('zt-ymax-val');
  ymaxSlider.addEventListener('input', () => {
    ymaxVal.textContent = ymaxSlider.value;
    applyAndPlot();
  });

  // Create a slider for each zone
  for (let i = 0; i < nZones; i++) {
    const label = zoneGeo.labels[i] || `Zone ${i + 1}`;
    const div = document.createElement('div');
    div.className = 'zone-slider-row';
    div.innerHTML = `
      <span class="zone-slider-label" style="color:${ZONE_COLORS_APP[i]}">${label}</span>
      <input type="range" id="zt-${i}" min="0" max="7" step="0.1" value="0" class="zone-slider-input">
      <span class="zone-slider-val" id="zt-val-${i}">0.0</span>
      <span class="zone-slider-unit">psi</span>
    `;
    container.appendChild(div);
    // Live display
    const slider = div.querySelector(`#zt-${i}`);
    const valSpan = div.querySelector(`#zt-val-${i}`);
    slider.addEventListener('input', () => {
      valSpan.textContent = parseFloat(slider.value).toFixed(1);
      applyAndPlot();
    });
  }

  // Reset all to 0
  document.getElementById('btn-zone-reset').addEventListener('click', () => {
    for (let i = 0; i < nZones; i++) {
      document.getElementById(`zt-${i}`).value = 0;
      document.getElementById(`zt-val-${i}`).textContent = '0.0';
    }
    applyAndPlot();
  });

  // All to nominal (3.5 psi)
  document.getElementById('btn-zone-nominal').addEventListener('click', () => {
    for (let i = 0; i < nZones; i++) {
      document.getElementById(`zt-${i}`).value = 3.5;
      document.getElementById(`zt-val-${i}`).textContent = '3.5';
    }
    applyAndPlot();
  });

  // Solo sweep: plot each zone alone
  document.getElementById('btn-zone-solo').addEventListener('click', () => {
    charts.updateZoneSoloChart(radialPos, g0, zoneGeo.labels, 3.5);
  });

  // Initial solo sweep
  charts.updateZoneSoloChart(radialPos, g0, zoneGeo.labels, 3.5);
}

const ZONE_COLORS_APP = [
  '#e6194b', '#3cb44b', '#4363d8', '#f58231', '#911eb4',
  '#42d4f4', '#f032e6', '#bfef45', '#fabed4', '#469990',
  '#dcbeff',
];

main();
