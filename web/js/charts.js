// Chart.js chart creation and update functions — CMP Physical Units (Å, psi, mm)

const COLORS = {
  blue: '#143D6B',
  teal: '#0F7C82',
  red: '#c0392b',
  green: '#27ae60',
  orange: '#e67e22',
  purple: '#8e44ad',
  softBlue: 'rgba(20,61,107,0.15)',
  softTeal: 'rgba(15,124,130,0.15)',
  softRed: 'rgba(192,57,43,0.15)',
  softGreen: 'rgba(39,174,96,0.15)',
};

function zoneColor(i) {
  const hue = 210 - (i / 10) * 180;
  return `hsl(${hue}, 65%, 50%)`;
}

let charts = {};

const MODE_COLORS = ['#143D6B', '#0F7C82', '#c0392b', '#e67e22', '#8e44ad', '#27ae60', '#2980b9', '#d35400', '#7f8c8d', '#2c3e50', '#16a085'];

export function destroyAll() {
  Object.entries(charts).forEach(([key, c]) => {
    if (c && typeof c.destroy === 'function') c.destroy();
  });
  charts = {};
}

// ---- Profile Chart (realtime turn-by-turn, -150 to +150 mm) ----
export function createProfileChart(ctx) {
  charts.profile = new Chart(ctx, {
    type: 'line',
    data: {
      labels: [],
      datasets: [
        {
          label: 'Target (2000 Å)',
          borderColor: COLORS.blue,
          borderDash: [6, 3],
          backgroundColor: COLORS.softBlue,
          fill: false,
          tension: 0.3,
          pointRadius: 2,
          borderWidth: 2,
          data: [],
        },
        {
          label: 'Initial Profile',
          borderColor: '#aaa',
          borderDash: [3, 3],
          backgroundColor: 'transparent',
          fill: false,
          tension: 0.3,
          pointRadius: 0,
          borderWidth: 1,
          data: [],
        },
        {
          label: 'Current Thickness',
          borderColor: COLORS.teal,
          backgroundColor: COLORS.softTeal,
          fill: false,
          tension: 0.3,
          pointRadius: 3,
          borderWidth: 2.5,
          data: [],
        },
        {
          label: 'Trajectory Target',
          borderColor: COLORS.green,
          borderDash: [4, 2],
          backgroundColor: 'transparent',
          fill: false,
          tension: 0.3,
          pointRadius: 0,
          borderWidth: 1.5,
          data: [],
        },
      ],
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      animation: { duration: 0 },
      plugins: {
        title: { display: true, text: 'Wafer Thickness Profile', color: COLORS.blue, font: { size: 14 } },
        legend: { position: 'top', labels: { boxWidth: 20, font: { size: 11 } } },
      },
      scales: {
        x: {
          title: { display: true, text: 'Radial Position (mm)', font: { size: 12 } },
          ticks: { maxTicksLimit: 11 },
        },
        y: {
          title: { display: true, text: 'Thickness (Å)', font: { size: 12 } },
          suggestedMin: 0,
          suggestedMax: 12000,
        },
      },
    },
  });
}

export function updateProfileChart(radialPos, target, current, initial, trajectory) {
  const c = charts.profile;
  if (!c) return;
  // Build symmetric x-axis: -150 to +150 mm (mirror the 0..150 data)
  const labels = radialPos.map(r => (-r).toFixed(1)).reverse()
    .concat(radialPos.slice(1).map(r => r.toFixed(1)));

  const mirrorData = (arr) => {
    if (!arr || arr.length === 0) return [];
    return [...arr].reverse().concat(arr.slice(1));
  };

  c.data.labels = labels;
  c.data.datasets[0].data = mirrorData(target);
  c.data.datasets[1].data = mirrorData(initial);
  c.data.datasets[2].data = mirrorData(current);
  c.data.datasets[3].data = trajectory ? mirrorData(trajectory) : [];

  // Auto-scale Y based on data
  const allVals = [...(current || []), ...(initial || [])].filter(v => v != null);
  if (allVals.length > 0) {
    const yMin = Math.min(...allVals) - 200;
    const yMax = Math.max(...allVals) + 200;
    c.options.scales.y.suggestedMin = Math.max(0, yMin);
    c.options.scales.y.suggestedMax = yMax;
  }

  c.update();
}

// ---- Trajectory Chart (thickness vs time) ----

export function createTrajectoryChart(ctx) {
  charts.trajectory = new Chart(ctx, {
    type: 'line',
    data: {
      labels: [],
      datasets: [
        {
          label: 'Trajectory (center)',
          borderColor: COLORS.blue,
          borderDash: [6, 3],
          backgroundColor: 'transparent',
          tension: 0.3,
          pointRadius: 0,
          borderWidth: 2,
          data: [],
        },
        {
          label: 'Trajectory (edge)',
          borderColor: COLORS.teal,
          borderDash: [6, 3],
          backgroundColor: 'transparent',
          tension: 0.3,
          pointRadius: 0,
          borderWidth: 2,
          data: [],
        },
        {
          label: 'Actual (center)',
          borderColor: COLORS.blue,
          backgroundColor: 'transparent',
          tension: 0.3,
          pointRadius: 0,
          borderWidth: 2.5,
          data: [],
        },
        {
          label: 'Actual (edge)',
          borderColor: COLORS.teal,
          backgroundColor: 'transparent',
          tension: 0.3,
          pointRadius: 0,
          borderWidth: 2.5,
          data: [],
        },
        {
          label: 'Target (2000 Å)',
          borderColor: '#c0392b',
          borderDash: [3, 3],
          backgroundColor: 'transparent',
          pointRadius: 0,
          borderWidth: 1.5,
          data: [],
        },
      ],
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      animation: { duration: 0 },
      plugins: {
        title: { display: true, text: 'Thickness Trajectory vs. Time', color: COLORS.blue, font: { size: 12 } },
        legend: { position: 'top', labels: { boxWidth: 16, font: { size: 10 }, padding: 6 } },
      },
      scales: {
        x: { title: { display: true, text: 'Turn (sec)' }, ticks: { maxTicksLimit: 10 } },
        y: { title: { display: true, text: 'Thickness (Å)' }, suggestedMin: 0, suggestedMax: 12000 },
      },
    },
  });
}

export function updateTrajectoryChart(config, initialProfile, targetProfile, turnSnapshots) {
  const c = charts.trajectory;
  if (!c) return;

  const nTurns = config.turns_per_wafer;
  const shape = config.trajectory_shape || 1.0;
  const centerIdx = 0;
  const edgeIdx = initialProfile.length - 1;
  const initCenter = initialProfile[centerIdx];
  const initEdge = initialProfile[edgeIdx];
  const tgtCenter = targetProfile[centerIdx];
  const tgtEdge = targetProfile[edgeIdx];

  // Sample every N turns to keep data small
  const step = Math.max(1, Math.floor(nTurns / 80));
  const labels = [];
  const trajCenter = [];
  const trajEdge = [];
  const targetLine = [];
  for (let j = 0; j <= nTurns; j += step) {
    labels.push(j);
    const t = j / nTurns;
    const progress = Math.pow(Math.min(t, 1.0), shape);
    trajCenter.push(initCenter * (1 - progress) + tgtCenter * progress);
    trajEdge.push(initEdge * (1 - progress) + tgtEdge * progress);
    targetLine.push(2000);
  }

  // Build actual thickness from snapshots — use a lookup map by turn number
  const actCenterMap = new Map();
  const actEdgeMap = new Map();
  if (turnSnapshots && turnSnapshots.length > 0) {
    for (const t of turnSnapshots) {
      actCenterMap.set(t.turn, t.profile[centerIdx]);
      actEdgeMap.set(t.turn, t.profile[edgeIdx]);
    }
  }

  const actCenter = labels.map(j => actCenterMap.has(j) ? actCenterMap.get(j) : null);
  const actEdge = labels.map(j => actEdgeMap.has(j) ? actEdgeMap.get(j) : null);

  c.data.labels = labels;
  c.data.datasets[0].data = trajCenter;
  c.data.datasets[1].data = trajEdge;
  c.data.datasets[2].data = actCenter;
  c.data.datasets[3].data = actEdge;
  c.data.datasets[4].data = targetLine;

  c.options.scales.y.suggestedMin = Math.min(tgtCenter, tgtEdge) - 500;
  c.options.scales.y.suggestedMax = Math.max(initCenter, initEdge) + 500;

  c.update();
}

// ---- Influence Matrix Chart (G₀ columns) ----

const ZONE_COLORS = [
  '#e6194b', '#3cb44b', '#4363d8', '#f58231', '#911eb4',
  '#42d4f4', '#f032e6', '#bfef45', '#fabed4', '#469990',
  '#dcbeff',  // retaining ring
];

export function createInfluenceChart(ctx) {
  charts.influence = new Chart(ctx, {
    type: 'line',
    data: { labels: [], datasets: [] },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      plugins: {
        title: {
          display: true,
          text: 'G₀ Influence Matrix — Removal Rate per Zone (Å / psi·turn)',
          color: COLORS.blue,
          font: { size: 14 },
        },
        legend: {
          position: 'right',
          labels: { boxWidth: 14, font: { size: 10 }, padding: 6 },
        },
      },
      scales: {
        x: {
          title: { display: true, text: 'Radial Position (mm)', font: { size: 12 } },
          ticks: { maxTicksLimit: 13 },
        },
        y: {
          title: { display: true, text: 'Removal Rate (Å / psi·turn)', font: { size: 12 } },
          beginAtZero: true,
        },
      },
    },
  });
}

export function updateInfluenceChart(radialPos, g0, zoneLabels) {
  const c = charts.influence;
  if (!c) return;

  // Mirror to -150..+150
  const labels = radialPos.map(r => (-r).toFixed(1)).reverse()
    .concat(radialPos.slice(1).map(r => r.toFixed(1)));

  const mirrorData = (arr) => [...arr].reverse().concat(arr.slice(1));

  // g0 is 21×11 (rows=output points, cols=zones)
  // Extract each column as a dataset
  const nZones = g0[0].length;
  const datasets = [];

  for (let i = 0; i < nZones; i++) {
    const col = g0.map(row => row[i]);
    const label = zoneLabels ? zoneLabels[i] : `Zone ${i + 1}`;
    datasets.push({
      label,
      data: mirrorData(col),
      borderColor: ZONE_COLORS[i % ZONE_COLORS.length],
      backgroundColor: 'transparent',
      borderWidth: i === nZones - 1 ? 2.5 : 1.8, // RR thicker
      borderDash: i === nZones - 1 ? [6, 3] : [],  // RR dashed
      tension: 0.3,
      pointRadius: 0,
    });
  }

  c.data.labels = labels;
  c.data.datasets = datasets;
  c.update();
}

// Zone coverage chart — shows each zone as a colored band on the radial axis
export function createZoneCoverageChart(ctx) {
  charts.zoneCoverage = new Chart(ctx, {
    type: 'bar',
    data: { labels: ['Carrier Zones'], datasets: [] },
    options: {
      indexAxis: 'y',
      responsive: true,
      maintainAspectRatio: false,
      plugins: {
        title: { display: true, text: 'Zone Layout: 0 (center) → 150 mm (edge) → RR', color: COLORS.blue },
        legend: {
          position: 'bottom',
          labels: { boxWidth: 12, font: { size: 9 }, padding: 4 },
        },
        tooltip: {
          callbacks: {
            label: (ctx) => {
              const ds = ctx.dataset;
              const d = ds.data[0];
              return `${ds.label}: ${d[0]}–${d[1]} mm`;
            }
          }
        },
      },
      scales: {
        x: {
          title: { display: true, text: 'Radius (mm)' },
          min: 0, max: 175,
          ticks: { stepSize: 15 },
        },
        y: { display: false },
      },
    },
  });
}

export function updateZoneCoverageChart(zoneGeo) {
  const c = charts.zoneCoverage;
  if (!c || !zoneGeo) return;

  // Each zone is a floating bar [inner, outer]
  c.data.datasets = zoneGeo.labels.map((label, i) => ({
    label,
    data: [[zoneGeo.inner[i], zoneGeo.outer[i]]],
    backgroundColor: ZONE_COLORS[i % ZONE_COLORS.length] + '99', // semi-transparent
    borderColor: ZONE_COLORS[i % ZONE_COLORS.length],
    borderWidth: 1,
    barPercentage: 1.0,
    categoryPercentage: 0.9,
  }));

  c.update();
}

// ---- Pressure Chart ----
export function createPressureChart(ctx, zoneLabels) {
  const labels = zoneLabels || [
    'Z1(30)', 'Z2(20)', 'Z3(20)', 'Z4(20)', 'Z5(20)',
    'Z6(10)', 'Z7(10)', 'Z8(10)', 'Z9(5)', 'Z10(5)', 'RR(20)'
  ];

  charts.pressure = new Chart(ctx, {
    type: 'bar',
    data: {
      labels,
      datasets: [{
        label: 'Pressure (psi)',
        data: new Array(11).fill(0),
        backgroundColor: labels.map((_, i) => zoneColor(i)),
        borderColor: labels.map((_, i) => zoneColor(i)),
        borderWidth: 1,
      }],
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      animation: { duration: 0 },
      plugins: {
        title: { display: true, text: 'Pressure Commands (psi)', color: COLORS.blue },
      },
      scales: {
        y: { beginAtZero: true, max: 8, title: { display: true, text: 'Pressure (psi)' } },
      },
    },
  });
}

export function updatePressureChart(pressure) {
  const c = charts.pressure;
  if (!c) return;
  c.data.datasets[0].data = pressure;
  c.update();
}

// ---- Error Charts (R2R wafer-level + InRun turn-level) ----
export function createErrorCharts(ctxR2R, ctxInRun) {
  charts.errorR2R = new Chart(ctxR2R, {
    type: 'line',
    data: {
      labels: [],
      datasets: [{
        label: 'RMS Error (Å)',
        borderColor: COLORS.blue,
        backgroundColor: COLORS.softBlue,
        fill: true,
        tension: 0.3,
        pointRadius: 3,
        data: [],
      }, {
        label: 'Profile Range (Å)',
        borderColor: COLORS.orange,
        backgroundColor: 'transparent',
        fill: false,
        tension: 0.3,
        pointRadius: 2,
        borderDash: [4, 2],
        data: [],
      }, {
        label: 'Target Range (50 Å)',
        borderColor: COLORS.green,
        borderDash: [6, 3],
        backgroundColor: 'transparent',
        fill: false,
        pointRadius: 0,
        data: [],
      }],
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      plugins: {
        title: { display: true, text: 'R2R: Error per Wafer', color: COLORS.blue },
      },
      scales: {
        x: { title: { display: true, text: 'Wafer #' } },
        y: { title: { display: true, text: 'Å' }, beginAtZero: true },
      },
    },
  });

  charts.errorInRun = new Chart(ctxInRun, {
    type: 'line',
    data: {
      labels: [],
      datasets: [{
        label: 'RMS Error (Å)',
        borderColor: COLORS.teal,
        backgroundColor: COLORS.softTeal,
        fill: true,
        tension: 0.3,
        pointRadius: 1,
        data: [],
      }, {
        label: 'Profile Range (Å)',
        borderColor: COLORS.orange,
        backgroundColor: 'transparent',
        tension: 0.3,
        pointRadius: 0,
        borderDash: [3, 2],
        data: [],
      }],
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      plugins: {
        title: { display: true, text: 'InRun: Error per Turn (selected wafer)', color: COLORS.teal },
      },
      scales: {
        x: { title: { display: true, text: 'Turn # (1 sec/turn)' } },
        y: { title: { display: true, text: 'Å' }, beginAtZero: true },
      },
    },
  });
}

export function updateR2RErrorChart(waferIndices, rmsErrors, profileRanges) {
  const c = charts.errorR2R;
  if (!c) return;
  c.data.labels = waferIndices;
  c.data.datasets[0].data = rmsErrors;
  c.data.datasets[1].data = profileRanges;
  // Target range line at 50 Å
  c.data.datasets[2].data = waferIndices.map(() => 50);
  c.update();
}

export function updateInRunErrorChart(turnSnapshots) {
  const c = charts.errorInRun;
  if (!c) return;
  c.data.labels = turnSnapshots.map(t => t.turn);
  c.data.datasets[0].data = turnSnapshots.map(t => t.rms_error);
  c.data.datasets[1].data = turnSnapshots.map(t => t.profile_range);
  c.update();
}

// ---- Pressure vs. Time Chart (all 11 zones over turns) ----

const ZONE_LABELS_SHORT = [
  'Z1', 'Z2', 'Z3', 'Z4', 'Z5', 'Z6', 'Z7', 'Z8', 'Z9', 'Z10', 'RR'
];

export function createPressureTimeChart(ctx) {
  charts.pressureTime = new Chart(ctx, {
    type: 'line',
    data: { labels: [], datasets: [] },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      animation: { duration: 0 },
      plugins: {
        title: {
          display: true,
          text: 'Pressure Vector vs. Time (selected wafer)',
          color: COLORS.blue,
          font: { size: 14 },
        },
        legend: {
          position: 'right',
          labels: { boxWidth: 12, font: { size: 10 }, padding: 4 },
        },
      },
      scales: {
        x: {
          title: { display: true, text: 'Time (sec)', font: { size: 12 } },
          ticks: { maxTicksLimit: 20 },
        },
        y: {
          title: { display: true, text: 'Pressure (psi)', font: { size: 12 } },
          min: 0,
          max: 8,
        },
      },
    },
  });
}

export function updatePressureTimeChart(turnSnapshots) {
  const c = charts.pressureTime;
  if (!c || !turnSnapshots || turnSnapshots.length === 0) return;

  const nZones = turnSnapshots[0].pressure.length;
  const timeLabels = turnSnapshots.map(t =>
    t.time_sec !== undefined ? t.time_sec.toFixed(0) : t.turn.toString()
  );

  const datasets = [];
  for (let i = 0; i < nZones; i++) {
    datasets.push({
      label: ZONE_LABELS_SHORT[i] || `Z${i + 1}`,
      data: turnSnapshots.map(t => t.pressure[i]),
      borderColor: ZONE_COLORS[i % ZONE_COLORS.length],
      backgroundColor: 'transparent',
      borderWidth: i === nZones - 1 ? 2.5 : 1.5,
      borderDash: i === nZones - 1 ? [6, 3] : [],
      tension: 0.2,
      pointRadius: 0,
    });
  }

  c.data.labels = timeLabels;
  c.data.datasets = datasets;
  c.update();
}

// ---- SVD Charts ----
export function createSvdCharts(ctxValues, ctxEnergy, ctxModes) {
  charts.svdValues = new Chart(ctxValues, {
    type: 'bar',
    data: { labels: [], datasets: [{ label: 'Singular Values', data: [], backgroundColor: [] }] },
    options: {
      responsive: true, maintainAspectRatio: false,
      plugins: { title: { display: true, text: 'Singular Values of G₀ (Å/(psi·turn))', color: COLORS.blue } },
      scales: { y: { type: 'logarithmic', title: { display: true, text: 'σ (log scale)' } } },
    },
  });

  charts.svdEnergy = new Chart(ctxEnergy, {
    type: 'line',
    data: { labels: [], datasets: [{ label: 'Cumulative Energy', borderColor: COLORS.teal, backgroundColor: COLORS.softTeal, fill: true, tension: 0.3, pointRadius: 4, data: [] }] },
    options: {
      responsive: true, maintainAspectRatio: false,
      plugins: { title: { display: true, text: 'Energy Capture vs. Modes', color: COLORS.teal } },
      scales: { x: { title: { display: true, text: 'Mode #' } }, y: { min: 0, max: 1.05, title: { display: true, text: 'Fraction' } } },
    },
  });

  charts.svdModes = new Chart(ctxModes, {
    type: 'line',
    data: { labels: [], datasets: [] },
    options: {
      responsive: true, maintainAspectRatio: false,
      plugins: { title: { display: true, text: 'Profile-Space Mode Shapes (-150 to +150 mm)', color: COLORS.blue } },
      scales: { x: { title: { display: true, text: 'Radius (mm)' } }, y: { title: { display: true, text: 'Mode Amplitude' } } },
    },
  });
}

export function updateSvdCharts(svdInfo, radialPos, rc) {
  const cv = charts.svdValues;
  if (cv) {
    cv.data.labels = svdInfo.singular_values.map((_, i) => `σ${i + 1}`);
    cv.data.datasets[0].data = svdInfo.singular_values;
    cv.data.datasets[0].backgroundColor = svdInfo.singular_values.map((_, i) => i < rc ? COLORS.teal : '#ccc');
    cv.update();
  }

  const ce = charts.svdEnergy;
  if (ce) {
    ce.data.labels = svdInfo.energy_ratios.map((_, i) => `${i + 1}`);
    ce.data.datasets[0].data = svdInfo.energy_ratios;
    ce.update();
  }

  const cm = charts.svdModes;
  if (cm) {
    // Mirror to -150..+150
    const labels = radialPos.map(r => (-r).toFixed(1)).reverse()
      .concat(radialPos.slice(1).map(r => r.toFixed(1)));
    const mirrorMode = (mode) => [...mode].reverse().concat(mode.slice(1));

    cm.data.labels = labels;
    cm.data.datasets = svdInfo.u_modes.slice(0, Math.min(rc, 6)).map((mode, i) => ({
      label: `Mode ${i + 1}`,
      borderColor: MODE_COLORS[i % MODE_COLORS.length],
      backgroundColor: 'transparent',
      tension: 0.3,
      pointRadius: 1,
      data: mirrorMode(mode),
    }));
    cm.update();
  }
}

// ---- Saturation Chart ----
export function createSaturationChart(ctx) {
  charts.saturation = new Chart(ctx, {
    type: 'bar',
    data: { labels: [], datasets: [{ label: 'Saturation Events', data: [], backgroundColor: COLORS.orange }] },
    options: {
      responsive: true, maintainAspectRatio: false,
      plugins: { title: { display: true, text: 'Actuator Saturation per Wafer', color: COLORS.blue } },
      scales: { x: { title: { display: true, text: 'Wafer #' } }, y: { beginAtZero: true, title: { display: true, text: 'Count' } } },
    },
  });
}

export function updateSaturationChart(waferIndices, satCounts) {
  const c = charts.saturation;
  if (!c) return;
  c.data.labels = waferIndices;
  c.data.datasets[0].data = satCounts;
  c.update();
}

// ---- Zone Test Charts ----

export function createZoneTestChart(ctx) {
  charts.zoneTest = new Chart(ctx, {
    type: 'line',
    data: { labels: [], datasets: [] },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      animation: { duration: 0 },
      plugins: {
        title: {
          display: true,
          text: 'Removal Profile from Custom Pressure Vector (Å/turn)',
          color: COLORS.blue, font: { size: 14 },
        },
        legend: { display: true, position: 'top' },
      },
      scales: {
        x: { title: { display: true, text: 'Radial Position (mm)' }, ticks: { maxTicksLimit: 13 } },
        y: { title: { display: true, text: 'Removal Rate (Å/turn)' }, beginAtZero: true },
      },
    },
  });
}

export function updateZoneTestChart(radialPos, totalRemoval, perZoneRemovals, zoneLabels, yMax) {
  const c = charts.zoneTest;
  if (!c) return;

  const labels = radialPos.map(r => (-r).toFixed(1)).reverse()
    .concat(radialPos.slice(1).map(r => r.toFixed(1)));
  const mirror = (arr) => [...arr].reverse().concat(arr.slice(1));

  const datasets = [];

  // Total removal (thick black)
  datasets.push({
    label: 'Total Removal',
    data: mirror(totalRemoval),
    borderColor: '#000',
    backgroundColor: 'rgba(0,0,0,0.05)',
    borderWidth: 3,
    fill: true,
    tension: 0.3,
    pointRadius: 0,
  });

  // Per-zone contributions (thin colored, only if non-zero)
  for (let i = 0; i < perZoneRemovals.length; i++) {
    const maxVal = Math.max(...perZoneRemovals[i].map(Math.abs));
    if (maxVal < 0.001) continue; // skip zero zones
    datasets.push({
      label: zoneLabels[i] || `Zone ${i + 1}`,
      data: mirror(perZoneRemovals[i]),
      borderColor: ZONE_COLORS[i % ZONE_COLORS.length],
      backgroundColor: 'transparent',
      borderWidth: 1.5,
      borderDash: i === perZoneRemovals.length - 1 ? [6, 3] : [],
      tension: 0.3,
      pointRadius: 0,
    });
  }

  c.data.labels = labels;
  c.data.datasets = datasets;
  if (yMax != null && yMax > 0) {
    c.options.scales.y.min = -yMax * 0.15;
    c.options.scales.y.max = yMax;
    c.options.scales.y.beginAtZero = false;
  }
  c.update();
}

// Solo sweep: each zone alone at nominal pressure
export function createZoneSoloChart(ctx) {
  charts.zoneSolo = new Chart(ctx, {
    type: 'line',
    data: { labels: [], datasets: [] },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      plugins: {
        title: {
          display: true,
          text: 'Solo Sweep: Each Zone at 3.5 psi Alone (Å/turn)',
          color: COLORS.blue, font: { size: 14 },
        },
        legend: {
          position: 'right',
          labels: { boxWidth: 12, font: { size: 10 }, padding: 4 },
        },
      },
      scales: {
        x: { title: { display: true, text: 'Radial Position (mm)' }, ticks: { maxTicksLimit: 13 } },
        y: { title: { display: true, text: 'Removal Rate (Å/turn)' } },
      },
    },
  });
}

export function updateZoneSoloChart(radialPos, g0, zoneLabels, nominalPressure) {
  const c = charts.zoneSolo;
  if (!c) return;

  const labels = radialPos.map(r => (-r).toFixed(1)).reverse()
    .concat(radialPos.slice(1).map(r => r.toFixed(1)));
  const mirror = (arr) => [...arr].reverse().concat(arr.slice(1));

  const nZones = g0[0].length;
  const datasets = [];

  for (let i = 0; i < nZones; i++) {
    // Removal from zone i alone at nominal pressure
    const removal = g0.map(row => row[i] * nominalPressure);
    datasets.push({
      label: zoneLabels[i] || `Zone ${i + 1}`,
      data: mirror(removal),
      borderColor: ZONE_COLORS[i % ZONE_COLORS.length],
      backgroundColor: 'transparent',
      borderWidth: i === nZones - 1 ? 2.5 : 1.5,
      borderDash: i === nZones - 1 ? [6, 3] : [],
      tension: 0.3,
      pointRadius: 0,
    });
  }

  // Also plot total at nominal (all zones at 3.5 psi)
  const totalRemoval = g0.map(row => {
    let sum = 0;
    for (let i = 0; i < nZones; i++) sum += row[i] * nominalPressure;
    return sum;
  });
  datasets.push({
    label: 'Total (all zones)',
    data: mirror(totalRemoval),
    borderColor: '#000',
    backgroundColor: 'rgba(0,0,0,0.05)',
    borderWidth: 3,
    fill: true,
    tension: 0.3,
    pointRadius: 0,
  });

  c.data.labels = labels;
  c.data.datasets = datasets;
  c.update();
}
