// Parameter panel: read UI values and build SimConfig JSON

export function gatherConfig() {
  return {
    n_wafers: parseInt(document.getElementById('cfg-wafers').value) || 30,
    turns_per_wafer: parseInt(document.getElementById('cfg-turns').value) || 160,
    metrology_delay: parseInt(document.getElementById('cfg-delay').value) || 1,
    rc: parseInt(document.getElementById('cfg-rc').value) || 8,
    enable_inrun: document.getElementById('cfg-inrun').checked,
    enable_r2r: document.getElementById('cfg-r2r').checked,
    enable_wear_drift: document.getElementById('cfg-wear').checked,
    wear_rate: parseFloat(document.getElementById('cfg-wear-rate').value) || 0.02,
    disturbance_amplitude: parseFloat(document.getElementById('cfg-dist').value) || 3.0,
    noise_amplitude: parseFloat(document.getElementById('cfg-noise').value) || 5.0,
    seed: parseInt(document.getElementById('cfg-seed').value) || 42,
    // Record turn detail for every 3rd wafer (balance memory vs animation)
    turn_detail_every_n: 3,
  };
}

export function setupRangeDisplays() {
  const pairs = [
    ['cfg-rc', 'cfg-rc-val'],
    ['cfg-dist', 'cfg-dist-val'],
    ['cfg-noise', 'cfg-noise-val'],
    ['cfg-wear-rate', 'cfg-wear-rate-val'],
    ['wafer-slider', 'wafer-slider-val'],
    ['anim-speed', 'anim-speed-val'],
  ];

  for (const [inputId, displayId] of pairs) {
    const input = document.getElementById(inputId);
    const display = document.getElementById(displayId);
    if (input && display) {
      input.addEventListener('input', () => {
        display.textContent = input.value;
      });
    }
  }
}

export function setWaferSliderMax(max) {
  const slider = document.getElementById('wafer-slider');
  if (slider) {
    slider.max = max;
    slider.value = Math.min(parseInt(slider.value), max);
    document.getElementById('wafer-slider-val').textContent = slider.value;
  }
}

export function getSelectedWafer() {
  return parseInt(document.getElementById('wafer-slider').value) || 0;
}

export function getAnimSpeed() {
  return parseInt(document.getElementById('anim-speed').value) || 50;
}
