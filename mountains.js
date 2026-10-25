(() => {
  'use strict';

  const $ = id => document.getElementById(id);
  const canvas = $('mountain-canvas');
  const status = $('studio-status');
  const fields = {
    ridge: $('ridge-select'), midRidge: $('mid-ridge-select'), farRidge: $('far-ridge-select'),
    layers: $('layers'), graphite: $('graphite'), shading: $('shading'),
    spacing: $('spacing'), sharpness: $('sharpness'), bumpiness: $('bumpiness'),
    entropy: $('entropy'), creativity: $('creativity'), paper: $('paper-select'), seed: $('seed-input')
  };
  const terrainStyles = ['alpine', 'jagged', 'rolling', 'massif', 'valley', 'plateau', 'spires', 'asymmetric'];
  const styles = new Set(['auto', ...terrainStyles]);
  const layerStyles = new Set(['same', 'auto', ...terrainStyles]);
  const styleNames = {
    alpine: 'alpine ridges', jagged: 'sharp peaks', rolling: 'gentle mountains',
    massif: 'solitary massif', valley: 'open valley', plateau: 'flat-topped cliffs',
    spires: 'broken spires', asymmetric: 'sweeping ridge'
  };
  const defaults = {
    ridge: 'auto', midRidge: 'same', farRidge: 'same', layers: 2, graphite: 65, spacing: 50,
    sharpness: 50, bumpiness: 50, entropy: 50, creativity: 0, shading: 50, character: 50,
    paper: 'transparent', v: '4'
  };
  const characterKeys = ['sharpness', 'bumpiness', 'entropy', 'creativity', 'spacing', 'graphite', 'shading'];
  const history = [];
  let state, initialState, rangeStart = null, frame = null, ready = false, versionNotice = '';

  function newSeed() {
    const values = new Uint32Array(3);
    window.crypto.getRandomValues(values);
    return Array.from(values, value => value.toString(36)).join('-');
  }

  function cleanSeed(value, fallback) {
    return Array.from(String(value || '').trim()).slice(0, 32).join('') || fallback;
  }

  function bounded(value, min, max, fallback) {
    if (value === null || String(value).trim() === '') return fallback;
    const number = Number(value);
    return Number.isFinite(number) ? Math.max(min, Math.min(max, Math.round(number))) : fallback;
  }

  function characterSettings(character) {
    const c = bounded(character, 0, 100, 50);
    return {
      sharpness: Math.round(15 + .7 * c),
      bumpiness: Math.round(10 + .85 * c),
      entropy: Math.round(20 + .7 * c),
      creativity: Math.round(.85 * c),
      spacing: Math.round(25 + .6 * c),
      graphite: Math.round(50 + .4 * c),
      shading: Math.round(40 + .6 * c)
    };
  }

  function deriveCharacter(settings) {
    const values = [
      [(settings.sharpness - 15) / .7, 1],
      [(settings.bumpiness - 10) / .85, 1],
      [(settings.entropy - 20) / .7, 1],
      [settings.creativity / .85, 1],
      [(settings.spacing - 25) / .6, .5],
      [(settings.graphite - 50) / .4, .5],
      [(settings.shading - 40) / .6, 1]
    ];
    const weighted = values.reduce((sum, [value, weight]) => sum + value * weight, 0);
    const weightTotal = values.reduce((sum, [, weight]) => sum + weight, 0);
    return Math.round(Math.max(0, Math.min(100, weighted / weightTotal)));
  }

  function isCharacterCustom() {
    const mapped = characterSettings(state.character);
    return characterKeys.some(key => state[key] !== mapped[key]);
  }

  function readHash(fallback) {
    const params = new URLSearchParams(location.hash.slice(1));
    const hasHash = location.hash.slice(1) !== '';
    let version;
    if (params.has('v')) {
      const requestedVersion = params.get('v');
      if (requestedVersion === '2' || requestedVersion === '3' || requestedVersion === '4') {
        version = requestedVersion;
        versionNotice = '';
      } else {
        version = '4';
        versionNotice = 'Renderer version “' + (requestedVersion || 'empty') + '” is unsupported; showing version 4.';
      }
    } else {
      version = params.toString() ? '2' : '4';
      versionNotice = '';
    }
    const parsed = {
      v: version,
      seed: cleanSeed(params.get('seed'), fallback.seed),
      ridge: styles.has(params.get('ridge')) ? params.get('ridge') : fallback.ridge,
      midRidge: layerStyles.has(params.get('midRidge')) ? params.get('midRidge') : defaults.midRidge,
      farRidge: layerStyles.has(params.get('farRidge')) ? params.get('farRidge') : defaults.farRidge,
      layers: bounded(params.get('layers'), 1, 3, fallback.layers),
      graphite: bounded(params.get('graphite'), 25, 100, fallback.graphite),
      shading: bounded(params.get('shading'), 0, 100, 50),
      spacing: bounded(params.get('spacing'), 0, 100, fallback.spacing),
      sharpness: bounded(params.get('sharpness'), 0, 100, defaults.sharpness),
      bumpiness: bounded(params.get('bumpiness'), 0, 100, defaults.bumpiness),
      entropy: bounded(params.get('entropy'), 0, 100, defaults.entropy),
      creativity: bounded(params.get('creativity'), 0, 100, bounded(params.get('wildness'), 0, 100, defaults.creativity)),
      paper: ['transparent', 'white'].includes(params.get('paper')) ? params.get('paper') : fallback.paper
    };
    parsed.character = params.has('character') ? bounded(params.get('character'), 0, 100, deriveCharacter(parsed)) : deriveCharacter(parsed);
    if (!hasHash) {
      return { ...parsed, ...characterSettings(50), character: 50, v: '4' };
    }
    return parsed;
  }

  function equal(a, b) {
    return Object.keys(defaults).concat('seed').every(key => a[key] === b[key]);
  }

  function updateVersionNotice() {
    const notice = $('renderer-version-notice');
    const message = $('renderer-version-message');
    const updateButton = $('update-renderer');
    if (state.v === '2') {
      message.textContent = 'This saved landscape uses its original drawing engine. Character and shading controls are available after upgrading to version 4. ';
      updateButton.hidden = false;
      notice.hidden = false;
    } else if (state.v === '3') {
      message.textContent = 'This saved landscape uses renderer version 3. Use the updated drawing to switch to version 4. ';
      updateButton.hidden = false;
      notice.hidden = false;
    } else if (versionNotice) {
      message.textContent = versionNotice;
      updateButton.hidden = true;
      notice.hidden = false;
    } else {
      notice.hidden = true;
    }
  }

  function buttons() {
    $('previous-mountains').disabled = history.length === 0;
    $('download-mountains').disabled = !ready;
    $('share-mountains').disabled = !ready;
    canvas.setAttribute('aria-busy', String(!ready));
  }

  function updateCharacterDisplay() {
    const custom = isCharacterCustom();
    $('character-value').textContent = state.character + '%' + (custom ? ' · custom' : '');
    const message = state.v === '2'
      ? 'Character and shading are available after using the updated drawing.'
      : custom
        ? 'custom mix · move character to replace the fine adjustments'
        : 'fine-tune individual settings under advanced';
    if ($('character-state').textContent !== message) $('character-state').textContent = message;
  }

  function syncControls() {
    for (const key of Object.keys(fields)) fields[key].value = String(state[key]);
    fields.midRidge.disabled = state.layers < 2;
    fields.farRidge.disabled = state.layers < 3;
    fields.shading.disabled = state.v === '2';
    $('character').value = String(state.character);
    $('character').disabled = state.v === '2';
    $('layers-value').textContent = state.layers;
    $('graphite-value').textContent = state.graphite + '%';
    $('shading-value').textContent = state.shading + '%';
    $('spacing-value').textContent = state.spacing + '%';
    $('sharpness-value').textContent = state.sharpness + '%';
    $('bumpiness-value').textContent = state.bumpiness + '%';
    $('entropy-value').textContent = state.entropy + '%';
    $('creativity-value').textContent = state.creativity + '%';
    updateCharacterDisplay();
  }

  function remember(previous) {
    if (!history.length || !equal(history[history.length - 1], previous)) history.push({ ...previous });
    if (history.length > 20) history.shift();
    buttons();
  }

  function commitRange() {
    if (rangeStart && !equal(rangeStart, state)) remember(rangeStart);
    rangeStart = null;
  }

  function render() {
    if (frame !== null) cancelAnimationFrame(frame);
    ready = false;
    buttons();
    updateVersionNotice();
    status.textContent = 'drawing the ridges…';
    frame = requestAnimationFrame(() => {
      frame = null;
      try {
        if (!window.MountainTerrain) throw new Error('The drawing engine did not load. Reload this page to try again.');
        const result = window.MountainTerrain.draw(canvas, { ...state });
        canvas.dataset.foreground = result.foregroundFingerprint || '';
        canvas.setAttribute('aria-label', 'Procedural pencil landscape with ' + state.layers + ' mountain ' + (state.layers === 1 ? 'range' : 'ranges'));
        const styleName = style => styleNames[style] || style || 'terrain';
        const layerStyle = layer => {
          if (layer.style) return styleName(layer.style);
          if (layer.depth === 0) return styleName(result.style || state.ridge);
          const requested = layer.depth === 1 ? state.midRidge : state.farRidge;
          if (requested === 'same') return styleName(result.style || state.ridge);
          return requested === 'auto' ? 'independent surprise' : styleName(requested);
        };
        const otherLayers = (result.layers || []).filter(layer => layer.depth !== 0).sort((a, b) => a.depth - b.depth).map(layer => {
          const position = layer.depth === 1 ? (state.layers === 2 ? 'distant' : 'middle') : 'distant';
          return position + ': ' + layerStyle(layer);
        });
        const layerSummary = otherLayers.length ? ' Additional layer styles: ' + otherLayers.join('; ') + '.' : '';
        $('studio-sources').textContent = 'This drawing was built from new terrain profiles across ' + state.layers + ' ' + (state.layers === 1 ? 'range' : 'ranges') + '. Foreground: ' + styleName(result.style || state.ridge) + '.' + layerSummary + ' Every contour and pencil stroke is calculated from seed “' + state.seed + '”. No image files are used.';
        status.textContent = 'ready · seed ' + state.seed;
        ready = true;
      } catch (error) {
        status.textContent = 'Could not draw this landscape. ' + (error.message || 'Please try a new landscape.');
      }
      buttons();
    });
  }

  function readControls() {
    return {
      ...state,
      ridge: styles.has(fields.ridge.value) ? fields.ridge.value : defaults.ridge,
      midRidge: layerStyles.has(fields.midRidge.value) ? fields.midRidge.value : defaults.midRidge,
      farRidge: layerStyles.has(fields.farRidge.value) ? fields.farRidge.value : defaults.farRidge,
      layers: bounded(fields.layers.value, 1, 3, state.layers),
      graphite: bounded(fields.graphite.value, 25, 100, state.graphite),
      shading: bounded(fields.shading.value, 0, 100, state.shading),
      spacing: bounded(fields.spacing.value, 0, 100, state.spacing),
      sharpness: bounded(fields.sharpness.value, 0, 100, state.sharpness),
      bumpiness: bounded(fields.bumpiness.value, 0, 100, state.bumpiness),
      entropy: bounded(fields.entropy.value, 0, 100, state.entropy),
      creativity: bounded(fields.creativity.value, 0, 100, state.creativity),
      paper: fields.paper.value === 'white' ? 'white' : 'transparent'
    };
  }

  function applyCharacter() {
    const character = bounded($('character').value, 0, 100, state.character);
    const next = { ...readControls(), ...characterSettings(character), character };
    if (equal(next, state)) return;
    state = next;
    syncControls();
    render();
  }

  function settingsChanged(record) {
    const next = readControls();
    if (equal(next, state)) return;
    if (record) remember(state);
    state = next;
    $('layers-value').textContent = state.layers;
    $('graphite-value').textContent = state.graphite + '%';
    $('spacing-value').textContent = state.spacing + '%';
    $('sharpness-value').textContent = state.sharpness + '%';
    $('bumpiness-value').textContent = state.bumpiness + '%';
    $('entropy-value').textContent = state.entropy + '%';
    $('creativity-value').textContent = state.creativity + '%';
    $('shading-value').textContent = state.shading + '%';
    updateCharacterDisplay();
    fields.midRidge.disabled = state.layers < 2;
    fields.farRidge.disabled = state.layers < 3;
    render();
  }

  function applySeed() {
    commitRange();
    const next = { ...readControls(), seed: cleanSeed(fields.seed.value, newSeed()) };
    if (!equal(state, next)) remember(state);
    state = next;
    syncControls();
    render();
  }

  function shareURL() {
    const url = new URL(location.href);
    const params = new URLSearchParams({ ...state });
    url.hash = params.toString();
    return url;
  }

  async function share() {
    if (!ready) return;
    const url = shareURL().href;
    try { window.history.pushState(null, '', url); } catch (_) { /* Link works without history. */ }
    try {
      await navigator.clipboard.writeText(url);
      status.textContent = 'landscape link copied.';
    } catch (_) {
      status.textContent = 'Copy this landscape link: ' + url;
    }
  }

  async function download() {
    if (!ready) return;
    const seed = state.seed.replace(/[^a-zA-Z0-9_-]+/g, '-').slice(0, 40) || 'study';
    try {
      const blob = await new Promise((resolve, reject) => canvas.toBlob(result => {
        if (result) resolve(result);
        else reject(new Error('The browser could not create a PNG.'));
      }, 'image/png'));
      const url = URL.createObjectURL(blob);
      const link = document.createElement('a');
      link.href = url;
      link.download = 'mountain-study-' + seed + '.png';
      document.body.append(link);
      link.click();
      link.remove();
      setTimeout(() => URL.revokeObjectURL(url), 10000);
      status.textContent = 'your pencil study is ready to save.';
    } catch (error) {
      status.textContent = 'Download failed. ' + error.message;
    }
  }

  // Optional local photo generation needs the exact current geometry without
  // changing the shared URL or the visible pencil drawing.
  window.MountainStudio = Object.freeze({
    snapshot: () => ready ? { ...state } : null
  });

  try {
    state = readHash({ ...defaults, seed: newSeed() });
    initialState = { ...state };
    syncControls();
    $('studio-controls').disabled = false;
    $('generate-mountains').addEventListener('click', () => {
      commitRange();
      remember(state);
      versionNotice = '';
      state = { ...readControls(), seed: newSeed(), v: '4' };
      syncControls();
      render();
    });
    $('update-renderer').addEventListener('click', () => {
      if (state.v !== '2' && state.v !== '3') return;
      commitRange();
      remember(state);
      versionNotice = '';
      state = { ...state, v: '4', shading: state.v === '2' ? 50 : state.shading };
      syncControls();
      render();
    });
    $('previous-mountains').addEventListener('click', () => {
      commitRange();
      const previous = history.pop();
      if (!previous) return;
      state = previous;
      syncControls();
      render();
    });
    for (const field of [
      fields.layers, fields.graphite, fields.shading, fields.spacing,
      fields.sharpness, fields.bumpiness, fields.entropy, fields.creativity
    ]) {
      field.addEventListener('input', () => {
        if (!rangeStart) rangeStart = { ...state };
        settingsChanged(false);
      });
      field.addEventListener('change', () => {
        if (rangeStart) { settingsChanged(false); commitRange(); }
        else settingsChanged(true);
      });
    }
    $('character').addEventListener('input', () => {
      if (!rangeStart) rangeStart = { ...state };
      applyCharacter();
    });
    $('character').addEventListener('change', () => {
      if (!rangeStart) rangeStart = { ...state };
      applyCharacter();
      commitRange();
    });
    for (const field of [fields.ridge, fields.midRidge, fields.farRidge, fields.paper]) field.addEventListener('change', () => {
      commitRange();
      settingsChanged(true);
    });
    $('apply-seed').addEventListener('click', applySeed);
    fields.seed.addEventListener('keydown', event => {
      if (event.key === 'Enter') { event.preventDefault(); applySeed(); }
    });
    $('download-mountains').addEventListener('click', download);
    $('share-mountains').addEventListener('click', share);
    window.addEventListener('hashchange', () => {
      commitRange();
      const next = readHash({ ...defaults, seed: initialState.seed });
      if (equal(next, state)) {
        updateVersionNotice();
        return;
      }
      remember(state);
      state = next;
      syncControls();
      render();
    });
    render();
  } catch (error) {
    status.textContent = 'The studio could not start. ' + error.message;
  }
})();
