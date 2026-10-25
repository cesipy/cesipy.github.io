(() => {
  'use strict';

  const canvas = document.getElementById('mountain-canvas');
  const sketchDownload = document.getElementById('download-mountains');
  const generate = document.getElementById('photo-generate');
  const engine = document.getElementById('photo-engine');
  const engineHint = document.getElementById('photo-engine-hint');
  const prompt = document.getElementById('photo-prompt');
  const status = document.getElementById('photo-status');
  const result = document.getElementById('photo-result');
  const image = document.getElementById('photo-image');
  const caption = document.getElementById('photo-caption');
  const download = document.getElementById('photo-download');
  if (!canvas || !sketchDownload || !generate || !engine || !engineHint || !prompt || !status || !result || !image || !caption || !download) return;

  const engines = {
    klein: {
      url: 'http://127.0.0.1:8767',
      label: 'FLUX.2 Klein',
      hint: 'More photographic interpretation; the exact ridge shapes can change. The quantized model is about 4.6 GB.'
    },
    controlnet: {
      url: 'http://127.0.0.1:8766',
      label: 'line-art ControlNet',
      hint: 'Follows the sketch contours more closely. Its model weights are about 5–6 GB.'
    }
  };
  let busy = false;
  let imageUrl = null;

  function syncReady() {
    generate.disabled = busy || sketchDownload.disabled;
    engine.disabled = busy;
  }

  function selectedEngine() {
    return engines[engine.value] || engines.klein;
  }

  engine.addEventListener('change', () => {
    engineHint.textContent = selectedEngine().hint;
  });

  function lineartFromCurrentLandscape() {
    const current = window.MountainStudio && window.MountainStudio.snapshot();
    if (!current || current.v === '2' || !window.MountainTerrain) return canvas.toDataURL('image/png');
    const control = document.createElement('canvas');
    control.width = canvas.width;
    control.height = canvas.height;
    // The photo model needs the mountain structure. Remove the graphite tone
    // only in this temporary canvas so shaded sketches do not become black blobs.
    window.MountainTerrain.draw(control, { ...current, shading: 0, graphite: 100, paper: 'white' });
    return control.toDataURL('image/png');
  }

  new MutationObserver(syncReady).observe(sketchDownload, {
    attributes: true,
    attributeFilter: ['disabled']
  });
  syncReady();

  async function responseError(response) {
    try {
      const data = await response.json();
      if (typeof data.error === 'string') return data.error;
    } catch (_) { /* The server may return plain text. */ }
    return 'The local model returned HTTP ' + response.status + '.';
  }

  generate.addEventListener('click', async () => {
    if (busy || sketchDownload.disabled) return;
    busy = true;
    syncReady();
    const sketchSeed = document.getElementById('seed-input').value.trim() || 'landscape';
    const choice = selectedEngine();
    status.textContent = 'sending this sketch to the local model… the first run may take a few minutes.';

    try {
      const response = await fetch(choice.url + '/generate', {
        method: 'POST',
        headers: {'Content-Type': 'application/json'},
        body: JSON.stringify({
          image: engine.value === 'controlnet' ? lineartFromCurrentLandscape() : canvas.toDataURL('image/png'),
          prompt: prompt.value.trim().slice(0, 250)
        })
      });
      if (!response.ok) throw new Error(await responseError(response));
      const blob = await response.blob();
      if (!blob.type.startsWith('image/')) throw new Error('The local model did not return an image.');

      if (imageUrl) URL.revokeObjectURL(imageUrl);
      imageUrl = URL.createObjectURL(blob);
      image.src = imageUrl;
      image.alt = 'Realistic landscape generated from pencil sketch seed ' + sketchSeed;
      const photoSeed = response.headers.get('X-Image-Seed');
      caption.textContent = choice.label + ' · pencil seed “' + sketchSeed + '”' + (photoSeed ? ' · photo seed ' + photoSeed : '');
      download.href = imageUrl;
      download.download = 'mountain-photo-' + (sketchSeed.replace(/[^a-zA-Z0-9_-]+/g, '-').slice(0, 32) || 'landscape') + '.png';
      result.hidden = false;
      status.textContent = choice.label + ' landscape ready.';
    } catch (error) {
      status.textContent = error instanceof TypeError
        ? 'Could not reach the ' + choice.label + ' server. Start it and allow this page to access localhost in your browser.'
        : 'Could not make a photo. ' + (error.message || 'Try again.');
    } finally {
      busy = false;
      syncReady();
    }
  });

  window.addEventListener('beforeunload', () => {
    if (imageUrl) URL.revokeObjectURL(imageUrl);
  });
})();
