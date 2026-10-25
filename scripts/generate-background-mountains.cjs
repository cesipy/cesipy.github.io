#!/usr/bin/env node
'use strict';

const fs = require('node:fs/promises');
const path = require('node:path');
const crypto = require('node:crypto');

const ROOT = path.resolve(__dirname, '..');
const RENDERER_PATH = path.join(ROOT, 'mountain-terrain.js');
const OUTPUT_DIR = path.join(ROOT, 'assets/mountains/background');
const DEFAULT_CONTACT_SHEET = '/private/tmp/mountain-background-contact.png';
const DEFAULT_CSS_CONTACT_SHEET = '/private/tmp/mountain-background-contact-css-opacity-20.png';
const SOURCE_WIDTH = 2400;
const SOURCE_HEIGHT = 800;
const OUTPUT_WIDTH = 720;
const OUTPUT_HEIGHT = 240;

const DEFAULTS = {
  midRidge: 'same',
  farRidge: 'same',
  paper: 'transparent',
  v: '4'
};

const SPECS = [
  { filename: 'ridge-01.png', state: { seed: 'ridge-a-001', ridge: 'alpine', layers: 1, graphite: 99, shading: 48, sharpness: 72, bumpiness: 42, entropy: 45, creativity: 8, spacing: 48 } },
  { filename: 'ridge-02.png', state: { seed: 'ridge-a-011', ridge: 'alpine', layers: 1, graphite: 97, shading: 52, sharpness: 80, bumpiness: 48, entropy: 45, creativity: 12, spacing: 44 } },
  { filename: 'ridge-03.png', state: { seed: 'ridge-a-013', ridge: 'alpine', layers: 1, graphite: 100, shading: 44, sharpness: 78, bumpiness: 55, entropy: 45, creativity: 10, spacing: 50 } },
  { filename: 'ridge-04.png', state: { seed: 'ridge-a-020', ridge: 'alpine', layers: 1, graphite: 96, shading: 47, sharpness: 64, bumpiness: 45, entropy: 45, creativity: 15, spacing: 52 } },
  { filename: 'ridge-05.png', state: { seed: 'ridge-a-021', ridge: 'alpine', midRidge: 'massif', layers: 2, graphite: 100, shading: 58, sharpness: 75, bumpiness: 54, entropy: 45, creativity: 20, spacing: 42 } },
  { filename: 'ridge-06.png', state: { seed: 'ridge-m-001', ridge: 'massif', layers: 1, graphite: 100, shading: 61, sharpness: 62, bumpiness: 48, entropy: 45, creativity: 12, spacing: 56 } },
  { filename: 'ridge-07.png', state: { seed: 'ridge-m-002', ridge: 'massif', layers: 1, graphite: 98, shading: 55, sharpness: 68, bumpiness: 42, entropy: 45, creativity: 22, spacing: 44 } },
  { filename: 'ridge-08.png', state: { seed: 'ridge-m-010', ridge: 'massif', midRidge: 'alpine', layers: 2, graphite: 97, shading: 64, sharpness: 70, bumpiness: 50, entropy: 45, creativity: 18, spacing: 58 } },
  { filename: 'ridge-09.png', state: { seed: 'ridge-m-015', ridge: 'massif', layers: 1, graphite: 99, shading: 41, sharpness: 84, bumpiness: 38, entropy: 45, creativity: 5, spacing: 42 } },
  { filename: 'ridge-10.png', state: { seed: 'ridge-a-025', ridge: 'alpine', layers: 1, graphite: 97, shading: 65, sharpness: 55, bumpiness: 60, entropy: 45, creativity: 25, spacing: 46 } },
  { filename: 'ridge-11.png', state: { seed: 'ridge-j-003', ridge: 'jagged', layers: 1, graphite: 100, shading: 48, sharpness: 80, bumpiness: 52, entropy: 45, creativity: 18, spacing: 52 } },
  { filename: 'ridge-12.png', state: { seed: 'ridge-v-001', ridge: 'valley', layers: 1, graphite: 96, shading: 55, sharpness: 70, bumpiness: 50, entropy: 45, creativity: 18, spacing: 50 } }
].map(({ filename, state }) => ({ filename, state: { ...DEFAULTS, ...state, filename } }));

function parseArgs(argv) {
  const options = {
    port: Number(process.env.CDP_PORT || 9222),
    alphaGamma: 0.6,
    contactSheet: DEFAULT_CONTACT_SHEET,
    cssContactSheet: DEFAULT_CSS_CONTACT_SHEET
  };
  for (let index = 0; index < argv.length; index += 1) {
    const argument = argv[index];
    const [key, inlineValue] = argument.split('=', 2);
    if (key === '--no-contact-sheet') {
      options.contactSheet = null;
      options.cssContactSheet = null;
      continue;
    }
    if (key === '--help' || key === '-h') {
      console.log('Requires an isolated Chrome instance listening on a CDP port. The script creates and closes its own about:blank target.');
      console.log('Example: node scripts/generate-background-mountains.cjs --port 9222');
      console.log('Options: --port 9222, --alpha-gamma 0.6, --contact-sheet PATH, --css-contact-sheet PATH, --no-contact-sheet');
      process.exit(0);
    }
    const value = inlineValue === undefined ? argv[++index] : inlineValue;
    if (value === undefined || value.startsWith('--')) throw new Error(`Missing value for ${key}`);
    if (key === '--port') options.port = Number(value);
    else if (key === '--alpha-gamma') options.alphaGamma = Number(value);
    else if (key === '--contact-sheet') options.contactSheet = value;
    else if (key === '--css-contact-sheet') options.cssContactSheet = value;
    else {
      throw new Error(`Unknown option: ${argument}`);
    }
  }
  if (!Number.isInteger(options.port) || options.port < 1 || options.port > 65535) {
    throw new Error(`Invalid CDP port: ${options.port}`);
  }
  if (!Number.isFinite(options.alphaGamma) || options.alphaGamma <= 0 || options.alphaGamma > 4) {
    throw new Error(`Invalid alpha gamma: ${options.alphaGamma}; expected a value in (0, 4]`);
  }
  if (options.contactSheet) options.contactSheet = path.resolve(options.contactSheet);
  if (options.cssContactSheet) options.cssContactSheet = path.resolve(options.cssContactSheet);
  return options;
}

class CdpConnection {
  constructor(socket) {
    this.socket = socket;
    this.nextId = 0;
    this.pending = new Map();
    socket.addEventListener('message', event => {
      const message = JSON.parse(event.data);
      if (!message.id || !this.pending.has(message.id)) return;
      const request = this.pending.get(message.id);
      this.pending.delete(message.id);
      if (message.error) request.reject(new Error(message.error.message));
      else request.resolve(message.result);
    });
    socket.addEventListener('close', () => {
      for (const request of this.pending.values()) request.reject(new Error('CDP connection closed'));
      this.pending.clear();
    });
  }

  call(method, params = {}, sessionId) {
    const id = ++this.nextId;
    const request = { id, method, params };
    if (sessionId) request.sessionId = sessionId;
    return new Promise((resolve, reject) => {
      this.pending.set(id, { resolve, reject });
      this.socket.send(JSON.stringify(request));
    });
  }

  async evaluate(expression, sessionId) {
    const response = await this.call('Runtime.evaluate', {
      expression,
      returnByValue: true,
      awaitPromise: true
    }, sessionId);
    if (response.exceptionDetails) {
      throw new Error(response.exceptionDetails.exception?.description || response.exceptionDetails.text);
    }
    return response.result.value;
  }
}

function makeRenderExpression(rendererSource, alphaGamma) {
  const serializedSpecs = JSON.stringify(SPECS);
  return `(() => {
    ${rendererSource}
    if (!window.MountainTerrain || typeof window.MountainTerrain.draw !== 'function') {
      throw new Error('MountainTerrain.draw was not installed');
    }
    const specs = ${serializedSpecs};
    const gamma = ${alphaGamma};
    const assets = [];
    const makeContactSheet = () => {
      const canvas = document.createElement('canvas');
      canvas.width = 960;
      canvas.height = 360;
      const context = canvas.getContext('2d', { alpha: false });
      context.fillStyle = '#ffffff';
      context.fillRect(0, 0, canvas.width, canvas.height);
      context.textAlign = 'center';
      context.textBaseline = 'middle';
      return { canvas, context };
    };
    const nativeContact = makeContactSheet();
    const cssContact = makeContactSheet();

    for (let index = 0; index < specs.length; index += 1) {
      const spec = specs[index];
      const source = document.createElement('canvas');
      source.width = ${SOURCE_WIDTH};
      source.height = ${SOURCE_HEIGHT};
      const drawResult = window.MountainTerrain.draw(source, spec.state);
      const outlineSource = document.createElement('canvas');
      outlineSource.width = ${SOURCE_WIDTH};
      outlineSource.height = ${SOURCE_HEIGHT};
      const outlineState = { ...spec.state, shading: 0 };
      const originalStroke = CanvasRenderingContext2D.prototype.stroke;
      const outlineWidthMultiplier = 4;
      let outlineDrawResult;
      try {
        CanvasRenderingContext2D.prototype.stroke = function (...args) {
          const width = this.lineWidth;
          this.lineWidth = width * outlineWidthMultiplier;
          try { return originalStroke.apply(this, args); }
          finally { this.lineWidth = width; }
        };
        outlineDrawResult = window.MountainTerrain.draw(outlineSource, outlineState);
      } finally {
        CanvasRenderingContext2D.prototype.stroke = originalStroke;
      }
      source.getContext('2d').drawImage(outlineSource, 0, 0);
      const output = document.createElement('canvas');
      output.width = ${OUTPUT_WIDTH};
      output.height = ${OUTPUT_HEIGHT};
      const outputContext = output.getContext('2d', { alpha: true, willReadFrequently: gamma !== 1 });
      outputContext.imageSmoothingEnabled = true;
      outputContext.imageSmoothingQuality = 'high';
      outputContext.drawImage(source, 0, 0, ${OUTPUT_WIDTH}, ${OUTPUT_HEIGHT});

      let alphaStats = { nonTransparentPixels: 0, meanAlpha: 0, maxAlpha: 0 };
      if (gamma !== 1) {
        const imageData = outputContext.getImageData(0, 0, output.width, output.height);
        let alphaTotal = 0;
        let maxAlpha = 0;
        let nonTransparentPixels = 0;
        for (let pixel = 3; pixel < imageData.data.length; pixel += 4) {
          const alpha = imageData.data[pixel];
          if (alpha === 0) continue;
          const corrected = Math.min(255, Math.round(255 * Math.pow(alpha / 255, gamma)));
          imageData.data[pixel] = corrected;
          alphaTotal += corrected;
          maxAlpha = Math.max(maxAlpha, corrected);
          nonTransparentPixels += 1;
        }
        outputContext.putImageData(imageData, 0, 0);
        alphaStats = {
          nonTransparentPixels,
          meanAlpha: nonTransparentPixels ? Math.round(alphaTotal / nonTransparentPixels) : 0,
          maxAlpha
        };
      } else {
        const pixels = outputContext.getImageData(0, 0, output.width, output.height).data;
        let alphaTotal = 0;
        let maxAlpha = 0;
        let nonTransparentPixels = 0;
        for (let pixel = 3; pixel < pixels.length; pixel += 4) {
          const alpha = pixels[pixel];
          if (alpha === 0) continue;
          alphaTotal += alpha;
          maxAlpha = Math.max(maxAlpha, alpha);
          nonTransparentPixels += 1;
        }
        alphaStats = {
          nonTransparentPixels,
          meanAlpha: nonTransparentPixels ? Math.round(alphaTotal / nonTransparentPixels) : 0,
          maxAlpha
        };
      }

      const contactX = (index % 4) * 240;
      const contactY = Math.floor(index / 4) * 120;
      for (const sheet of [nativeContact, cssContact]) {
        const context = sheet.context;
        context.save();
        if (sheet === cssContact) context.globalAlpha = 0.20;
        context.drawImage(output, contactX + 60, contactY + 12, 120, 40);
        context.restore();
        context.fillStyle = '#27313a';
        context.font = '12px sans-serif';
        context.fillText(spec.filename.replace('.png', '') + ' · ' + spec.state.ridge + ' · ' + drawResult.layers.length + 'L', contactX + 120, contactY + 69);
        context.fillStyle = '#69737c';
        context.font = '10px sans-serif';
        context.fillText('graphite ' + spec.state.graphite + ' · shading ' + spec.state.shading, contactX + 120, contactY + 86);
      }

      assets.push({
        filename: spec.filename,
        state: spec.state,
        drawResult,
        outlineDrawResult,
        alphaStats,
        data: output.toDataURL('image/png').split(',')[1]
      });
    }
    return {
      assets,
      contactSheet: nativeContact.canvas.toDataURL('image/png').split(',')[1],
      cssContactSheet: cssContact.canvas.toDataURL('image/png').split(',')[1]
    };
  })()`;
}

async function main() {
  const options = parseArgs(process.argv.slice(2));
  const rendererSource = await fs.readFile(RENDERER_PATH, 'utf8');
  const rendererHash = crypto.createHash('sha256').update(rendererSource).digest('hex');
  const versionUrl = `http://127.0.0.1:${options.port}/json/version`;
  const versionInfo = await fetch(versionUrl).then(response => {
    if (!response.ok) throw new Error(`CDP endpoint returned HTTP ${response.status}`);
    return response.json();
  });
  if (!versionInfo.webSocketDebuggerUrl) throw new Error(`No browser WebSocket URL from ${versionUrl}`);

  const socket = new WebSocket(versionInfo.webSocketDebuggerUrl);
  await new Promise((resolve, reject) => {
    socket.addEventListener('open', resolve, { once: true });
    socket.addEventListener('error', reject, { once: true });
  });
  const cdp = new CdpConnection(socket);
  let targetId;
  try {
    const target = await cdp.call('Target.createTarget', { url: 'about:blank' });
    targetId = target.targetId;
    const attached = await cdp.call('Target.attachToTarget', { targetId, flatten: true });
    const sessionId = attached.sessionId;
    await cdp.call('Runtime.enable', {}, sessionId);

    const rendered = await cdp.evaluate(makeRenderExpression(rendererSource, options.alphaGamma), sessionId);
    if (!rendered || rendered.assets.length !== SPECS.length) {
      throw new Error(`Expected ${SPECS.length} rendered assets`);
    }
    await fs.mkdir(OUTPUT_DIR, { recursive: true });

    const assets = [];
    for (const item of rendered.assets) {
      const bytes = Buffer.from(item.data, 'base64');
      await fs.writeFile(path.join(OUTPUT_DIR, item.filename), bytes);
      assets.push({
        filename: item.filename,
        state: item.state,
        drawResult: item.drawResult,
        outlineDrawResult: item.outlineDrawResult,
        alphaStats: item.alphaStats,
        bytes: bytes.length,
        sha256: crypto.createHash('sha256').update(bytes).digest('hex')
      });
    }

    if (options.contactSheet) {
      await fs.mkdir(path.dirname(options.contactSheet), { recursive: true });
      await fs.writeFile(options.contactSheet, Buffer.from(rendered.contactSheet, 'base64'));
    }
    if (options.cssContactSheet) {
      await fs.mkdir(path.dirname(options.cssContactSheet), { recursive: true });
      await fs.writeFile(options.cssContactSheet, Buffer.from(rendered.cssContactSheet, 'base64'));
    }

    const manifest = {
      generatedBy: 'scripts/generate-background-mountains.cjs',
      renderer: {
        file: 'mountain-terrain.js',
        api: 'MountainTerrain.draw(canvas, state)',
        version: 4,
        sha256: rendererHash
      },
      rendering: {
        sourceCanvas: { width: SOURCE_WIDTH, height: SOURCE_HEIGHT },
        output: { width: OUTPUT_WIDTH, height: OUTPUT_HEIGHT, format: 'PNG' },
        resampling: 'Canvas 2D drawImage, imageSmoothingEnabled=true, imageSmoothingQuality=high',
        crop: 'none; full renderer canvas retained',
        paper: 'transparent',
        outlinePass: {
          method: 'second transparent MountainTerrain.draw pass with identical state except shading=0, composited over the shaded source before downsampling; a temporary CanvasRenderingContext2D.stroke wrapper scales lineWidth and is restored in finally',
          strokeWidthMultiplier: 4,
          transparentBackground: true
        },
        alphaPostprocessing: options.alphaGamma === 1
          ? { method: 'none', preservesTransparentPixels: true }
          : {
              method: 'alpha-only gamma curve: alpha = round(255 * (alpha / 255) ^ gamma)',
              gamma: options.alphaGamma,
              preservesTransparentPixels: true
            }
      },
      assets
    };
    await fs.writeFile(path.join(OUTPUT_DIR, 'manifest.json'), `${JSON.stringify(manifest, null, 2)}\n`);

    console.log(JSON.stringify({
      count: assets.length,
      port: options.port,
      alphaGamma: options.alphaGamma,
      contactSheet: options.contactSheet,
      cssContactSheet: options.cssContactSheet,
      items: assets.map(({ filename, drawResult, alphaStats, bytes }) => ({
        filename,
        style: drawResult.style,
        peaks: drawResult.peakCount,
        layers: drawResult.layers.length,
        meanAlpha: alphaStats.meanAlpha,
        bytes
      }))
    }, null, 2));
  } finally {
    if (targetId) {
      try { await cdp.call('Target.closeTarget', { targetId }); } catch {}
    }
    socket.close();
  }
}

main().catch(error => {
  console.error(error.stack || error.message);
  process.exitCode = 1;
});
