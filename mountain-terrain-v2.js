// Frozen renderer for v=2 landscape links. Do not change its drawing algorithm.
(() => {
  "use strict";

  const BASE_WIDTH = 2400;
  const BASE_HEIGHT = 800;
  const TERRAIN_CANVASES = new WeakMap();
  const STYLES = ["alpine", "jagged", "rolling", "massif", "valley", "plateau", "spires", "asymmetric"];

  function clamp(value, minimum, maximum) {
    return Math.max(minimum, Math.min(maximum, value));
  }

  function randomSource(seed) {
    let state = 2166136261;
    const text = String(seed);
    for (let index = 0; index < text.length; index += 1) {
      state ^= text.charCodeAt(index);
      state = Math.imul(state, 16777619);
    }

    return function random() {
      state += 0x6D2B79F5;
      let value = state;
      value = Math.imul(value ^ (value >>> 15), value | 1);
      value ^= value + Math.imul(value ^ (value >>> 7), value | 61);
      return ((value ^ (value >>> 14)) >>> 0) / 4294967296;
    };
  }

  function between(random, minimum, maximum) {
    return minimum + (maximum - minimum) * random();
  }

  function chooseStyle(requested, seed) {
    if (STYLES.includes(requested)) return requested;
    const random = randomSource(`${seed}|style`);
    return STYLES[Math.floor(random() * STYLES.length)];
  }

  function smoothNoise(random, step, amplitude) {
    const values = [];
    for (let x = 0; x <= BASE_WIDTH + step; x += step) {
      values.push((random() * 2 - 1) * amplitude);
    }

    return function valueAt(x) {
      const position = x / step;
      const index = Math.min(values.length - 2, Math.floor(position));
      const fraction = clamp(position - index, 0, 1);
      const eased = fraction * fraction * (3 - 2 * fraction);
      return values[index] + (values[index + 1] - values[index]) * eased;
    };
  }

  function interpolateBezier(first, second, third, fourth, amount) {
    const inverse = 1 - amount;
    const inverseSquared = inverse * inverse;
    const amountSquared = amount * amount;
    return {
      x: inverseSquared * inverse * first.x +
        3 * inverseSquared * amount * second.x +
        3 * inverse * amountSquared * third.x +
        amountSquared * amount * fourth.x,
      y: inverseSquared * inverse * first.y +
        3 * inverseSquared * amount * second.y +
        3 * inverse * amountSquared * third.y +
        amountSquared * amount * fourth.y
    };
  }

  function composeTerrain(points, peaks, baseY, seed, depth, creativity) {
    if (creativity <= 0) return;
    const amount = creativity / 100;
    const random = randomSource(seed + '|composition');
    const arrangement = Math.floor(random() * 4);
    const phase = between(random, 0, Math.PI * 2);
    const focus = random() < 0.5 ? 0.28 : 0.72;
    const offset = Math.sin(depth * 1.7 + phase) * 0.045;
    const tau = Math.PI * 2;
    const bell = (position, center, width) => Math.exp(-Math.pow((position - center) / width, 2));
    // Art direction, not distortion: trade uniform spacing for a focal group
    // and breathing room. Positive spacing preserves a coherent silhouette.
    const speed = x => Math.exp(amount * (
      0.65 * Math.sin(x / BASE_WIDTH * tau + phase) +
      0.2 * Math.sin(x / BASE_WIDTH * tau * 2 + offset)
    ));
    const mapping = [{x: points[0].x, y: 0}];
    let distance = 0;
    for (let index = 1; index < points.length; index++) {
      const left = points[index - 1].x, right = points[index].x;
      distance += (right - left) * (speed(left) + speed(right)) / 2;
      mapping.push({x: right, y: distance});
    }
    mapping.forEach(point => { point.y *= BASE_WIDTH / distance; });
    const warpX = x => profileYAt(mapping, clamp(x, 0, BASE_WIDTH));
    peaks.forEach(peak => {
      const left = warpX(peak.center - peak.leftSpan);
      const right = warpX(peak.center + peak.rightSpan);
      peak.center = warpX(peak.center);
      peak.leftSpan = peak.center - left;
      peak.rightSpan = right - peak.center;
      peak.segmentWidth = right - left;
    });
    points.forEach((point, index) => {
      const position = point.x / BASE_WIDTH;
      let composed;
      if (arrangement === 0) {
        // A clear off-center focal summit with quieter shoulders.
        composed = 0.68 + 0.7 * bell(position, focus + offset, 0.2);
      } else if (arrangement === 1) {
        // An open pass between two groups of mountains.
        composed = 1.06 - 0.57 * bell(position, 0.5 + offset, 0.22);
      } else if (arrangement === 2) {
        // Paired clusters separated by a pause in the skyline.
        composed = 0.7 + 0.58 * Math.max(
          bell(position, 0.24 + offset, 0.15), bell(position, 0.74 + offset, 0.18)
        );
      } else {
        // A directional composition that builds gradually to one side.
        composed = 0.62 + 0.72 * (focus > 0.5 ? position : 1 - position);
      }
      const gain = 1 + amount * (composed - 1);
      point.y = baseY - (baseY - point.y) * gain;
      point.x = mapping[index].y;
    });
    // Keep even a tall focal summit inside the paper without flattening it.
    const tallest = Math.max(...points.map(point => baseY - point.y), 1);
    const fit = Math.min(1, 520 / tallest);
    points.forEach(point => {
      point.y = clamp(baseY - (baseY - point.y) * fit, 30, BASE_HEIGHT - 100);
    });
  }

  function makeProfile(seed, style, depth, shape) {
    const source = randomSource(seed + '|profile|' + depth);
    // Keep the random stream fixed as controls move. Entropy changes the spread
    // of the draws rather than selecting an unrelated seed or terrain style.
    const spread = shape.entropy / 50;
    const random = () => {
      const value = source();
      if (spread === 1) return value;
      if (spread < 1) return 0.5 + (value - 0.5) * spread;
      const centered = value - 0.5;
      return clamp(0.5 + Math.sign(centered) * 0.5 * Math.pow(Math.abs(centered) * 2, 1 / (2 * spread - 1)), 0, 1 - Number.EPSILON);
    };
    const bumpScale = Math.pow(shape.bumpiness / 50, 1.6);
    const settings = {
      alpine: { minimum: 3, maximum: 6, low: 175, high: 360, roughness: 1 },
      jagged: { minimum: 5, maximum: 8, low: 160, high: 350, roughness: 1.5 },
      rolling: { minimum: 3, maximum: 5, low: 100, high: 205, roughness: 0.35 },
      massif: { minimum: 1, maximum: 3, low: 280, high: 405, roughness: 0.9 },
      valley: { minimum: 4, maximum: 6, low: 225, high: 360, roughness: 1.05 },
      plateau: { minimum: 2, maximum: 4, low: 200, high: 310, roughness: 0.65 },
      spires: { minimum: 6, maximum: 10, low: 150, high: 375, roughness: 1.2 },
      asymmetric: { minimum: 4, maximum: 7, low: 180, high: 365, roughness: 0.85 }
    }[style];
    const baseY = 565;
    const count = settings.minimum + Math.floor(random() * (settings.maximum - settings.minimum + 1));
    const weights = Array.from({length: count}, () => between(random, 0.6, 1.7));
    const total = weights.reduce((sum, value) => sum + value, 0);
    const hero = Math.floor(random() * count);
    const focus = between(random, BASE_WIDTH * 0.32, BASE_WIDTH * 0.68);
    const risingRight = random() > 0.5;
    const peaks = [];
    const boundaries = [0];
    let cursor = 0;
    for (let index = 0; index < count; index++) {
      const span = BASE_WIDTH * weights[index] / total;
      let center = cursor + span * between(random, 0.28, 0.72);
      let height = between(random, settings.low, settings.high) * (index === hero ? 1.18 : 1);
      if (style === 'massif') center = focus + (center - BASE_WIDTH / 2) * 0.38;
      if (style === 'valley') center = index < count / 2 ? center * 0.56 : BASE_WIDTH - (BASE_WIDTH - center) * 0.56;
      if (style === 'asymmetric') height *= 0.25 + 0.85 * (risingRight ? center / BASE_WIDTH : 1 - center / BASE_WIDTH);
      peaks.push({ center, height, peakY: baseY - height, leftSpan: center - cursor, rightSpan: cursor + span - center, segmentWidth: span });
      cursor += span;
      boundaries.push(cursor);
    }
    if (style === 'massif' || style === 'valley') {
      for (let index = 1; index < count; index++) boundaries[index] = (peaks[index - 1].center + peaks[index].center) / 2;
      peaks.forEach((peak, index) => {
        peak.leftSpan = peak.center - boundaries[index];
        peak.rightSpan = boundaries[index + 1] - peak.center;
        peak.segmentWidth = peak.leftSpan + peak.rightSpan;
      });
    }
    const valleys = boundaries.map((x, index) => {
      if (index === 0 || index === count) return {x, y: baseY + between(random, 20, 55)};
      const height = Math.min(peaks[index - 1].height, peaks[index].height);
      if (style === 'valley' && peaks[index].center - peaks[index - 1].center > BASE_WIDTH * 0.3) return {x, y: baseY + between(random, 5, 40)};
      return {x, y: baseY - height * between(random, style === 'massif' ? 0.68 : 0.25, style === 'massif' ? 0.85 : 0.72)};
    });
    const anchors = [valleys[0]];
    peaks.forEach((peak, index) => {
      const summit = {x: peak.center, y: peak.peakY};
      if (style === 'plateau') {
        anchors.push(
          {x: peak.center - peak.leftSpan * 0.68, y: valleys[index].y - 28},
          {x: peak.center - peak.leftSpan * 0.45, y: peak.peakY + 8},
          {x: peak.center - peak.leftSpan * 0.2, y: peak.peakY + between(random, -9, 9)},
          summit,
          {x: peak.center + peak.rightSpan * 0.44, y: peak.peakY + between(random, -12, 12)},
          {x: peak.center + peak.rightSpan * 0.67, y: valleys[index + 1].y - 22},
          valleys[index + 1]
        );
        return;
      }
      if (style === 'spires') {
        anchors.push(
          {x: peak.center - peak.leftSpan * 0.27, y: peak.peakY + (valleys[index].y - peak.peakY) * 0.65},
          {x: peak.center - peak.leftSpan * 0.07, y: peak.peakY + peak.height * 0.18},
          summit,
          {x: peak.center + peak.rightSpan * 0.11, y: peak.peakY + peak.height * 0.46},
          {x: peak.center + peak.rightSpan * 0.3, y: peak.peakY + (valleys[index + 1].y - peak.peakY) * 0.8},
          valleys[index + 1]
        );
        return;
      }
      for (const [from, to] of [[valleys[index], summit], [summit, valleys[index + 1]]]) {
        const divisions = style === 'rolling' ? 3 : 4;
        for (let step = 1; step < divisions; step++) {
          const t = (step + between(random, -0.22, 0.22)) / divisions;
          const displacement = between(random, -1, 1) * Math.abs(to.y - from.y) * 0.13 * settings.roughness * bumpScale;
          anchors.push({x: from.x + (to.x - from.x) * t, y: from.y + (to.y - from.y) * t + displacement});
        }
        anchors.push(to);
      }
    });
    const coarse = smoothNoise(random, 125, 10 * settings.roughness * bumpScale);
    const middle = smoothNoise(random, 32, 4.8 * settings.roughness * bumpScale);
    const fine = smoothNoise(random, 8, 1.6 * settings.roughness * bumpScale);
    const points = [];
    for (let index = 0; index < anchors.length - 1; index++) {
      const first = anchors[index], second = anchors[index + 1];
      const steps = Math.max(2, Math.ceil((second.x - first.x) / 4));
      for (let step = index === 0 ? 0 : 1; step <= steps; step++) {
        const t = step / steps;
        const x = first.x + (second.x - first.x) * t;
        const blend = style === 'rolling' ? t * t * (3 - 2 * t) : t;
        const y = first.y + (second.y - first.y) * blend;
        points.push({x, y});
      }
    }
    // Round the broad silhouette before adding small crags, so sharpness and
    // bumpiness remain distinct controls. The midpoint preserves the old shape.
    const roundingRadius = Math.max(0, (50 - shape.sharpness) * 1.5);
    if (roundingRadius > 0) {
      const original = points.map(point => point.y);
      for (let index = 0; index < points.length; index++) {
        const x = points[index].x;
        let total = original[index], weight = 1;
        for (const direction of [-1, 1]) {
          for (let neighbor = index + direction; neighbor >= 0 && neighbor < points.length; neighbor += direction) {
            const distance = Math.abs(points[neighbor].x - x);
            if (distance >= roundingRadius) break;
            const influence = 1 - distance / roundingRadius;
            total += original[neighbor] * influence;
            weight += influence;
          }
        }
        points[index].y = total / weight;
      }
    }
    const maxElevation = Math.max(...points.map(point => baseY - point.y), 1);
    const sharpExponent = 1 + Math.max(0, shape.sharpness - 50) * 0.014;
    points.forEach(point => {
      if (sharpExponent > 1 && point.y < baseY) {
        const elevation = (baseY - point.y) / maxElevation;
        point.y = baseY - maxElevation * Math.pow(elevation, sharpExponent);
      }
      const envelope = clamp(Math.min(point.x, BASE_WIDTH - point.x) / 90, 0, 1);
      point.y += envelope * (coarse(point.x) + middle(point.x) + fine(point.x));
      point.y = clamp(point.y, 30, BASE_HEIGHT - 100);
    });
    composeTerrain(points, peaks, baseY, seed, depth, shape.creativity);
    peaks.forEach(peak => { peak.peakY = profileYAt(points, peak.center); });
    return { points, peaks, baseY, peakY: Math.min(...points.map(p => p.y)), style, depth };
  }

  function profileYAt(points, x) {
    if (x <= points[0].x) return points[0].y;
    if (x >= points[points.length - 1].x) return points[points.length - 1].y;
    let low = 0;
    let high = points.length - 1;
    while (high - low > 1) {
      const middle = (low + high) >> 1;
      if (points[middle].x < x) low = middle;
      else high = middle;
    }
    const left = points[low];
    const right = points[high];
    const amount = (x - left.x) / (right.x - left.x);
    return left.y + (right.y - left.y) * amount;
  }

  function profileFingerprint(profile) {
    let hash = 2166136261;
    for (let index = 0; index < profile.points.length; index += 1) {
      const point = profile.points[index];
      hash ^= Math.round(point.x * 4);
      hash = Math.imul(hash, 16777619);
      hash ^= Math.round(point.y * 4);
      hash = Math.imul(hash, 16777619);
    }
    return (hash >>> 0).toString(36);
  }

  function makeLayout(profile, seed, spacing, nearer) {
    const random = randomSource(seed + '|layout|' + profile.depth);
    if (profile.depth === 0) return {scale: 0.9, x: 120 + between(random, -18, 18), y: 80, peakY: 80 + profile.peakY * 0.9};
    const scale = profile.depth === 1 ? 0.82 - spacing * 0.05 : 0.71 - spacing * 0.05;
    const heights = nearer.profile.points.filter(point => point.x > BASE_WIDTH * 0.2 && point.x < BASE_WIDTH * 0.8)
      .map(point => nearer.layout.y + point.y * nearer.layout.scale).sort((a, b) => a - b);
    const horizon = heights[Math.floor(heights.length * 0.55)];
    const y = Math.max(8 - profile.peakY * scale, horizon + 65 - spacing * 40 - profile.baseY * scale);
    const x = (BASE_WIDTH - BASE_WIDTH * scale) / 2 + between(random, -65, 65);
    return {scale, x, y, peakY: y + profile.peakY * scale};
  }

  function withLayerTransform(context, viewport, layout, callback) {
    context.save();
    context.setTransform(viewport.scale, 0, 0, viewport.scale, viewport.x, viewport.y);
    context.translate(layout.x, layout.y);
    context.scale(layout.scale, layout.scale);
    callback();
    context.restore();
  }

  function eraseBehind(context, profile, layout, viewport) {
    withLayerTransform(context, viewport, layout, () => {
      const bottom = (BASE_HEIGHT - layout.y) / layout.scale;
      const fade = 58;
      const mask = context.createLinearGradient(0, 0, BASE_WIDTH, 0);
      mask.addColorStop(0, "rgba(0, 0, 0, 0)");
      mask.addColorStop(fade / BASE_WIDTH, "rgba(0, 0, 0, 1)");
      mask.addColorStop(1 - fade / BASE_WIDTH, "rgba(0, 0, 0, 1)");
      mask.addColorStop(1, "rgba(0, 0, 0, 0)");
      context.globalCompositeOperation = "destination-out";
      context.globalAlpha = 1;
      context.fillStyle = mask;
      context.beginPath();
      context.moveTo(profile.points[0].x, profile.points[0].y);
      for (let index = 1; index < profile.points.length; index += 1) {
        context.lineTo(profile.points[index].x, profile.points[index].y);
      }
      context.lineTo(BASE_WIDTH, bottom);
      context.lineTo(0, bottom);
      context.closePath();
      context.fill();
    });
  }

  function drawProfilePass(context, points, style) {
    context.beginPath();
    context.moveTo(points[0].x, points[0].y + style.offset(points[0].x));
    for (let index = 1; index < points.length; index += 1) {
      const point = points[index];
      context.lineTo(point.x, point.y + style.offset(point.x));
    }
    context.lineWidth = style.width;
    context.globalAlpha = style.alpha;
    context.stroke();
  }

  function drawContour(context, profile, peak, random, alpha, style) {
    const side = random() < 0.5 ? -1 : 1;
    const span = side < 0 ? peak.leftSpan : peak.rightSpan;
    const startX = peak.center - side * span * between(random, 0.025, 0.085);
    const endX = peak.center + side * span * between(random, 0.22, style === "rolling" ? 0.48 : 0.39);
    const lowX = Math.min(startX, endX);
    const highX = Math.max(startX, endX);
    const verticalOffset = between(random, 20, style === "rolling" ? 54 : 42);
    const points = profile.points.filter((point) => point.x >= lowX && point.x <= highX);
    if (points.length < 3) return;

    context.beginPath();
    points.forEach((point, index) => {
      const wave = Math.sin((point.x - lowX) / Math.max(1, highX - lowX) * Math.PI) * between(random, -3.5, 3.5);
      const y = point.y + verticalOffset + wave;
      if (index === 0) context.moveTo(point.x, y);
      else context.lineTo(point.x, y);
    });
    context.lineWidth = between(random, 0.62, 0.95);
    context.globalAlpha = alpha * between(random, 0.2, 0.37);
    context.stroke();
  }

  function drawSpur(context, profile, peak, side, random, alpha) {
    const span = side < 0 ? peak.leftSpan : peak.rightSpan;
    const startX = peak.center + side * span * between(random, 0.07, 0.15);
    const endX = peak.center + side * span * between(random, 0.42, 0.7);
    const startTop = profileYAt(profile.points, startX);
    const endTop = profileYAt(profile.points, endX);
    const startY = startTop + between(random, 24, 63);
    const endY = endTop + between(random, 18, 55);
    const controlX = startX + (endX - startX) * between(random, 0.38, 0.65);
    const controlY = startY + (endY - startY) * between(random, 0.42, 0.76);

    context.beginPath();
    context.moveTo(startX, startY);
    context.quadraticCurveTo(controlX, controlY, endX, endY);
    context.lineWidth = between(random, 0.8, 1.25);
    context.globalAlpha = alpha * between(random, 0.24, 0.48);
    context.stroke();

    if (random() < 0.62) {
      const branchStartX = startX + (endX - startX) * between(random, 0.43, 0.68);
      const branchStartY = startY + (endY - startY) * between(random, 0.43, 0.68);
      const branchEndX = branchStartX + side * between(random, 28, 92);
      const branchEndY = branchStartY + between(random, 24, 76);
      context.beginPath();
      context.moveTo(branchStartX, branchStartY);
      context.quadraticCurveTo(
        branchStartX + side * between(random, 12, 44),
        branchStartY + between(random, 12, 38),
        branchEndX,
        branchEndY
      );
      context.lineWidth = between(random, 0.62, 0.95);
      context.globalAlpha = alpha * between(random, 0.15, 0.31);
      context.stroke();
    }
  }

  function drawHatches(context, profile, peak, random, alpha, style) {
    const rows = style === "rolling" ? 2 : 3;
    [-1, 1].forEach((side) => {
      if (random() < 0.18) return;
      const span = side < 0 ? peak.leftSpan : peak.rightSpan;
      for (let row = 0; row < rows; row += 1) {
        const position = 0.14 + row * (style === "rolling" ? 0.15 : 0.12) + random() * 0.05;
        const x = peak.center + side * span * position;
        const top = profileYAt(profile.points, x);
        const y = top + 26 + row * 17 + random() * 18;
        const count = 2 + Math.floor(random() * 3);
        for (let mark = 0; mark < count; mark += 1) {
          const startX = x + side * (mark * between(random, 3, 7)) + between(random, -5, 5);
          const startY = y + mark * between(random, 3, 7);
          const length = between(random, 7, style === "rolling" ? 14 : 20);
          context.beginPath();
          context.moveTo(startX, startY);
          context.lineTo(startX + side * length * between(random, 0.38, 0.75), startY + length * between(random, 0.18, 0.58));
          context.lineWidth = between(random, 0.58, 0.92);
          context.globalAlpha = alpha * between(random, 0.09, 0.2);
          context.stroke();
        }
      }
    });
  }

  function drawRockFace(context, profile, peak, random, alpha) {
    const side = random() < 0.5 ? -1 : 1;
    const footY = profile.baseY + between(random, 45, 115);
    const footX = clamp(peak.center + side * peak.segmentWidth * between(random, 0.12, 0.34), 12, BASE_WIDTH - 12);
    const spine = [];
    const steps = 14;
    for (let i = 0; i <= steps; i++) {
      const t = i / steps;
      const x = peak.center + (footX - peak.center) * t + Math.sin(t * Math.PI) * between(random, -16, 16);
      const y = Math.max(profileYAt(profile.points, x) + 2, peak.peakY + (footY - peak.peakY) * t);
      spine.push({x, y});
    }
    const shadeWidth = peak.segmentWidth * between(random, 0.13, 0.29);
    const flank = spine.map((point, i) => {
      const t = i / steps;
      const x = clamp(point.x - side * shadeWidth * Math.sin(t * Math.PI * 0.85), 0, BASE_WIDTH);
      return {x, y: Math.max(profileYAt(profile.points, x) + 2, point.y - Math.sin(t * Math.PI) * 20)};
    });
    context.save();
    context.beginPath();
    spine.forEach((point, i) => i ? context.lineTo(point.x, point.y) : context.moveTo(point.x, point.y));
    flank.slice().reverse().forEach(point => context.lineTo(point.x, point.y));
    context.closePath();
    context.clip();
    context.globalAlpha = alpha * 0.035;
    context.fill();
    // Narrow broken strokes follow the rock face, with varied pressure and length.
    const marks = profile.style === 'rolling' ? 65 : 210;
    for (let i = 0; i < marks; i++) {
      const t = between(random, 0.04, 0.94);
      const index = Math.min(steps - 1, Math.floor(t * steps));
      const a = spine[index], b = flank[index];
      const f = random();
      const x = a.x + (b.x - a.x) * f;
      const y = a.y + (b.y - a.y) * f + between(random, -9, 9);
      const length = between(random, 9, 65);
      context.beginPath();
      context.moveTo(x, y);
      context.lineTo(x + side * length * between(random, 0.12, 0.5), y + length);
      context.lineWidth = between(random, 0.65, 1.5);
      context.globalAlpha = alpha * between(random, 0.14, 0.4) * (1 - t * 0.65);
      context.stroke();
    }
    context.restore();
    for (let pass = 0; pass < 2; pass++) {
      for (let i = 1; i < spine.length; i++) {
        context.beginPath();
        context.moveTo(spine[i - 1].x + pass * 1.2, spine[i - 1].y);
        context.lineTo(spine[i].x + between(random, -1.2, 1.2), spine[i].y);
        context.globalAlpha = alpha * (0.65 - i / spine.length * 0.5) / (pass + 1);
        context.lineWidth = pass === 0 ? 1.65 : 0.85;
        context.stroke();
      }
    }
    // Branching creases carry the face down into the foothills.
    for (let branch = 0; branch < 4; branch++) {
      const start = spine[3 + branch * 2];
      const direction = branch % 2 ? side : -side;
      const endX = clamp(start.x + direction * peak.segmentWidth * between(random, 0.12, 0.3), 0, BASE_WIDTH);
      const endY = Math.max(profileYAt(profile.points, endX) + 10, start.y + between(random, 65, 145));
      context.beginPath();
      context.moveTo(start.x, start.y);
      context.lineTo(start.x + (endX - start.x) * 0.32, start.y + (endY - start.y) * 0.3 + between(random, -12, 12));
      context.lineTo(start.x + (endX - start.x) * 0.7, start.y + (endY - start.y) * 0.75);
      context.lineTo(endX, endY);
      context.lineWidth = between(random, 0.8, 1.3);
      context.globalAlpha = alpha * between(random, 0.18, 0.34);
      context.stroke();
    }
  }

  function drawLayer(context, profile, layout, viewport, seed, graphite, depth) {
    const random = randomSource(`${seed}|marks|${depth}`);
    const strength = graphite / 100;
    const depthFactor = strength * (depth === 0 ? 1 : depth === 1 ? 0.46 : 0.29);
    const scale = viewport.scale * layout.scale;

    withLayerTransform(context, viewport, layout, () => {
      context.globalCompositeOperation = "source-over";
      context.strokeStyle = "#292a2b";
      context.fillStyle = "#292a2b";
      context.lineCap = "round";
      context.lineJoin = "round";

      const imperfect = smoothNoise(random, 94, 1.2);
      drawProfilePass(context, profile.points, {
        offset: () => 0,
        width: 1.75,
        alpha: depthFactor * 0.58
      });
      drawProfilePass(context, profile.points, {
        offset: (x) => imperfect(x),
        width: 1.05,
        alpha: depthFactor * 0.28
      });
      if (depth === 0) {
        drawProfilePass(context, profile.points, {
          offset: (x) => -imperfect(x) * 0.48 + Math.sin(x * 0.018) * 0.24,
          width: 0.68,
          alpha: depthFactor * 0.17
        });
      }

      profile.peaks.forEach((peak) => {
        drawRockFace(context, profile, peak, random, depthFactor);
        const contourCount = profile.style === "rolling" ? 1 : (random() < 0.72 ? 2 : 1);
        for (let contour = 0; contour < contourCount; contour += 1) {
          drawContour(context, profile, peak, random, depthFactor, profile.style);
        }
        if (random() < 0.9) drawSpur(context, profile, peak, -1, random, depthFactor);
        if (random() < 0.88) drawSpur(context, profile, peak, 1, random, depthFactor);
        drawHatches(context, profile, peak, random, depthFactor, profile.style);
      });

      const grainCount = depth === 0 ? 105 : depth === 1 ? 62 : 38;
      context.globalAlpha = depthFactor * 0.11;
      for (let mark = 0; mark < grainCount; mark += 1) {
        const peak = profile.peaks[Math.floor(random() * profile.peaks.length)];
        const side = random() < 0.5 ? -1 : 1;
        const x = clamp(peak.center + side * between(random, 22, Math.max(24, peak.segmentWidth * 0.42)), 0, BASE_WIDTH);
        const top = profileYAt(profile.points, x);
        const y = top + between(random, 20, profile.style === "rolling" ? 90 : 132);
        context.fillRect(x, y, between(random, 0.45, 1.05), between(random, 0.45, 1.05));
      }

      context.globalAlpha = 1;
    });
  }

  function getTerrainCanvas(canvas) {
    let terrainCanvas = TERRAIN_CANVASES.get(canvas);
    if (!terrainCanvas || terrainCanvas.width !== canvas.width || terrainCanvas.height !== canvas.height) {
      terrainCanvas = document.createElement("canvas");
      terrainCanvas.width = canvas.width;
      terrainCanvas.height = canvas.height;
      TERRAIN_CANVASES.set(canvas, terrainCanvas);
    }
    return terrainCanvas;
  }

  function draw(canvas, state) {
    if (!canvas || typeof canvas.getContext !== "function") return null;
    const width = canvas.width;
    const height = canvas.height;
    if (!width || !height) return null;

    const mainContext = canvas.getContext("2d", { alpha: true });
    const terrainCanvas = getTerrainCanvas(canvas);
    const terrainContext = terrainCanvas.getContext("2d", { alpha: true });
    if (!mainContext || !terrainContext) return null;

    const seed = typeof state?.seed === "string" && state.seed.length > 0 ? state.seed : "mountain-studio";
    const style = chooseStyle(state?.ridge, seed);
    const layerCount = Math.round(clamp(Number(state?.layers) || 1, 1, 3));
    const graphite = clamp(Number(state?.graphite) || 65, 25, 100);
    const spacing = clamp(Number(state?.spacing) || 0, 0, 100) / 100;
    const paper = state?.paper === "white" ? "white" : "transparent";
    const shape = {};
    for (const key of ['sharpness', 'bumpiness', 'entropy', 'creativity']) {
      const value = key === 'creativity' ? state?.creativity ?? state?.wildness : state?.[key];
      shape[key] = value !== null && value !== undefined && Number.isFinite(Number(value))
        ? clamp(Number(value), 0, 100) : key === 'creativity' ? 0 : 50;
    }
    const viewportScale = Math.min(width / BASE_WIDTH, height / BASE_HEIGHT);
    const viewport = {
      scale: viewportScale,
      x: (width - BASE_WIDTH * viewportScale) / 2,
      y: (height - BASE_HEIGHT * viewportScale) / 2
    };

    const foregroundProfile = makeProfile(seed, style, 0, shape);
    const foregroundLayout = makeLayout(foregroundProfile, seed, spacing, 0);
    const profiles = [{ profile: foregroundProfile, layout: foregroundLayout }];
    let nearer = {profile: foregroundProfile, layout: foregroundLayout};
    const usedStyles = new Set([style]);

    for (let depth = 1; depth < layerCount; depth += 1) {
      const requested = depth === 1 ? state?.midRidge : state?.farRidge;
      let layerStyle = STYLES.includes(requested) ? requested : style;
      if (requested === 'auto') {
        const choices = STYLES.filter(candidate => !usedStyles.has(candidate));
        const random = randomSource(seed + '|terrain|' + depth);
        layerStyle = choices[Math.floor(random() * choices.length)];
      }
      usedStyles.add(layerStyle);
      const profile = makeProfile(seed, layerStyle, depth, shape);
      const layout = makeLayout(profile, seed, spacing, nearer);
      profiles.unshift({ profile, layout });
      nearer = {profile, layout};
    }

    terrainContext.setTransform(1, 0, 0, 1, 0, 0);
    terrainContext.globalCompositeOperation = "source-over";
    terrainContext.globalAlpha = 1;
    terrainContext.clearRect(0, 0, width, height);

    profiles.forEach((layer, index) => {
      if (index > 0) eraseBehind(terrainContext, layer.profile, layer.layout, viewport);
      drawLayer(terrainContext, layer.profile, layer.layout, viewport, seed, graphite, layer.profile.depth);
    });

    mainContext.setTransform(1, 0, 0, 1, 0, 0);
    mainContext.globalCompositeOperation = "source-over";
    mainContext.globalAlpha = 1;
    mainContext.clearRect(0, 0, width, height);
    if (paper === "white") {
      mainContext.fillStyle = "#ffffff";
      mainContext.fillRect(0, 0, width, height);
    }
    mainContext.drawImage(terrainCanvas, 0, 0);

    const layers = profiles.map((layer) => ({
      source: layer.profile.depth === 0 ? "procedural-foreground" : layer.profile.depth === 1 ? "procedural-near" : "procedural-far",
      depth: layer.profile.depth,
      style: layer.profile.style
    }));
    return {
      style,
      peakCount: foregroundProfile.peaks.length,
      layers,
      foregroundFingerprint: profileFingerprint(foregroundProfile)
    };
  }

  window.MountainTerrainV2 = Object.freeze({ draw });
})();
