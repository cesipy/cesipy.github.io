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

  function pencilLine(context, points, random, alpha, width = 1) {
    // Pressure changes and gaps, without tracing every mark twice.
    for (let index = 1; index < points.length; index++) {
      if (random() < 0.12) continue;
      const fade = 1 - 0.58 * index / points.length;
      context.beginPath();
      context.moveTo(points[index - 1].x, points[index - 1].y);
      context.lineTo(points[index].x, points[index].y);
      context.lineWidth = width * between(random, 0.7, 1.15);
      context.globalAlpha = clamp(alpha * fade * between(random, 0.65, 1), 0, 1);
      context.stroke();
    }
  }

  function slopeContour(context, profile, peak, random, alpha, side) {
    const span = side < 0 ? peak.leftSpan : peak.rightSpan;
    const start = between(random, 0.08, 0.3);
    const end = between(random, 0.5, 0.87);
    const inset = between(random, 18, 45);
    const points = [];
    for (let index = 0; index <= 18; index++) {
      const t = index / 18;
      const x = peak.center + side * span * (start + (end - start) * t);
      points.push({x, y: profileYAt(profile.points, x) + inset * (1 - t * 0.6) + Math.sin(t * Math.PI) * 12});
    }
    pencilLine(context, points, random, alpha, 0.95);
  }

  function shadowWash(context, points, peak, baseY, alpha, extra) {
    if (extra <= 0) return;
    context.save();
    const tone = context.createLinearGradient(0, peak.peakY, 0, baseY + 30);
    tone.addColorStop(0, 'rgba(41,42,43,0)');
    tone.addColorStop(0.12, 'rgba(41,42,43,1)');
    tone.addColorStop(0.62, 'rgba(41,42,43,.75)');
    tone.addColorStop(1, 'rgba(41,42,43,0)');
    context.fillStyle = tone;
    context.globalAlpha = Math.min(0.34, alpha * extra * 0.14);
    context.beginPath();
    points.forEach((point, index) => index ? context.lineTo(point.x, point.y) : context.moveTo(point.x, point.y));
    context.closePath();
    context.fill();
    context.restore();
  }

  function drawShadowFieldV4(context, profile, peak, random, alpha, shadowSide, shading, focal) {
    const strength = Math.pow(shading / 100, 1.35);
    const span = shadowSide < 0 ? peak.leftSpan : peak.rightSpan;
    const height = profile.baseY - peak.peakY;
    if (span < 12 || height < 25 || strength <= 0) return;

    // A broad shaded face follows the actual skyline. The seeded light side is
    // shared by all layers; the per-peak pressure keeps the wash from repeating.
    const terrainWeight = profile.style === 'rolling' ? 0.52 :
      profile.style === 'spires' ? 0.72 : profile.style === 'massif' ? 1.12 : 1;
    const pressure = (focal ? 1.12 : between(random, 0.68, 1)) * terrainWeight;
    const edge = peak.center + shadowSide * span * between(random, 0.8, 0.94);
    const depth = height * (profile.style === 'rolling' ? 0.48 :
      profile.style === 'spires' ? 0.64 : 0.82);
    const gradient = context.createLinearGradient(0, peak.peakY, 0, profile.baseY + 35);
    gradient.addColorStop(0, 'rgba(41,42,43,0)');
    gradient.addColorStop(0.16, 'rgba(41,42,43,.8)');
    gradient.addColorStop(0.48, 'rgba(41,42,43,1)');
    gradient.addColorStop(0.82, 'rgba(41,42,43,.58)');
    gradient.addColorStop(1, 'rgba(41,42,43,0)');
    context.save();
    context.fillStyle = gradient;
    context.globalAlpha = clamp(alpha * strength * 0.16 * pressure, 0, 0.18);
    context.beginPath();
    context.moveTo(peak.center, profileYAt(profile.points, peak.center) + 2);
    for (let step = 1; step <= 32; step++) {
      const x = peak.center + (edge - peak.center) * step / 32;
      context.lineTo(x, profileYAt(profile.points, x) + 2);
    }
    // The lower edge curls back into the slope at both ends. This avoids the
    // vertical cutoffs that a summit-to-base polygon leaves on the paper.
    for (let step = 31; step >= 0; step--) {
      const t = step / 32;
      const x = peak.center + (edge - peak.center) * t;
      const feather = Math.pow(Math.sin(Math.PI * t), 0.72);
      context.lineTo(x, profileYAt(profile.points, x) + 2 + depth * feather);
    }
    context.closePath();
    context.fill();
    context.restore();

    // Distributed graphite does most of the darkening. Unlike a filled face,
    // the marks thin out naturally toward the valley and the lit shoulder.
    const textureWeight = profile.style === 'rolling' ? 0.55 :
      profile.style === 'plateau' ? 0.7 : 1;
    const count = Math.round(clamp(span * height / 360, 45, 380) * strength * pressure * textureWeight);
    for (let mark = 0; mark < count; mark++) {
      const along = between(random, 0.06, 0.93);
      const x = peak.center + shadowSide * span * along;
      const top = profileYAt(profile.points, x);
      const vertical = random();
      const y = top + 8 + (profile.baseY - top) * vertical * 0.75;
      const length = between(random, 8, profile.style === 'rolling' ? 30 : 42);
      let dx, dy;
      if (profile.style === 'rolling') {
        dx = shadowSide * length;
        dy = (profileYAt(profile.points, x + dx) - top) * 0.55 + length * 0.08;
      } else if (profile.style === 'plateau') {
        dx = shadowSide * length * 0.16;
        dy = length * 0.72;
      } else {
        dx = shadowSide * length * between(random, 0.28, 0.72);
        dy = length * between(random, 0.55, 0.9);
      }
      const fade = (1 - vertical * 0.72) * (1 - along * 0.45);
      pencilLine(context, [
        {x, y},
        {x: x + dx * 0.48 + between(random, -2, 2), y: y + dy * 0.48},
        {x: x + dx, y: y + dy}
      ], random, alpha * strength * pressure * fade * 0.55, between(random, 0.7, 1.15));
    }
  }

  function drawStrata(context, profile, peak, random, alpha, shadowSide, shading, version) {
    const height = Math.max(30, profile.baseY - peak.peakY);
    const extra = Math.max(0, shading - 50) / 50;
    const shadowSpan = shadowSide < 0 ? peak.leftSpan : peak.rightSpan;
    if (extra > 0 && version === '3') shadowWash(context, [
      {x: peak.center, y: peak.peakY},
      {x: peak.center + shadowSide * shadowSpan * 0.8, y: peak.peakY + height * 0.12},
      {x: peak.center + shadowSide * shadowSpan * 0.95, y: profile.baseY + 30},
      {x: peak.center + shadowSide * shadowSpan * 0.12, y: profile.baseY + 30}
    ], peak, profile.baseY, alpha, extra);
    const tilt = between(random, -0.045, 0.045);
    const rows = 3 + Math.floor(random() * 4);
    for (let row = 0; row < rows; row++) {
      const y = peak.peakY + height * (0.16 + row / rows * 0.62);
      const left = peak.center - peak.leftSpan * between(random, 0.3, 0.8);
      const right = peak.center + peak.rightSpan * between(random, 0.3, 0.8);
      const points = [];
      for (let x = left; x <= right; x += 8) {
        points.push({x, y: y + (x - peak.center) * tilt + Math.sin(x * 0.025 + row) * 2 + between(random, -1.2, 1.2)});
      }
      pencilLine(context, points, random, alpha * between(random, 0.34, 0.5), 1.05);
    }
    // Short sedimentary joints, not summit-to-ground alpine creases.
    const span = shadowSide < 0 ? peak.leftSpan : peak.rightSpan;
    for (let joint = 0; joint < 3; joint++) {
      const x = peak.center + shadowSide * span * between(random, 0.18, 0.75);
      const y = profileYAt(profile.points, x) + between(random, 12, 50);
      const length = height * between(random, 0.12, 0.34);
      const points = [{x, y}, {x: x + shadowSide * 5, y: y + length * 0.55}, {x: x - shadowSide * 3, y: y + length}];
      pencilLine(context, points, random, alpha * 0.42, 1.05);
    }
    // Extra pencil tone lies on the shaded cliff, leaving the strata readable.
    for (let mark = 0; mark < Math.round(220 * extra); mark++) {
      const x = peak.center + shadowSide * span * between(random, 0.08, 0.8);
      const top = profileYAt(profile.points, x);
      const y = top + (profile.baseY - top) * between(random, 0.12, 0.78);
      const length = between(random, 8, 32);
      pencilLine(context, [{x, y}, {x: x + shadowSide * length * 0.24, y: y + length}], random, alpha * 0.2 * extra, 0.95);
    }
  }

  function drawFacets(context, profile, peak, random, alpha, shadowSide, shading, version) {
    const slender = profile.style === 'spires';
    const massive = profile.style === 'massif';
    const height = Math.max(25, profile.baseY - peak.peakY);
    const span = shadowSide < 0 ? peak.leftSpan : peak.rightSpan;
    const drift = span * between(random, 0.06, slender ? 0.23 : 0.42);
    const bend = between(random, -0.08, 0.08) * peak.segmentWidth;
    const rib = t => ({
      x: peak.center + shadowSide * drift * t + Math.sin(t * Math.PI) * bend,
      y: peak.peakY + height * t
    });
    const opening = between(random, slender ? 0.2 : 0.5, massive ? 1.05 : 0.88);
    const extra = Math.max(0, shading - 50) / 50;
    if (extra > 0 && version === '3') {
      const inner = [], outer = [];
      for (let index = 0; index <= 18; index++) {
        const t = index / 18;
        const point = rib(t);
        inner.push(point);
        outer.push({x: point.x + shadowSide * span * Math.pow(t, 0.8) * opening, y: point.y + height * t * 0.05});
      }
      shadowWash(context, inner.concat(outer.reverse()), peak, profile.baseY, alpha, extra);
    }
    const baseMarks = Math.round(clamp(span * height / 450, 65, massive ? 390 : 290) * (slender ? 0.55 : 1));
    const marks = Math.round(baseMarks * (1 + extra * 1.5));
    // An open fan suggests a shaded plane, without a filled ribbon or outline.
    for (let mark = 0; mark < marks; mark++) {
      const t = between(random, 0.07, slender ? 0.72 : 0.88);
      const across = Math.pow(random(), 1.25);
      const start = rib(t);
      const x = start.x + shadowSide * span * Math.pow(t, 0.8) * opening * across;
      const y = start.y + across * height * between(random, -0.05, 0.09);
      if (y < profileYAt(profile.points, x) + 4) continue;
      const length = between(random, 12, slender ? 42 : 72);
      const dx = shadowSide * length * between(random, 0.3, 0.85);
      const dy = length * between(random, 0.45, 0.95);
      const points = [
        {x, y},
        {x: x + dx * 0.45 + between(random, -2, 2), y: y + dy * 0.45},
        {x: x + dx, y: y + dy}
      ];
      const fade = Math.pow(1 - t, 0.65) * (1 - across * 0.55);
      pencilLine(context, points, random, alpha * between(random, 0.34, 0.62) * fade, between(random, 0.85, 1.4));
    }
    // Interrupted structural folds vary in length and branching by rock type.
    const folds = slender ? 1 : massive ? 3 : 1 + Math.floor(random() * 3);
    for (let fold = 0; fold < folds; fold++) {
      const begin = fold === 0 ? between(random, 0.02, 0.12) : between(random, 0.2, 0.45);
      const end = between(random, 0.53, slender ? 0.75 : 0.9);
      const reach = fold === 0 ? 0 : shadowSide * span * between(random, 0.18, 0.65);
      const points = [];
      for (let index = 0; index <= 12; index++) {
        const t = index / 12;
        const point = rib(begin + (end - begin) * t);
        point.x += reach * Math.pow(t, 1.3) + Math.sin(t * Math.PI) * between(random, -5, 5);
        points.push(point);
      }
      pencilLine(context, points, random, alpha * (fold === 0 ? 0.6 : 0.36), fold === 0 ? 1.25 : 0.9);
    }
  }

  function drawHillTone(context, profile, peak, random, alpha, shadowSide, extra, version) {
    const span = shadowSide < 0 ? peak.leftSpan : peak.rightSpan;
    const height = Math.max(25, profile.baseY - peak.peakY);
    if (version === '3') shadowWash(context, [
      {x: peak.center, y: peak.peakY},
      {x: peak.center + shadowSide * span * 0.48, y: peak.peakY + height * 0.32},
      {x: peak.center + shadowSide * span * 0.92, y: profile.baseY + 25},
      {x: peak.center + shadowSide * span * 0.14, y: profile.baseY + 25}
    ], peak, profile.baseY, alpha * 0.65, extra);
    // Short curved strokes follow the hillside instead of creating rock folds.
    for (let mark = 0; mark < Math.round(extra * 190); mark++) {
      const start = between(random, 0.06, 0.62);
      const length = between(random, 0.06, 0.2);
      const inset = between(random, 10, 55);
      const points = [];
      for (let index = 0; index <= 4; index++) {
        const t = index / 4;
        const x = peak.center + shadowSide * span * (start + length * t);
        points.push({x, y: profileYAt(profile.points, x) + inset + Math.sin(t * Math.PI) * 2});
      }
      pencilLine(context, points, random, alpha * extra * 0.19, 0.95);
    }
  }

  function drawTerrainDetails(context, profile, seed, alpha, shadowSide, shading, version) {
    if (shading === 0) return;
    const extra = Math.max(0, shading - 50) / 50;
    // The upper half deliberately opens a much stronger contrast range.
    // 50 remains the archived v3 look; 70 is visibly shaded, 100 is bold.
    const pressure = shading <= 50 ? Math.pow(shading / 50, 1.3) : 1 + extra * 5;
    const focal = profile.peaks.reduce((best, peak, index) => peak.peakY < profile.peaks[best].peakY ? index : best, 0);
    context.save();
    // Clip details below their own skyline; never draw marks into the sky.
    context.beginPath();
    profile.points.forEach((point, index) => index ? context.lineTo(point.x, point.y + 1) : context.moveTo(point.x, point.y + 1));
    context.lineTo(BASE_WIDTH, profile.baseY + 110);
    context.lineTo(0, profile.baseY + 110);
    context.closePath();
    context.clip();

    profile.peaks.forEach((peak, index) => {
      const random = randomSource(seed + '|faces-v3|' + profile.depth + '|' + index);
      const prominent = index === focal || random() < (profile.depth === 0 ? 0.48 : 0.3) + extra * 0.3;
      const detail = alpha * (profile.depth === 0 ? 1 : profile.depth === 1 ? 0.67 : 0.4) * pressure;
      if (version === '4') {
        const toneRandom = randomSource(seed + '|shade-v4|' + profile.depth + '|' + index);
        drawShadowFieldV4(context, profile, peak, toneRandom, alpha, shadowSide, shading, index === focal);
      }
      if (profile.style === 'rolling') {
        // Rounded hills stay open and soft, with no sharp rock-face skeleton.
        if (prominent || random() < 0.55) slopeContour(context, profile, peak, random, detail * 0.38, shadowSide);
        if (prominent && profile.depth === 0) slopeContour(context, profile, peak, random, detail * 0.12, -shadowSide);
        if (prominent && extra > 0) drawHillTone(context, profile, peak, random, detail, shadowSide, extra, version);
      } else if (profile.style === 'plateau') {
        if (prominent) drawStrata(context, profile, peak, random, detail, shadowSide, shading, version);
      } else {
        if (prominent) drawFacets(context, profile, peak, random, detail, shadowSide, shading, version);
        if (random() < 0.6) slopeContour(context, profile, peak, random, detail * 0.18, shadowSide);
      }
    });
    context.restore();
  }

  function drawLayer(context, profile, layout, viewport, seed, graphite, depth, shadowSide, shading, version) {
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

      drawTerrainDetails(context, profile, seed, depthFactor, shadowSide, shading, version);

      const grainCount = depth === 0 ? 105 : depth === 1 ? 62 : 38;
      context.globalAlpha = depthFactor * 0.11 * Math.min(2, shading / 50);
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
    if (String(state?.v) === '2') {
      if (!window.MountainTerrainV2) throw new Error('The saved drawing renderer did not load. Reload to try again.');
      return window.MountainTerrainV2.draw(canvas, state);
    }
    const version = String(state?.v) === '3' ? '3' : '4';
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
    // Missing shading retains the exact v3 appearance of older saved links.
    const shading = state?.shading !== null && state?.shading !== undefined && Number.isFinite(Number(state.shading))
      ? clamp(Number(state.shading), 0, 100) : 50;
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

    const shadowSide = randomSource(seed + '|lighting-v3')() < 0.5 ? -1 : 1;
    profiles.forEach((layer, index) => {
      if (index > 0) eraseBehind(terrainContext, layer.profile, layer.layout, viewport);
      drawLayer(terrainContext, layer.profile, layer.layout, viewport, seed, graphite, layer.profile.depth, shadowSide, shading, version);
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
      version,
      lightDirection: shadowSide === 1 ? 'upper-left' : 'upper-right',
      style,
      peakCount: foregroundProfile.peaks.length,
      layers,
      foregroundFingerprint: profileFingerprint(foregroundProfile)
    };
  }

  window.MountainTerrain = Object.freeze({ draw });
})();
