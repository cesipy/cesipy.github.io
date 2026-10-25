(function () {
  'use strict';

  if (window.__cesipyMountainBackgroundInitialized || !document.body) return;
  window.__cesipyMountainBackgroundInitialized = true;

  var canvas = document.createElement('canvas');
  canvas.className = 'mountain-background';
  canvas.setAttribute('aria-hidden', 'true');
  var context = canvas.getContext('2d', { alpha: true });
  if (!context) return;

  var images = [];
  var tiles = [];
  var textRects = [];
  var frame = 0;
  var resizeFrame = 0;
  var scrollFrame = 0;
  var pointer = { x: -1000, y: -1000, active: false };
  var lastFrameTime = 0;
  var interactive = false;
  var capability = window.matchMedia && window.matchMedia(
    '(hover: hover) and (pointer: fine) and (prefers-reduced-motion: no-preference)'
  );
  var moveEvent = window.PointerEvent ? 'pointermove' : 'mousemove';

  document.body.classList.add('mountain-background-enabled');
  document.body.appendChild(canvas);

  function random(seed) {
    var value = seed >>> 0;
    return function () {
      value += 0x6D2B79F5;
      var t = value;
      t = Math.imul(t ^ t >>> 15, t | 1);
      t ^= t + Math.imul(t ^ t >>> 7, t | 61);
      return ((t ^ t >>> 14) >>> 0) / 4294967296;
    };
  }

  function makeTiles(width, height) {
    var next = random(412903);
    var field = [];
    var compact = width < 650;
    var count = Math.max(5, Math.round(width * height / (compact ? 38000 : 43000)));
    var separation = compact ? 135 : 165;

    // Seeded dart placement keeps an even spread without visible rows.
    for (var attempt = 0; field.length < count && attempt < count * 100; attempt += 1) {
      var x = 24 + next() * Math.max(1, width - 48);
      var y = 25 + next() * Math.max(1, height - 50);
      if (field.some(function (tile) {
        return Math.hypot(tile.x - x, (tile.y - y) * 1.2) < separation;
      })) continue;
      var tileWidth = (compact ? 108 : 122) + next() * 30;
      field.push({
        image: images[field.length % images.length],
        x: x,
        y: y,
        width: tileWidth,
        height: tileWidth / 3,
        tilt: -.25,
        opacity: .8 + next() * .2,
        visibility: 1,
        offsetX: 0,
        offsetY: 0,
        velocityX: 0,
        velocityY: 0
      });
    }
    return field;
  }

  function measureText() {
    var protectedRects = [];
    var width = window.innerWidth;
    var height = window.innerHeight;
    var walker = document.createTreeWalker(document.body, window.NodeFilter ? NodeFilter.SHOW_TEXT : 4);
    var range = document.createRange();
    var node;
    while ((node = walker.nextNode())) {
      if (!node.nodeValue.trim() || !node.parentElement ||
          node.parentElement.closest('script, style, noscript')) continue;
      range.selectNodeContents(node);
      Array.prototype.forEach.call(range.getClientRects(), function (rect) {
        if (rect.width && rect.height && rect.right >= 0 && rect.left <= width &&
            rect.bottom >= 0 && rect.top <= height) {
          protectedRects.push({
            left: rect.left - 9, right: rect.right + 9,
            top: rect.top - 5, bottom: rect.bottom + 5
          });
        }
      });
    }
    document.querySelectorAll('header img, .mountain-sketch, .post-sketch, .not-found-sketch')
      .forEach(function (element) {
        var rect = element.getBoundingClientRect();
        if (rect.width && rect.height && rect.right >= 0 && rect.left <= width &&
            rect.bottom >= 0 && rect.top <= height) {
          protectedRects.push({
            left: rect.left - 5, right: rect.right + 5,
            top: rect.top - 5, bottom: rect.bottom + 5
          });
        }
      });
    textRects = protectedRects;
    tiles.forEach(function (tile) {
      var halfWidth = tile.width / 2 + 12;
      var halfHeight = tile.height / 2 + tile.width * .13 + 10;
      var coveredArea = 0;
      textRects.forEach(function (rect) {
        var overlapWidth = Math.max(0, Math.min(tile.x + halfWidth, rect.right) -
          Math.max(tile.x - halfWidth, rect.left));
        var overlapHeight = Math.max(0, Math.min(tile.y + halfHeight, rect.bottom) -
          Math.max(tile.y - halfHeight, rect.top));
        coveredArea += overlapWidth * overlapHeight;
      });
      tile.visibility = Math.max(.16, 1 - coveredArea / (4 * halfWidth * halfHeight) * 1.8);
    });
  }

  function render(timestamp) {
    frame = 0;
    if (document.hidden) return;
    var width = window.innerWidth;
    var height = window.innerHeight;
    var step = lastFrameTime ? Math.min((timestamp - lastFrameTime) / 16.667, 2) : 1;
    lastFrameTime = timestamp;
    var moving = false;
    context.clearRect(0, 0, width, height);

    tiles.forEach(function (tile) {
      var image = tile.image;
      if (!image.complete || !image.naturalWidth) return;
      var dx = tile.x - pointer.x;
      var dy = tile.y - pointer.y;
      var distance = Math.hypot(dx, dy);
      var influence = pointer.active && interactive ? Math.max(0, 1 - distance / 170) : 0;
      influence = influence * influence * (3 - 2 * influence);
      var direction = Math.atan2(dy, dx);
      var targetX = Math.cos(direction) * influence * 18;
      var targetY = Math.sin(direction) * influence * 14;

      if (interactive) {
        var damping = Math.pow(.74, step);
        tile.velocityX = (tile.velocityX + (targetX - tile.offsetX) * .075 * step) * damping;
        tile.velocityY = (tile.velocityY + (targetY - tile.offsetY) * .075 * step) * damping;
        tile.offsetX += tile.velocityX * step;
        tile.offsetY += tile.velocityY * step;
        if (Math.abs(targetX - tile.offsetX) + Math.abs(targetY - tile.offsetY) +
            Math.abs(tile.velocityX) + Math.abs(tile.velocityY) > .03) moving = true;
        else {
          tile.offsetX = targetX;
          tile.offsetY = targetY;
          tile.velocityX = tile.velocityY = 0;
        }
      } else {
        tile.offsetX = tile.offsetY = tile.velocityX = tile.velocityY = 0;
      }

      context.save();
      context.translate(tile.x + tile.offsetX, tile.y + tile.offsetY);
      context.rotate(tile.tilt);
      context.globalAlpha = tile.opacity * tile.visibility;
      context.drawImage(image, -tile.width / 2, -tile.height / 2, tile.width, tile.height);
      context.restore();
    });

    if (moving) frame = window.requestAnimationFrame(render);
    else lastFrameTime = 0;
  }

  function queueRender() {
    if (!frame) frame = window.requestAnimationFrame(render);
  }

  function resize() {
    resizeFrame = 0;
    var scale = Math.min(window.devicePixelRatio || 1, 2);
    var width = window.innerWidth;
    var height = window.innerHeight;
    canvas.width = Math.round(width * scale);
    canvas.height = Math.round(height * scale);
    context.setTransform(scale, 0, 0, scale, 0, 0);
    tiles = makeTiles(width, height);
    measureText();
    queueRender();
  }

  function onResize() {
    if (!resizeFrame) resizeFrame = window.requestAnimationFrame(resize);
  }

  function onScroll() {
    if (scrollFrame) return;
    scrollFrame = window.requestAnimationFrame(function () {
      scrollFrame = 0;
      measureText();
      queueRender();
    });
  }

  function onMove(event) {
    if ((event.pointerType && event.pointerType !== 'mouse') ||
        (event.sourceCapabilities && event.sourceCapabilities.firesTouchEvents)) return;
    pointer.x = event.clientX;
    pointer.y = event.clientY;
    pointer.active = true;
    queueRender();
  }

  function settle() {
    pointer.active = false;
    queueRender();
  }

  function onKey(event) {
    if (['Tab', 'ArrowUp', 'ArrowDown', 'ArrowLeft', 'ArrowRight',
      'Home', 'End', 'PageUp', 'PageDown', 'Enter', ' '].indexOf(event.key) !== -1) settle();
  }

  function updateCapability() {
    var shouldInteract = !!(capability && capability.matches);
    if (shouldInteract === interactive) return;
    interactive = shouldInteract;
    if (interactive) document.addEventListener(moveEvent, onMove, { passive: true });
    else {
      document.removeEventListener(moveEvent, onMove);
      settle();
    }
  }

  for (var index = 1; index <= 12; index += 1) {
    var image = new Image();
    image.decoding = 'async';
    image.onload = queueRender;
    image.src = '/assets/mountains/background/ridge-' + String(index).padStart(2, '0') + '.png';
    images.push(image);
  }

  if (capability) {
    if (capability.addEventListener) capability.addEventListener('change', updateCapability);
    else if (capability.addListener) capability.addListener(updateCapability);
  }
  updateCapability();
  document.addEventListener('keydown', onKey, true);
  document.addEventListener('mouseleave', settle);
  document.addEventListener('visibilitychange', settle);
  window.addEventListener('blur', settle);
  window.addEventListener('resize', onResize, { passive: true });
  window.addEventListener('scroll', onScroll, { passive: true });
  window.addEventListener('load', onScroll);
  resize();
})();
