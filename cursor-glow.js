(function () {
  'use strict';

  var guardName = '__cesipyCursorGlowInitialized';
  if (window[guardName]) return;
  window[guardName] = true;

  if (!window.matchMedia) return;

  var capability = window.matchMedia(
    '(hover: hover) and (pointer: fine) and (prefers-reduced-motion: no-preference)'
  );
  var glow = null;
  var enabled = false;
  var pointerSeen = false;
  var needsInstantPosition = true;
  var currentX = 0;
  var currentY = 0;
  var targetX = 0;
  var targetY = 0;
  var frameId = 0;
  var lastFrameTime = 0;
  var smoothingTime = 65;
  var settleDistance = 0.2;
  var moveEventName = window.PointerEvent ? 'pointermove' : 'mousemove';

  function setPosition(x, y) {
    if (!glow) return;
    glow.style.transform =
      'translate3d(' + x + 'px, ' + y + 'px, 0) translate(-50%, -50%)';
  }

  function stopAnimation() {
    if (frameId) {
      window.cancelAnimationFrame(frameId);
      frameId = 0;
    }
    lastFrameTime = 0;
  }

  function animate(timestamp) {
    frameId = 0;
    if (!enabled || !glow || document.hidden) return;

    var elapsed = lastFrameTime ? Math.min(timestamp - lastFrameTime, 100) : 16;
    lastFrameTime = timestamp;
    var amount = 1 - Math.exp(-elapsed / smoothingTime);
    currentX += (targetX - currentX) * amount;
    currentY += (targetY - currentY) * amount;

    var deltaX = targetX - currentX;
    var deltaY = targetY - currentY;
    if (Math.hypot(deltaX, deltaY) <= settleDistance) {
      currentX = targetX;
      currentY = targetY;
      lastFrameTime = 0;
      setPosition(currentX, currentY);
      return;
    }

    setPosition(currentX, currentY);
    frameId = window.requestAnimationFrame(animate);
  }

  function queueAnimation() {
    if (!frameId) frameId = window.requestAnimationFrame(animate);
  }

  function onMouseMove(event) {
    if (!enabled || !glow) return;
    if ((event.pointerType && event.pointerType !== 'mouse') ||
        (event.sourceCapabilities && event.sourceCapabilities.firesTouchEvents)) {
      hideGlow();
      return;
    }

    targetX = event.clientX;
    targetY = event.clientY;

    if (!pointerSeen || needsInstantPosition) {
      currentX = targetX;
      currentY = targetY;
      pointerSeen = true;
      needsInstantPosition = false;
      stopAnimation();
      setPosition(currentX, currentY);
    } else {
      queueAnimation();
    }

    glow.style.opacity = '1';
  }

  function hideGlow() {
    if (!glow) return;
    glow.style.opacity = '0';
    needsInstantPosition = true;
    stopAnimation();
  }

  function onKeyDown(event) {
    var navigationKeys = [
      'Tab', 'ArrowUp', 'ArrowDown', 'ArrowLeft', 'ArrowRight',
      'Home', 'End', 'PageUp', 'PageDown', 'Enter', ' '
    ];
    if (navigationKeys.indexOf(event.key) !== -1) hideGlow();
  }

  function addInteractionListeners() {
    document.addEventListener(moveEventName, onMouseMove, { passive: true });
    document.addEventListener('keydown', onKeyDown, true);
    document.addEventListener('mouseleave', hideGlow);
    document.addEventListener('visibilitychange', onVisibilityChange);
    window.addEventListener('blur', hideGlow);
  }

  function removeInteractionListeners() {
    document.removeEventListener(moveEventName, onMouseMove);
    document.removeEventListener('keydown', onKeyDown, true);
    document.removeEventListener('mouseleave', hideGlow);
    document.removeEventListener('visibilitychange', onVisibilityChange);
    window.removeEventListener('blur', hideGlow);
  }

  function onVisibilityChange() {
    if (document.hidden) hideGlow();
  }

  function enable() {
    if (enabled || !document.body) return;
    enabled = true;
    pointerSeen = false;
    needsInstantPosition = true;
    glow = document.createElement('div');
    glow.className = 'cursor-glow';
    glow.setAttribute('aria-hidden', 'true');
    document.body.classList.add('cursor-glow-enabled');
    document.body.appendChild(glow);
    addInteractionListeners();
  }

  function disable() {
    if (!enabled) return;
    enabled = false;
    removeInteractionListeners();
    stopAnimation();
    if (glow && glow.parentNode) glow.parentNode.removeChild(glow);
    glow = null;
    if (document.body) document.body.classList.remove('cursor-glow-enabled');
  }

  function updateCapability() {
    if (capability.matches) enable();
    else disable();
  }

  if (capability.addEventListener) {
    capability.addEventListener('change', updateCapability);
  } else if (capability.addListener) {
    capability.addListener(updateCapability);
  }

  updateCapability();
})();
