(() => {
  const motionQuery = window.matchMedia(
    '(hover: hover) and (pointer: fine) and (prefers-reduced-motion: no-preference)'
  );

  if (!motionQuery.matches) return;

  const glow = document.createElement('div');
  glow.className = 'cursor-glow';
  glow.setAttribute('aria-hidden', 'true');
  document.body.appendChild(glow);

  let x = 0;
  let y = 0;
  let targetX = 0;
  let targetY = 0;
  let animationFrame = 0;
  let isVisible = false;

  function updatePosition() {
    animationFrame = 0;
    x += (targetX - x) * 0.16;
    y += (targetY - y) * 0.16;
    glow.style.transform = `translate3d(${x}px, ${y}px, 0) translate(-50%, -50%)`;

    if (Math.abs(targetX - x) + Math.abs(targetY - y) > 0.5) {
      animationFrame = window.requestAnimationFrame(updatePosition);
    }
  }

  function hideGlow() {
    isVisible = false;
    glow.classList.remove('is-visible');
    if (animationFrame) {
      window.cancelAnimationFrame(animationFrame);
      animationFrame = 0;
    }
  }

  function followPointer(event) {
    if (!motionQuery.matches) {
      hideGlow();
      return;
    }
    if (event.pointerType && event.pointerType !== 'mouse' && event.pointerType !== 'pen') return;

    targetX = event.clientX;
    targetY = event.clientY;

    if (!isVisible) {
      x = targetX;
      y = targetY;
      isVisible = true;
      glow.classList.add('is-visible');
    }

    if (!animationFrame) animationFrame = window.requestAnimationFrame(updatePosition);
  }

  document.addEventListener('pointermove', followPointer, { passive: true });
  document.addEventListener('pointerout', (event) => {
    if (!event.relatedTarget) hideGlow();
  });
  window.addEventListener('blur', hideGlow);

  const handleMotionChange = () => {
    if (!motionQuery.matches) hideGlow();
  };
  if (motionQuery.addEventListener) {
    motionQuery.addEventListener('change', handleMotionChange);
  } else {
    motionQuery.addListener(handleMotionChange);
  }
})();
