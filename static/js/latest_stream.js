// Latest-only HTTP puller: ดึงรูปทีละใบและยกเลิกคำขอเก่าทันที => ไม่มีคิวค้าง
function startLatestOnlyStream(imgEl, camId, opts = {}) {
  const endpoint = opts.endpoint || `/ws_snapshot/${camId}`;
  const targetFps = Number(opts.fps ?? 10);
  const frameInterval = targetFps > 0 ? (1000 / targetFps) : 0;

  let running = true;
  let controller = null;
  let lastUrl = null;
  let lastTick = 0;

  function revoke(url) {
    try { if (url && url.startsWith('blob:')) URL.revokeObjectURL(url); } catch(e) {}
  }

  async function tick(now) {
    if (!running) return;
    if (frameInterval && now - lastTick < frameInterval) {
      return requestAnimationFrame(tick);
    }
    lastTick = now;

    if (controller) controller.abort();         // ตัดคิวเก่าออก
    controller = new AbortController();

    try {
      const r = await fetch(`${endpoint}?_=${Date.now()}`, {
        signal: controller.signal,
        cache: 'no-store',
        headers: { 'Accept': 'image/jpeg' }
      });
      if (r.ok) {
        const blob = await r.blob();
        revoke(lastUrl);
        lastUrl = URL.createObjectURL(blob);
        imgEl.src = lastUrl;                    // โชว์เฟรมล่าสุด
      }
    } catch (_) {
      // abort หรือเน็ตช้า -> ข้ามไปเฟรมถัดไป
    } finally {
      requestAnimationFrame(tick);
    }
  }

  const rafId = requestAnimationFrame(tick);

  return {
    stop() {
      running = false;
      try { if (controller) controller.abort(); } catch(_) {}
      try { if (rafId) cancelAnimationFrame(rafId); } catch(_) {}
      revoke(lastUrl);
    }
  };
}
