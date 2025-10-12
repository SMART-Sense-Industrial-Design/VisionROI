const DASHBOARD_ENDPOINT = '/api/dashboard';
const REFRESH_INTERVAL = 6000;
const ALERT_INTENSITY_REFERENCE = 20;

function formatDateTime(value) {
  if (!value) {
    return '-';
  }
  try {
    const date = new Date(value);
    if (Number.isNaN(date.getTime())) {
      return value;
    }
    return date.toLocaleString('th-TH', {
      hour12: false,
      year: 'numeric',
      month: 'short',
      day: 'numeric',
      hour: '2-digit',
      minute: '2-digit',
      second: '2-digit',
    });
  } catch (err) {
    return value;
  }
}

function parseDate(value) {
  if (!value) {
    return null;
  }
  if (value instanceof Date) {
    return Number.isNaN(value.getTime()) ? null : value;
  }
  const parsed = new Date(value);
  if (Number.isNaN(parsed.getTime())) {
    return null;
  }
  return parsed;
}

function getElapsedSeconds(value) {
  const date = parseDate(value);
  if (!date) {
    return null;
  }
  const diff = Date.now() - date.getTime();
  if (!Number.isFinite(diff)) {
    return null;
  }
  return diff / 1000;
}

function formatRelativeTime(value) {
  const elapsed = getElapsedSeconds(value);
  if (elapsed === null) {
    return null;
  }
  if (elapsed < 1) {
    return 'ไม่ถึง 1 วินาทีที่แล้ว';
  }
  if (elapsed < 60) {
    return `${Math.floor(elapsed)} วินาทีที่แล้ว`;
  }
  const minutes = Math.floor(elapsed / 60);
  if (minutes < 60) {
    return `${minutes} นาทีที่แล้ว`;
  }
  const hours = Math.floor(minutes / 60);
  if (hours < 24) {
    const remainingMinutes = minutes % 60;
    if (remainingMinutes > 0) {
      return `${hours} ชม. ${remainingMinutes} นาทีที่แล้ว`;
    }
    return `${hours} ชั่วโมงที่แล้ว`;
  }
  const days = Math.floor(hours / 24);
  if (days < 7) {
    const remainingHours = hours % 24;
    if (remainingHours > 0) {
      return `${days} วัน ${remainingHours} ชม. ที่แล้ว`;
    }
    return `${days} วันที่แล้ว`;
  }
  return formatDateTime(value);
}

function setMetric(id, value, decimals = 0) {
  const el = document.getElementById(id);
  if (!el) return;
  const display = typeof value === 'number' && !Number.isNaN(value)
    ? value.toFixed(decimals)
    : value ?? '-';
  el.textContent = display;
}

function setElementText(id, value) {
  const el = document.getElementById(id);
  if (el !== null && el !== undefined) {
    el.textContent = value;
  }
}

function setProgress(id, value) {
  const el = document.getElementById(id);
  if (!el) return;
  const percent = Math.max(0, Math.min(Number(value) || 0, 100));
  el.style.width = `${percent}%`;
}

function formatResolution(value) {
  if (!value) {
    return '-';
  }
  let width;
  let height;
  if (typeof value === 'string') {
    return value;
  }
  if (Array.isArray(value)) {
    [width, height] = value;
  } else if (typeof value === 'object') {
    ({ width, height } = value);
  }
  const w = Number(width);
  const h = Number(height);
  if (Number.isFinite(w) && Number.isFinite(h) && w > 0 && h > 0) {
    return `${w}×${h}`;
  }
  if (Number.isFinite(w) && w > 0) {
    return `${w}`;
  }
  if (Number.isFinite(h) && h > 0) {
    return `${h}`;
  }
  return '-';
}

function updateSummary(summary = {}) {
  setMetric('metric-total', summary.total_cameras ?? 0);
  setMetric('metric-online', summary.online_cameras ?? 0);
  setMetric('metric-running', summary.inference_running ?? 0);
  setMetric('metric-alerts', summary.alerts_last_hour ?? 0);
  setMetric('metric-interval', summary.average_interval ?? 0, 2);
  setMetric('metric-fps', summary.average_fps ?? 0, 2);
  setMetric('metric-groups', summary.total_groups ?? 0);
  setMetric('metric-pages', summary.page_jobs ?? 0);

  const total = summary.total_cameras ?? 0;
  const online = summary.online_cameras ?? 0;
  const running = summary.inference_running ?? 0;
  const alerts = summary.alerts_last_hour ?? 0;
  const runningGroups = summary.running_groups ?? 0;
  const pageJobsRunning = summary.page_jobs_running ?? 0;

  const offline = Math.max(total - online, 0);
  const onlineRate = total ? (online / total) * 100 : 0;
  const runningRate = total ? (running / Math.max(total, 1)) * 100 : 0;
  const alertDensity = total ? alerts / Math.max(total, 1) : 0;
  const queueSize = summary.inference_queue_size ?? 0;
  const queueCapacity = summary.inference_queue_capacity ?? 0;
  const queueUtilization = queueCapacity
    ? Math.min(Math.max((queueSize / queueCapacity) * 100, 0), 999)
    : 0;
  const latencyAverage = summary.average_latency;
  const latencyP95 = summary.latency_p95;
  const latencySamples = summary.latency_samples ?? 0;

  setElementText('summary-online-rate', `${onlineRate.toFixed(0)}%`);
  setElementText('summary-offline', offline);
  setElementText('summary-alert-density', alertDensity.toFixed(2));
  setElementText('metric-online-rate-meta', `${onlineRate.toFixed(0)}%`);
  setElementText('metric-running-ratio', `${runningRate.toFixed(0)}%`);
  setElementText('metric-alert-density', alertDensity.toFixed(2));
  setElementText('metric-groups-running', runningGroups);
  setElementText('metric-pages-running', pageJobsRunning);
  setElementText('chip-group-running', runningGroups);
  setElementText('chip-camera-running', running);
  setElementText('chip-page-running', pageJobsRunning);
}

function formatSeconds(value) {
  if (value === null || value === undefined) {
    return '-';
  }
  const num = Number(value);
  if (!Number.isFinite(num)) {
    return '-';
  }
  if (num === 0) {
    return '0.00s';
  }
  if (Math.abs(num) < 1) {
    return `${Math.round(num * 1000)}ms`;
  }
  return `${num.toFixed(2)}s`;
}

function formatSecondsFixed(value) {
  if (value === null || value === undefined) {
    return null;
  }
  const num = Number(value);
  if (!Number.isFinite(num)) {
    return null;
  }
  return num.toFixed(2);
}

function formatSecondsWithUnit(value) {
  const fixed = formatSecondsFixed(value);
  return fixed !== null ? `${fixed} วินาที` : null;
}

function formatRoiDetail(entry) {
  if (!entry) {
    return 'รอข้อมูล ROI';
  }
  const roiNameRaw = entry?.latest_roi_name ?? entry?.roi_name;
  const roiIdRaw = entry?.latest_roi_id ?? entry?.roi_id;
  const sourceRaw = entry?.latest_source ?? entry?.source;
  const camRaw = entry?.latest_cam_id ?? entry?.cam_id;

  const roiName = typeof roiNameRaw === 'string' ? roiNameRaw.trim() : '';
  const roiId =
    roiIdRaw !== null && roiIdRaw !== undefined
      ? String(roiIdRaw).trim()
      : '';
  const source = typeof sourceRaw === 'string' ? sourceRaw.trim() : '';
  const camId =
    camRaw !== null && camRaw !== undefined ? String(camRaw).trim() : '';
  const moduleRaw = entry?.module ?? entry?.name;
  const moduleLabel =
    typeof moduleRaw === 'string' && moduleRaw.trim()
      ? moduleRaw.trim()
      : '';

  const roiLabel = roiName || (roiId ? `ROI ${roiId}` : '');
  const locationLabel = source || (camId ? `กล้อง ${camId}` : '');

  const parts = [];
  if (roiLabel) {
    parts.push(roiLabel);
  }
  if (locationLabel) {
    parts.push(locationLabel);
  }
  if (moduleLabel) {
    parts.push(`โมดูล ${moduleLabel}`);
  }

  return parts.length ? parts.join(' · ') : 'รอข้อมูล ROI';
}

function calculateCameraInsights(cameras = []) {
  return cameras.reduce((acc, camera) => {
    const roiCount = Number(camera?.roi_count);
    if (Number.isFinite(roiCount)) {
      acc.totalRoi += roiCount;
    }

    const interval = Number(camera?.interval);
    if (Number.isFinite(interval)) {
      acc.minInterval = Math.min(acc.minInterval, interval);
      acc.intervalCount += 1;
    }

    const fps = Number(camera?.fps);
    if (Number.isFinite(fps)) {
      acc.maxFps = Math.max(acc.maxFps, fps);
      acc.fpsTotal += fps;
      acc.fpsCount += 1;
    }

    return acc;
  }, {
    totalRoi: 0,
    minInterval: Number.POSITIVE_INFINITY,
    intervalCount: 0,
    maxFps: Number.NEGATIVE_INFINITY,
    fpsTotal: 0,
    fpsCount: 0,
  });
}

function updateCameraInsights(cameras = []) {
  const stats = calculateCameraInsights(cameras);
  const minInterval = stats.intervalCount ? stats.minInterval : null;
  const maxFps = stats.fpsCount ? stats.maxFps : null;
  const averageFps = stats.fpsCount ? stats.fpsTotal / stats.fpsCount : 0;

  setElementText('metric-total-roi', stats.totalRoi);
  setElementText('metric-total-roi-chip', stats.totalRoi);
  setElementText('metric-min-interval', minInterval !== null ? `${minInterval.toFixed(2)}s` : '-');
  setElementText('metric-max-fps', maxFps !== null ? maxFps.toFixed(2) : '-');
  setElementText('insight-average-fps', stats.fpsCount ? averageFps.toFixed(2) : '0.0');
}

function createModuleBadge(name) {
  const badge = document.createElement('span');
  badge.className = 'module-badge';
  badge.textContent = name || 'ไม่ระบุ';
  return badge;
}

function createRoiRunStat(label, value, options = {}) {
  const wrapper = document.createElement('div');
  wrapper.className = 'roi-run-card__stat';

  const labelEl = document.createElement('p');
  labelEl.className = 'roi-run-card__stat-label';
  labelEl.textContent = label;
  wrapper.appendChild(labelEl);

  const valueEl = document.createElement('p');
  valueEl.className = 'roi-run-card__stat-value';
  const { decimals, valueClass, meta } = options ?? {};
  if (typeof value === 'string') {
    valueEl.textContent = value;
  } else if (value === null || value === undefined) {
    valueEl.textContent = '-';
  } else {
    const numeric = Number(value);
    if (Number.isFinite(numeric)) {
      if (Number.isInteger(decimals)) {
        valueEl.textContent = numeric.toFixed(decimals);
      } else {
        valueEl.textContent = numeric.toString();
      }
    } else {
      valueEl.textContent = '-';
    }
  }
  if (valueClass) {
    valueEl.classList.add(valueClass);
  }
  wrapper.appendChild(valueEl);

  if (meta) {
    const metaEl = document.createElement('p');
    metaEl.className = 'roi-run-card__stat-meta';
    metaEl.textContent = meta;
    wrapper.appendChild(metaEl);
  }

  return wrapper;
}

function normalizeQueueState(queue) {
  if (!queue || typeof queue !== 'object') {
    return null;
  }
  const stateRaw = queue.state ?? queue.status ?? 'unknown';
  const state = typeof stateRaw === 'string' ? stateRaw.toLowerCase() : 'unknown';
  const pending = Number(queue.pending);
  const expected = Number(queue.expected);
  const inflight = Number(queue.inflight_frames ?? queue.frames ?? queue.inflight);
  return {
    state,
    pending: Number.isFinite(pending) ? pending : null,
    expected: Number.isFinite(expected) ? expected : null,
    inflight: Number.isFinite(inflight) ? inflight : null,
    lastUpdated: queue.last_updated ?? queue.lastUpdated ?? null,
  };
}

function getQueueStateLabel(state) {
  switch (state) {
    case 'idle':
      return 'พร้อมทำงาน';
    case 'processing':
      return 'กำลังประมวลผล';
    case 'backlog':
      return 'คิวหนาแน่น';
    default:
      return 'ไม่ทราบ';
  }
}

function getQueueStateClass(state) {
  if (state === 'idle') {
    return 'roi-run-card__stat-value--queue-idle';
  }
  if (state === 'processing') {
    return 'roi-run-card__stat-value--queue-processing';
  }
  if (state === 'backlog') {
    return 'roi-run-card__stat-value--queue-backlog';
  }
  return 'roi-run-card__stat-value--queue-unknown';
}

function buildQueueStat(queue) {
  const normalized = normalizeQueueState(queue);
  if (!normalized) {
    return null;
  }

  const metaParts = [];
  if (normalized.pending !== null && normalized.expected !== null) {
    metaParts.push(`รอ ${normalized.pending}/${normalized.expected} งาน`);
  } else if (normalized.pending !== null) {
    metaParts.push(`รอ ${normalized.pending} งาน`);
  }
  if (normalized.inflight && normalized.inflight > 0) {
    metaParts.push(`${normalized.inflight} เฟรมค้าง`);
  }
  if (normalized.lastUpdated) {
    const relative = formatRelativeTime(normalized.lastUpdated);
    if (relative) {
      metaParts.push(`อัปเดต ${relative}`);
    }
  }

  return createRoiRunStat('สถานะคิว', getQueueStateLabel(normalized.state), {
    meta: metaParts.join(' · ') || null,
    valueClass: getQueueStateClass(normalized.state),
  });
}

function createRoiRunExtrema(label, entry, type) {
  const wrapper = document.createElement('div');
  wrapper.className = `roi-run-card__extrema-item roi-run-card__extrema-item--${type}`;

  const labelEl = document.createElement('p');
  labelEl.className = 'roi-run-card__extrema-label';
  labelEl.textContent = label;
  wrapper.appendChild(labelEl);

  const durationEl = document.createElement('p');
  durationEl.className = 'roi-run-card__extrema-duration';
  if (entry?.duration != null && Number.isFinite(Number(entry.duration))) {
    durationEl.textContent = formatSeconds(entry.duration);
  } else {
    durationEl.textContent = '-';
  }
  wrapper.appendChild(durationEl);

  const detailEl = document.createElement('p');
  detailEl.className = 'roi-run-card__extrema-detail';
  detailEl.textContent = entry ? formatRoiDetail(entry) : 'รอข้อมูล ROI';
  wrapper.appendChild(detailEl);

  return wrapper;
}

function renderSourceRunList(runs = []) {
  const container = document.getElementById('roi-run-list');
  if (!container) return;
  container.innerHTML = '';

  if (!Array.isArray(runs) || !runs.length) {
    const empty = document.createElement('div');
    empty.className = 'roi-run-list__empty';
    empty.textContent = 'รอข้อมูลการประมวลผลจากแหล่งสัญญาณ';
    container.appendChild(empty);
    return;
  }

  runs.forEach((run) => {
    const card = document.createElement('div');
    card.className = 'roi-run-card';

    const header = document.createElement('div');
    header.className = 'roi-run-card__header';

    const titleWrap = document.createElement('div');
    const titleEl = document.createElement('p');
    titleEl.className = 'roi-run-card__title';
    titleEl.textContent = run?.display_name || run?.cam_id || 'ไม่ทราบแหล่ง';
    titleWrap.appendChild(titleEl);

    const subtitleEl = document.createElement('p');
    subtitleEl.className = 'roi-run-card__subtitle';
    const subtitleParts = [];
    if (run?.cam_id) {
      subtitleParts.push(`รหัส ${run.cam_id}`);
    }
    if (run?.group) {
      subtitleParts.push(`กลุ่ม ${run.group}`);
    }
    if (run?.source && run.source !== run.cam_id) {
      const sourceText = String(run.source);
      subtitleParts.push(sourceText.length > 36 ? `${sourceText.slice(0, 36)}…` : sourceText);
    }
    subtitleEl.textContent = subtitleParts.join(' · ') || 'ไม่ระบุรายละเอียดเพิ่มเติม';
    titleWrap.appendChild(subtitleEl);

    header.appendChild(titleWrap);

    const timeEl = document.createElement('p');
    timeEl.className = 'roi-run-card__time';
    if (run?.timestamp) {
      const absolute = formatDateTime(run.timestamp) || '-';
      timeEl.textContent = `เฟรมล่าสุด ${absolute}`;
      const relative = formatRelativeTime(run.timestamp);
      if (relative && relative !== absolute) {
        const br = document.createElement('br');
        const relativeSpan = document.createElement('span');
        relativeSpan.textContent = relative;
        timeEl.appendChild(br);
        timeEl.appendChild(relativeSpan);
      }
    } else {
      timeEl.textContent = 'ยังไม่มีข้อมูลเฟรมล่าสุด';
    }
    header.appendChild(timeEl);

    card.appendChild(header);

    const stats = document.createElement('div');
    stats.className = 'roi-run-card__stats';
    stats.appendChild(createRoiRunStat('จำนวน ROI', run?.roi_count));
    const moduleCount = Array.isArray(run?.modules) ? run.modules.length : 0;
    stats.appendChild(createRoiRunStat('จำนวนโมดูล', moduleCount));
    const fpsValue = Number(
      run?.actual_fps !== undefined && run?.actual_fps !== null
        ? run.actual_fps
        : run?.fps
    );
    if (Number.isFinite(fpsValue)) {
      const fpsMetaParts = [];
      if (Number.isFinite(Number(run?.target_fps))) {
        fpsMetaParts.push(`เป้าหมาย ${Number(run.target_fps).toFixed(2)}`);
      }
      if (Number.isFinite(Number(run?.interval))) {
        fpsMetaParts.push(`Interval ${formatSeconds(run.interval)}`);
      }
      stats.appendChild(
        createRoiRunStat('FPS ล่าสุด', fpsValue, {
          decimals: 2,
          meta: fpsMetaParts.join(' · ') || null,
        }),
      );
    }

    const latencySeconds = Number(
      run?.frame_duration !== undefined && run?.frame_duration !== null
        ? run.frame_duration
        : run?.latency_latest !== undefined && run?.latency_latest !== null
          ? run.latency_latest
          : run?.total_duration
    );
    if (Number.isFinite(latencySeconds)) {
      const latencyMetaParts = [];
      if (Number.isFinite(Number(run?.latency_average))) {
        latencyMetaParts.push(`เฉลี่ย ${formatSeconds(run.latency_average)}`);
      }
      if (Number.isFinite(Number(run?.latency_max))) {
        latencyMetaParts.push(`สูงสุด ${formatSeconds(run.latency_max)}`);
      }
      stats.appendChild(
        createRoiRunStat('Latency รอบนี้', formatSeconds(latencySeconds), {
          meta: latencyMetaParts.join(' · ') || null,
        }),
      );
    }

    const queueStatEl = buildQueueStat(run?.queue);
    if (queueStatEl) {
      stats.appendChild(queueStatEl);
    }
    card.appendChild(stats);

    const modulesSection = document.createElement('div');
    modulesSection.className = 'roi-run-card__modules';
    const modulesLabel = document.createElement('p');
    modulesLabel.className = 'roi-run-card__stat-label';
    modulesLabel.textContent = 'โมดูลที่ทำงานในเฟรม';
    modulesSection.appendChild(modulesLabel);

    if (Array.isArray(run?.modules) && run.modules.length) {
      const badgeWrap = document.createElement('div');
      badgeWrap.className = 'module-badge-group';
      run.modules.forEach((moduleName) => {
        badgeWrap.appendChild(createModuleBadge(moduleName));
      });
      modulesSection.appendChild(badgeWrap);
    } else {
      const emptyModules = document.createElement('p');
      emptyModules.className = 'roi-run-card__modules-empty';
      emptyModules.textContent = 'ยังไม่มีข้อมูลโมดูล';
      modulesSection.appendChild(emptyModules);
    }
    card.appendChild(modulesSection);

    const extrema = document.createElement('div');
    extrema.className = 'roi-run-card__extrema';
    extrema.appendChild(createRoiRunExtrema('เร็วที่สุด', run?.fastest, 'fast'));
    extrema.appendChild(createRoiRunExtrema('ช้าที่สุด', run?.slowest, 'slow'));
    card.appendChild(extrema);

    container.appendChild(card);
  });
}

function createStatusBadge(detail) {
  const badge = document.createElement('span');
  badge.className = 'processing-status processing-status--unknown';
  const icon = document.createElement('i');
  let text = 'รอข้อมูล';
  let stateClass = 'processing-status--unknown';

  if (!detail?.roi_count) {
    stateClass = 'processing-status--idle';
    text = 'ไม่มี ROI';
    icon.className = 'bi bi-collection';
  } else if (!detail?.inference_running) {
    stateClass = 'processing-status--idle';
    text = 'หยุดทำงาน';
    icon.className = 'bi bi-pause-circle';
  } else if (detail?.meets_interval === true) {
    stateClass = 'processing-status--ok';
    text = 'รันทัน';
    icon.className = 'bi bi-check-circle';
  } else if (detail?.meets_interval === false) {
    stateClass = 'processing-status--risk';
    text = 'ช้ากว่า Interval';
    icon.className = 'bi bi-exclamation-triangle';
  } else {
    stateClass = 'processing-status--unknown';
    text = 'รอข้อมูล';
    icon.className = 'bi bi-question-circle';
  }

  badge.className = `processing-status ${stateClass}`;
  badge.append(icon, document.createTextNode(text));
  return badge;
}

function updateRoiSourceTable(details = []) {
  const tbody = document.getElementById('roi-source-table-body');
  if (!tbody) return;
  tbody.innerHTML = '';

  if (!Array.isArray(details) || !details.length) {
    const row = document.createElement('tr');
    const cell = document.createElement('td');
    cell.colSpan = 8;
    cell.className = 'text-center text-muted py-4';
    cell.textContent = 'ยังไม่มีการตั้งค่า ROI';
    row.appendChild(cell);
    tbody.appendChild(row);
    return;
  }

  details.forEach((detail) => {
    const row = document.createElement('tr');

    const sourceCell = document.createElement('td');
    const nameEl = document.createElement('p');
    nameEl.className = 'roi-source-name mb-0';
    nameEl.textContent = detail?.display_name || detail?.cam_id || '-';

    const metaEl = document.createElement('p');
    metaEl.className = 'roi-source-meta';
    const metaParts = [];
    if (detail?.cam_id) {
      metaParts.push(detail.cam_id);
    }
    if (detail?.source) {
      const src = String(detail.source);
      metaParts.push(src.length > 36 ? `${src.slice(0, 36)}…` : src);
    }
    metaEl.textContent = metaParts.join(' · ') || 'ไม่ระบุแหล่ง';
    sourceCell.append(nameEl, metaEl);

    const groupCell = document.createElement('td');
    groupCell.textContent = detail?.group || '-';

    const roiCell = document.createElement('td');
    roiCell.textContent = Number(detail?.roi_count ?? 0);

    const moduleCell = document.createElement('td');
    if (Array.isArray(detail?.modules) && detail.modules.length) {
      const wrap = document.createElement('div');
      wrap.className = 'module-badge-group';
      detail.modules.forEach((moduleName) => {
        wrap.appendChild(createModuleBadge(moduleName));
      });
      moduleCell.appendChild(wrap);
    } else {
      moduleCell.innerHTML = '<span class="text-muted small">ยังไม่กำหนด</span>';
    }
    if (Number(detail?.samples ?? 0) > 0) {
      const sampleMeta = document.createElement('div');
      sampleMeta.className = 'module-badge-meta';
      sampleMeta.textContent = `${detail.samples} รอบตัวอย่าง`;
      moduleCell.appendChild(sampleMeta);
    }

    const avgCell = document.createElement('td');
    const durationWrapper = document.createElement('div');
    durationWrapper.className = 'roi-source-duration';

    const durationValue = document.createElement('div');
    durationValue.className = 'roi-source-duration__value';

    const latestDurationSeconds = Number(detail?.latest_duration);
    const hasLatestDuration = Number.isFinite(latestDurationSeconds);
    if (hasLatestDuration) {
      durationValue.textContent = formatSeconds(latestDurationSeconds);
    } else {
      durationValue.textContent = 'รอข้อมูล';
    }
    durationWrapper.appendChild(durationValue);

    const latestCompleted = detail?.latest_completed_at;
    const intervalSeconds = Number(detail?.interval);
    const hasInterval = Number.isFinite(intervalSeconds) && intervalSeconds > 0;
    const relativeLabel = formatRelativeTime(latestCompleted);
    const latestTooltip = formatDateTime(latestCompleted);
    const elapsed = getElapsedSeconds(latestCompleted);

    if (hasLatestDuration && hasInterval) {
      const ratio = latestDurationSeconds / intervalSeconds;
      const percent = Number.isFinite(ratio) ? Math.max(ratio * 100, 0) : null;

      const barTrack = document.createElement('div');
      barTrack.className = 'roi-source-duration__bar';

      if (percent !== null) {
        const barFill = document.createElement('div');
        barFill.className = 'roi-source-duration__bar-fill';
        if (ratio > 1) {
          barFill.classList.add('roi-source-duration__bar-fill--over');
        }
        barFill.style.width = `${Math.min(percent, 100)}%`;
        barTrack.appendChild(barFill);
      }

      const ratioEl = document.createElement('div');
      ratioEl.className = 'roi-source-duration__ratio';
      if (percent !== null) {
        ratioEl.textContent = `ใช้เวลา ${percent.toFixed(0)}% ของ Interval ${formatSeconds(intervalSeconds)}`;
        if (ratio > 1) {
          ratioEl.classList.add('roi-source-duration__ratio--over');
        }
      } else {
        ratioEl.textContent = `Interval ${formatSeconds(intervalSeconds)}`;
      }

      durationWrapper.append(barTrack, ratioEl);
    } else if (hasInterval) {
      const ratioEl = document.createElement('div');
      ratioEl.className = 'roi-source-duration__ratio';
      ratioEl.textContent = `Interval ${formatSeconds(intervalSeconds)}`;
      durationWrapper.appendChild(ratioEl);
    }

    if (latestCompleted) {
      const metaEl = document.createElement('div');
      metaEl.className = 'roi-source-duration__meta';
      const tooltipParts = [];
      if (latestTooltip && latestTooltip !== '-') {
        tooltipParts.push(`เฟรมล่าสุดเมื่อ ${latestTooltip}`);
      }
      if (relativeLabel && relativeLabel !== latestTooltip) {
        tooltipParts.push(`(${relativeLabel})`);
      }

      if (hasInterval && Number.isFinite(elapsed)) {
        const staleThreshold = intervalSeconds * 3;
        const criticalThreshold = intervalSeconds * 6;
        if (elapsed > criticalThreshold) {
          metaEl.classList.add('roi-source-duration__meta--critical');
        } else if (elapsed > staleThreshold) {
          metaEl.classList.add('roi-source-duration__meta--stale');
        }
      } else if (Number.isFinite(elapsed) && elapsed > 300) {
        metaEl.classList.add('roi-source-duration__meta--stale');
      }

      if (relativeLabel) {
        metaEl.textContent = `เฟรมล่าสุด · ${relativeLabel}`;
      } else if (latestTooltip && latestTooltip !== '-') {
        metaEl.textContent = `เฟรมล่าสุด · ${latestTooltip}`;
      } else {
        metaEl.textContent = 'เฟรมล่าสุด · ไม่ทราบเวลา';
      }

      if (tooltipParts.length) {
        avgCell.title = tooltipParts.join(' ');
      } else {
        const intervalText = hasInterval
          ? ` เทียบ Interval ${formatSeconds(intervalSeconds)}`
          : '';
        avgCell.title =
          `เวลาที่ใช้ในการ inference ครบทุก ROI ในเฟรมล่าสุด (ตั้งแต่ ROI แรกจนถึง ROI สุดท้าย)${intervalText}`;
      }
      durationWrapper.appendChild(metaEl);
    } else {
      const intervalText = hasInterval
        ? ` เทียบ Interval ${formatSeconds(intervalSeconds)}`
        : '';
      avgCell.title =
        `เวลาที่ใช้ในการ inference ครบทุก ROI ในเฟรมล่าสุด (ตั้งแต่ ROI แรกจนถึง ROI สุดท้าย)${intervalText}`;
    }

    avgCell.appendChild(durationWrapper);

    const maxCell = document.createElement('td');
    maxCell.textContent = formatSeconds(detail?.max_duration);

    const statusCell = document.createElement('td');
    statusCell.appendChild(createStatusBadge(detail));

    row.append(
      sourceCell,
      groupCell,
      roiCell,
      moduleCell,
      avgCell,
      maxCell,
      statusCell,
    );
    tbody.appendChild(row);
  });
}

function updateRoiAnalytics(metrics = {}, summary = {}) {
  const totalRoi = metrics?.total_roi ?? summary?.total_roi ?? 0;
  setElementText('roi-total-count', totalRoi);
  setElementText('roi-module-count', metrics?.unique_modules ?? 0);
  setElementText('roi-source-count', metrics?.sources_with_roi ?? 0);
  setElementText('roi-running-count', summary?.inference_running ?? 0);

  renderSourceRunList(metrics?.source_runs);
  updateRoiSourceTable(metrics?.source_details);
}

function getAlertAnalytics(alerts = []) {
  const analytics = {
    topCamera: null,
    topCameraCount: 0,
    topCameraGroup: '',
    topLabel: null,
    topLabelCount: 0,
    activeCameraCount: 0,
    latestTimestamp: null,
  };

  if (!Array.isArray(alerts) || !alerts.length) {
    return analytics;
  }

  const cameraCounts = new Map();
  const cameraGroups = new Map();
  const labelCounts = new Map();
  let latest = null;

  alerts.forEach((alert) => {
    const camId = alert?.cam_id || 'ไม่ทราบกล้อง';
    cameraCounts.set(camId, (cameraCounts.get(camId) || 0) + 1);
    if (!cameraGroups.has(camId) && alert?.group) {
      cameraGroups.set(camId, alert.group);
    }

    if (Array.isArray(alert?.results)) {
      alert.results.forEach((item) => {
        const label = item?.text || item?.name;
        if (label) {
          labelCounts.set(label, (labelCounts.get(label) || 0) + 1);
        }
      });
    }

    if (alert?.timestamp) {
      const timestamp = new Date(alert.timestamp);
      if (!Number.isNaN(timestamp.getTime())) {
        if (!latest || timestamp > latest) {
          latest = timestamp;
        }
      }
    }
  });

  analytics.activeCameraCount = cameraCounts.size;

  if (cameraCounts.size) {
    const [cameraId, count] = [...cameraCounts.entries()].sort((a, b) => b[1] - a[1])[0];
    analytics.topCamera = cameraId;
    analytics.topCameraCount = count;
    analytics.topCameraGroup = cameraGroups.get(cameraId) || '';
  }

  if (labelCounts.size) {
    const [label, count] = [...labelCounts.entries()].sort((a, b) => b[1] - a[1])[0];
    analytics.topLabel = label;
    analytics.topLabelCount = count;
  }

  if (latest) {
    analytics.latestTimestamp = latest.toISOString();
  }

  return analytics;
}

function updateInsights(summary = {}, cameras = [], alerts = [], analytics = null) {
  const total = summary.total_cameras ?? 0;
  const online = summary.online_cameras ?? 0;
  const running = summary.inference_running ?? 0;
  const alertsLastHour = summary.alerts_last_hour ?? 0;

  const offline = Math.max(total - online, 0);
  const onlineRate = total ? (online / total) * 100 : 0;
  const runningRate = total ? (running / Math.max(total, 1)) * 100 : 0;

  setElementText('insight-online-value', `${online}/${total}`);
  setElementText('insight-offline', offline);
  setElementText('insight-online-rate', `${onlineRate.toFixed(0)}%`);
  setProgress('progress-online', onlineRate);

  setElementText('insight-running', running);
  setElementText('insight-running-rate', `${runningRate.toFixed(0)}%`);
  setProgress('progress-utilization', runningRate);

  setElementText('insight-alerts', alertsLastHour);

  const analyticsData = analytics ?? getAlertAnalytics(alerts);
  setElementText('insight-alert-active', analyticsData.activeCameraCount ?? 0);

  if (analyticsData.topCamera) {
    const suffix = analyticsData.topCameraGroup ? ` · ${analyticsData.topCameraGroup}` : '';
    setElementText(
      'insight-top-camera',
      `${analyticsData.topCamera}${suffix} (${analyticsData.topCameraCount} ครั้ง)`,
    );
  } else {
    setElementText('insight-top-camera', 'ยังไม่มีข้อมูล');
  }

  setElementText(
    'insight-alert-latest',
    analyticsData.latestTimestamp ? formatDateTime(analyticsData.latestTimestamp) : '-',
  );

  const alertIntensity = Math.min(alertsLastHour, ALERT_INTENSITY_REFERENCE)
    / ALERT_INTENSITY_REFERENCE * 100;
  setProgress('progress-alerts', alertIntensity);
}

function createSummaryItem({ title, value, meta }) {
  const wrapper = document.createElement('div');
  wrapper.className = 'alert-summary__item';

  const titleEl = document.createElement('p');
  titleEl.className = 'alert-summary__title';
  titleEl.textContent = title;

  const valueEl = document.createElement('p');
  valueEl.className = 'alert-summary__value';
  valueEl.textContent = value;

  wrapper.append(titleEl, valueEl);

  if (meta) {
    const metaEl = document.createElement('p');
    metaEl.className = 'alert-summary__meta';
    metaEl.textContent = meta;
    wrapper.appendChild(metaEl);
  }

  return wrapper;
}

function updateAlertSummary(alerts = [], analytics = null) {
  const container = document.getElementById('alert-summary');
  if (!container) return;
  container.innerHTML = '';

  if (!alerts.length) {
    const empty = document.createElement('div');
    empty.className = 'alert-summary__empty';
    empty.textContent = 'ยังไม่มีแจ้งเตือน';
    container.appendChild(empty);
    return;
  }

  const data = analytics ?? getAlertAnalytics(alerts);

  const items = [
    {
      title: 'กล้องที่แจ้งเตือนบ่อย',
      value: data.topCamera
        ? `${data.topCamera}${data.topCameraGroup ? ` · ${data.topCameraGroup}` : ''}`
        : 'ยังไม่มีข้อมูล',
      meta: data.topCamera ? `${data.topCameraCount} ครั้ง` : 'รอดูเหตุการณ์แรก',
    },
    {
      title: 'เหตุการณ์ที่พบบ่อย',
      value: data.topLabel ?? 'ยังไม่มีข้อมูล',
      meta: data.topLabel ? `${data.topLabelCount} ครั้ง` : 'รอดูเหตุการณ์แรก',
    },
    {
      title: 'อัปเดตล่าสุด',
      value: data.latestTimestamp ? formatDateTime(data.latestTimestamp) : '-',
      meta: data.activeCameraCount ? `เกิดจาก ${data.activeCameraCount} กล้อง` : 'รอดูเหตุการณ์แรก',
    },
  ];

  items.forEach((item) => {
    container.appendChild(createSummaryItem(item));
  });
}

function aggregateGroupsFromCameras(cameras = []) {
  const groups = new Map();
  cameras.forEach((camera) => {
    const key = camera?.group || 'ไม่ระบุกลุ่ม';
    const entry = groups.get(key) || {
      name: key,
      cameras: 0,
      running: 0,
      online: 0,
      roi: 0,
      fpsTotal: 0,
      fpsCount: 0,
      alerts: 0,
    };

    entry.cameras += 1;
    if (camera?.inference_running) {
      entry.running += 1;
    }

    const status = typeof camera?.status === 'string' ? camera.status.toLowerCase() : '';
    if (camera?.inference_running || camera?.roi_running || status.includes('online')) {
      entry.online += 1;
    }

    const roiCount = Number(camera?.roi_count);
    if (Number.isFinite(roiCount)) {
      entry.roi += roiCount;
    }

    const fps = Number(camera?.fps);
    if (Number.isFinite(fps)) {
      entry.fpsTotal += fps;
      entry.fpsCount += 1;
    }

    const alertsCount = Number(camera?.alerts_count);
    if (Number.isFinite(alertsCount)) {
      entry.alerts += alertsCount;
    }

    groups.set(key, entry);
  });

  return [...groups.values()].map((group) => {
    const averageFps = group.fpsCount ? group.fpsTotal / group.fpsCount : 0;
    return {
      name: group.name,
      cameras: group.cameras,
      running: group.running,
      online: group.online,
      roi: group.roi,
      average_fps: averageFps,
      alerts: group.alerts,
    };
  });
}

function normalizeGroupEntry(group) {
  const cameras = Number(group?.cameras ?? group?.total ?? 0);
  const running = Number(group?.running ?? 0);
  const online = Number(group?.online ?? 0);
  const roi = Number(group?.roi ?? group?.roi_total ?? 0);
  const averageFps = Number(group?.average_fps ?? group?.avg_fps ?? 0);
  const alerts = Number(group?.alerts ?? 0);
  return {
    name: group?.name || group?.group || 'ไม่ระบุกลุ่ม',
    cameras: Number.isFinite(cameras) ? cameras : 0,
    running: Number.isFinite(running) ? running : 0,
    online: Number.isFinite(online) ? online : 0,
    roi: Number.isFinite(roi) ? roi : 0,
    averageFps: Number.isFinite(averageFps) ? averageFps : 0,
    alerts: Number.isFinite(alerts) ? alerts : 0,
  };
}

function updateGroupOverview(groups = [], cameras = []) {
  const container = document.getElementById('group-overview');
  if (!container) return;
  container.innerHTML = '';

  const dataset = Array.isArray(groups) && groups.length
    ? groups.map(normalizeGroupEntry)
    : aggregateGroupsFromCameras(cameras);

  if (!dataset.length) {
    const empty = document.createElement('div');
    empty.className = 'group-overview__empty';
    empty.textContent = 'ยังไม่มีกลุ่มที่สร้างไว้';
    container.appendChild(empty);
    return;
  }

  const sorted = dataset.sort((a, b) => {
    if (b.running !== a.running) {
      return b.running - a.running;
    }
    return b.cameras - a.cameras;
  });

  sorted.forEach((group) => {
    const runningRate = group.cameras ? (group.running / group.cameras) * 100 : 0;

    const card = document.createElement('article');
    card.className = 'group-card';

    const header = document.createElement('div');
    header.className = 'group-card__header';

    const title = document.createElement('p');
    title.className = 'group-card__title mb-0';
    title.textContent = group.name;

    const badge = document.createElement('span');
    badge.className = 'group-card__badge';
    badge.textContent = `รันอยู่ ${group.running}/${group.cameras}`;

    header.append(title, badge);

    const meta = document.createElement('p');
    meta.className = 'group-card__meta';
    const roiText = Number.isFinite(group.roi) ? group.roi : 0;
    meta.textContent = `FPS เฉลี่ย ${group.averageFps.toFixed(1)} · ROI ${roiText} · แจ้งเตือน ${group.alerts}`;

    const progressTrack = document.createElement('div');
    progressTrack.className = 'group-card__progress-track';

    const progressFill = document.createElement('div');
    progressFill.className = 'group-card__progress-fill';
    progressFill.style.width = `${Math.max(0, Math.min(runningRate, 100)).toFixed(0)}%`;
    progressTrack.appendChild(progressFill);

    const progressMeta = document.createElement('div');
    progressMeta.className = 'group-card__progress-meta';
    progressMeta.innerHTML = `<span>โหลด</span><span>${runningRate.toFixed(0)}%</span>`;

    const footer = document.createElement('div');
    footer.className = 'group-card__footer';
    footer.innerHTML = `
      <span><i class="bi bi-wifi"></i>ออนไลน์ ${group.online}</span>
      <span><i class="bi bi-camera-video"></i> ${group.cameras} กล้อง</span>
      <span><i class="bi bi-bell"></i> ${group.alerts}</span>
    `;

    card.append(header, meta, progressTrack, progressMeta, footer);
    container.appendChild(card);
  });
}

function renderPageCard(job) {
  const card = document.createElement('article');
  card.className = 'page-card';

  const header = document.createElement('div');
  header.className = 'page-card__header';

  const titleWrap = document.createElement('div');
  titleWrap.className = 'page-card__titles';

  const title = document.createElement('p');
  title.className = 'page-card__title';
  title.textContent = job.title || job.cam_id || 'งาน Page';

  const subtitle = document.createElement('p');
  subtitle.className = 'page-card__subtitle';
  const groupText = job.group ? `กลุ่ม ${job.group}` : 'ไม่มีกลุ่ม';
  subtitle.textContent = `${groupText} · ${job.cam_id || '-'}`;

  titleWrap.append(title, subtitle);

  const badge = document.createElement('span');
  const isRunning = Boolean(job.inference_running);
  badge.className = `page-card__badge ${isRunning ? 'page-card__badge--running' : 'page-card__badge--idle'}`;
  badge.textContent = isRunning ? 'กำลังรัน' : (job.status || 'พร้อม');

  header.append(titleWrap, badge);

  const metrics = document.createElement('div');
  metrics.className = 'page-card__metrics';

  const interval = Number(job.interval);
  const fps = Number(job.fps);
  const alerts = Number(job.alerts_count ?? 0);

  const metricData = [
    { label: 'Interval', value: Number.isFinite(interval) ? `${interval.toFixed(2)}s` : '-' },
    { label: 'FPS', value: Number.isFinite(fps) ? fps.toFixed(2) : '-' },
    { label: 'แจ้งเตือน', value: alerts },
  ];

  metricData.forEach((item) => {
    const metric = document.createElement('div');
    metric.className = 'page-card__metric';
    const label = document.createElement('span');
    label.className = 'page-card__metric-label';
    label.textContent = item.label;
    const value = document.createElement('span');
    value.className = 'page-card__metric-value';
    value.textContent = item.value ?? '-';
    metric.append(label, value);
    metrics.appendChild(metric);
  });

  const output = document.createElement('div');
  output.className = 'page-card__output';
  const outputLabel = document.createElement('span');
  outputLabel.className = 'page-card__output-label';
  outputLabel.textContent = 'ผลล่าสุด';
  const outputValue = document.createElement('span');
  outputValue.className = 'page-card__output-value';
  outputValue.textContent = job.last_output || '-';
  outputValue.title = job.last_output || '';
  output.append(outputLabel, outputValue);

  const footer = document.createElement('div');
  footer.className = 'page-card__footer';
  footer.innerHTML = `
    <span><i class="bi bi-clock-history"></i> ${formatDateTime(job.last_activity)}</span>
    <span><i class="bi bi-bell"></i> ${alerts} ครั้ง</span>
  `;

  card.append(header, metrics, output, footer);
  return card;
}

function updatePageOverview(pageJobs = []) {
  const container = document.getElementById('page-overview');
  if (!container) return;
  container.innerHTML = '';

  if (!Array.isArray(pageJobs) || !pageJobs.length) {
    const empty = document.createElement('div');
    empty.className = 'page-overview__empty';
    empty.textContent = 'ยังไม่มีงาน Page Inference';
    container.appendChild(empty);
    return;
  }

  pageJobs.forEach((job) => {
    container.appendChild(renderPageCard(job));
  });
}

function createCameraRow(camera) {
  const tr = document.createElement('tr');

  const firstCell = document.createElement('td');
  const camId = document.createElement('div');
  camId.className = 'fw-semibold';
  camId.textContent = camera.cam_id ?? '-';
  const camMeta = document.createElement('div');
  camMeta.className = 'text-muted small';
  const backend = camera.backend || '-';
  const roiInfo = camera.roi_count ? ` · ROI ${camera.roi_count}` : '';
  camMeta.textContent = `${backend}${roiInfo}`;
  firstCell.append(camId, camMeta);

  const secondCell = document.createElement('td');
  const camName = document.createElement('div');
  camName.className = 'fw-semibold';
  camName.textContent = camera.name || '-';
  const camSource = document.createElement('div');
  camSource.className = 'text-muted small';
  camSource.textContent = camera.source || '-';
  secondCell.append(camName, camSource);

  const resolutionCell = document.createElement('td');
  resolutionCell.textContent = formatResolution(camera.resolution);

  const statusCell = document.createElement('td');
  const statusBadge = document.createElement('span');
  const statusBadgeClass = camera.inference_running
    ? 'bg-success'
    : camera.roi_running
      ? 'bg-info text-dark'
      : 'bg-secondary';
  statusBadge.className = `badge ${statusBadgeClass} bg-opacity-75`;
  statusBadge.textContent = camera.status || '-';
  statusCell.appendChild(statusBadge);

  const groupCell = document.createElement('td');
  groupCell.textContent = camera.group || '-';

  const intervalCell = document.createElement('td');
  intervalCell.textContent = camera.interval ? `${camera.interval.toFixed(2)}s` : '-';

  const fpsCell = document.createElement('td');
  fpsCell.textContent = camera.fps ? camera.fps.toFixed(2) : '-';

  const outputCell = document.createElement('td');
  const outputWrapper = document.createElement('div');
  outputWrapper.className = 'small text-truncate';
  outputWrapper.style.maxWidth = '180px';
  outputWrapper.textContent = camera.last_output || '-';
  outputCell.appendChild(outputWrapper);

  const alertCell = document.createElement('td');
  const alertBadge = document.createElement('span');
  alertBadge.className = 'badge bg-primary-subtle text-primary';
  alertBadge.textContent = String(camera.alerts_count ?? 0);
  alertCell.appendChild(alertBadge);

  const activityCell = document.createElement('td');
  activityCell.textContent = formatDateTime(camera.last_activity);

  tr.append(
    firstCell,
    secondCell,
    resolutionCell,
    statusCell,
    groupCell,
    intervalCell,
    fpsCell,
    outputCell,
    alertCell,
    activityCell,
  );

  return tr;
}

function updateCameraTable(cameras = []) {
  const tbody = document.getElementById('camera-table-body');
  if (!tbody) return;
  tbody.innerHTML = '';
  if (!cameras.length) {
    const emptyRow = document.createElement('tr');
    emptyRow.innerHTML = '<td colspan="10" class="text-center text-muted py-4">ไม่มีข้อมูลกล้อง</td>';
    tbody.appendChild(emptyRow);
    return;
  }
  cameras.forEach((camera) => {
    tbody.appendChild(createCameraRow(camera));
  });
}

function renderAlertItem(alert) {
  const li = document.createElement('li');
  li.className = 'timeline__item';

  const icon = document.createElement('div');
  icon.className = 'timeline__icon';
  const iconInner = document.createElement('i');
  iconInner.className = 'bi bi-bell-fill';
  icon.appendChild(iconInner);

  const content = document.createElement('div');
  content.className = 'timeline__content';

  const title = document.createElement('p');
  title.className = 'timeline__title mb-1';
  const groupSuffix = alert.group ? ` · ${alert.group}` : '';
  title.textContent = `${alert.cam_id || 'ไม่ทราบกล้อง'}${groupSuffix}`;

  const meta = document.createElement('p');
  meta.className = 'timeline__meta mb-1';
  const sourceText = alert.source || 'ไม่ระบุแหล่ง';
  meta.textContent = `${formatDateTime(alert.timestamp)} · ${sourceText}`;

  const details = document.createElement('div');
  details.className = 'small text-muted';
  const resultsPreview = (alert.results || [])
    .map((item) => item.text || item.name || '')
    .filter(Boolean);
  details.textContent = resultsPreview.length
    ? resultsPreview.join(', ')
    : 'ไม่มีรายละเอียดเพิ่มเติม';

  content.append(title, meta, details);
  li.append(icon, content);

  return li;
}

function updateAlerts(alerts = []) {
  const container = document.getElementById('alert-timeline');
  if (!container) return;
  container.innerHTML = '';
  if (!alerts.length) {
    const empty = document.createElement('li');
    empty.className = 'timeline__empty';
    empty.textContent = 'ยังไม่มีแจ้งเตือน';
    container.appendChild(empty);
    return;
  }
  alerts.slice().reverse().forEach((alert) => {
    container.appendChild(renderAlertItem(alert));
  });
}

function updateStreams(cameras = []) {
  const container = document.getElementById('stream-grid');
  if (!container) return;
  container.innerHTML = '';
  const running = cameras.filter((camera) => camera.inference_running && camera.snapshot_url);
  if (!running.length) {
    const empty = document.createElement('div');
    empty.className = 'stream-grid__empty text-muted';
    empty.textContent = 'ยังไม่มีกล้องที่กำลังประมวลผล';
    container.appendChild(empty);
    return;
  }
  running.forEach((camera) => {
    const item = document.createElement('div');
    item.className = 'stream-grid__item';

    const link = document.createElement('a');
    link.className = 'text-decoration-none text-white';
    link.href = `/inference?cam_id=${encodeURIComponent(camera.cam_id)}`;

    const img = document.createElement('img');
    img.loading = 'lazy';
    img.src = camera.snapshot_url;
    img.alt = `snapshot-${camera.cam_id}`;

    const caption = document.createElement('div');
    caption.className = 'stream-grid__caption';

    const title = document.createElement('p');
    title.className = 'stream-grid__title mb-1';
    title.textContent = camera.cam_id || '-';

    const meta = document.createElement('p');
    meta.className = 'stream-grid__meta mb-0';
    const fpsText = camera.fps ? camera.fps.toFixed(2) : '-';
    meta.textContent = `${camera.group || 'ไม่มีข้อมูลกลุ่ม'} · FPS ${fpsText}`;

    caption.append(title, meta);
    link.append(img, caption);
    item.appendChild(link);
    container.appendChild(item);
  });
}

function updateGeneratedAt(timestamp) {
  const el = document.getElementById('dashboard-updated');
  if (el) {
    el.textContent = formatDateTime(timestamp);
  }
}

async function loadDashboard() {
  try {
    const response = await fetch(DASHBOARD_ENDPOINT, {
      headers: {
        'Cache-Control': 'no-cache',
      },
    });
    if (!response.ok) {
      throw new Error(`HTTP ${response.status}`);
    }
    const data = await response.json();
    const alertAnalytics = getAlertAnalytics(data.alerts);

    updateSummary(data.summary);
    updateCameraInsights(data.cameras);
    updateRoiAnalytics(data.roi_metrics, data.summary);
    updateInsights(data.summary, data.cameras, data.alerts, alertAnalytics);
    updateCameraTable(data.cameras);
    updateAlerts(data.alerts);
    updateAlertSummary(data.alerts, alertAnalytics);
    updateGroupOverview(data.groups, data.cameras);
    updatePageOverview(data.page_jobs);
    updateStreams(data.cameras);
    updateGeneratedAt(data.generated_at);
  } catch (error) {
    console.error('โหลดข้อมูลแดชบอร์ดไม่สำเร็จ:', error);
  }
}

document.addEventListener('DOMContentLoaded', () => {
  setElementText('chip-refresh', Math.round(REFRESH_INTERVAL / 1000));
  loadDashboard();
  setInterval(loadDashboard, REFRESH_INTERVAL);
});
