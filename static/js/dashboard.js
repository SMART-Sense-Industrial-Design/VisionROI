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

function formatSeconds(value, decimals = 2) {
  const numeric = Number(value);
  if (!Number.isFinite(numeric)) {
    return '-';
  }
  return `${Math.max(0, numeric).toFixed(decimals)}s`;
}

function updateSummary(summary = {}, analytics = null) {
  const total = summary.total_cameras ?? 0;
  const online = summary.online_cameras ?? 0;
  const running = summary.inference_running ?? 0;
  const alerts = summary.alerts_last_hour ?? 0;
  const runningGroups = summary.running_groups ?? 0;
  const pageJobsRunning = summary.page_jobs_running ?? 0;
  const totalRoi = summary.total_roi ?? 0;
  const moduleTypes = summary.module_types ?? 0;

  if (summary.total_roi !== undefined) {
    setElementText('metric-total-roi-chip', totalRoi);
  }

  const offline = Math.max(total - online, 0);
  const onlineRate = total ? (online / total) * 100 : 0;
  const runningRate = total ? (running / Math.max(total, 1)) * 100 : 0;
  const alertDensity = total ? alerts / Math.max(total, 1) : 0;

  setElementText('summary-online-rate', `${onlineRate.toFixed(0)}%`);
  setElementText('summary-offline', offline);
  setElementText('summary-alert-density', alertDensity.toFixed(2));
  setElementText('chip-group-running', runningGroups);
  setElementText('chip-page-running', pageJobsRunning);
  setElementText('chip-inference-running', running);
  setElementText('summary-module-types', moduleTypes);
  setElementText('summary-total-roi', totalRoi);

  const cycleStats = analytics?.processing?.per_cycle;
  const intervalAverageDisplay = formatSeconds(summary.average_interval ?? 0, 2);
  setElementText('summary-average-interval', intervalAverageDisplay);

  if (cycleStats && cycleStats.count) {
    setElementText('summary-average-cycle', formatSeconds(cycleStats.average ?? 0, 2));
  } else if (analytics?.interval_health?.average_cycle) {
    setElementText('summary-average-cycle', formatSeconds(analytics.interval_health.average_cycle, 2));
  } else {
    setElementText('summary-average-cycle', '-');
  }
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

function updateCameraInsights(cameras = [], roiTotalOverride = null) {
  const stats = calculateCameraInsights(cameras);
  if (Number.isFinite(Number(roiTotalOverride))) {
    stats.totalRoi = Number(roiTotalOverride);
  }
  const averageFps = stats.fpsCount ? stats.fpsTotal / stats.fpsCount : 0;

  setElementText('metric-total-roi-chip', stats.totalRoi);
  setElementText('insight-average-fps', stats.fpsCount ? averageFps.toFixed(2) : '0.0');
}

function updateModuleSummary(modules = [], roiSummary = {}) {
  const container = document.getElementById('module-summary');
  if (!container) return;
  container.innerHTML = '';

  const moduleList = Array.isArray(modules) ? modules : [];
  const totalRoi = Number(roiSummary?.total ?? 0);

  if (!moduleList.length) {
    const empty = document.createElement('p');
    empty.className = 'analytics-card__empty';
    empty.textContent = 'ยังไม่มีโมดูลที่ถูกกำหนดใน ROI';
    container.appendChild(empty);
    return;
  }

  const overview = document.createElement('div');
  overview.className = 'module-summary__overview';
  overview.innerHTML = `
    <span>${moduleList.length} โมดูล</span>
    <span>${totalRoi} ROI</span>
  `;
  container.appendChild(overview);

  const grid = document.createElement('div');
  grid.className = 'module-summary__grid';

  moduleList.forEach((module) => {
    const shareValue = Number(module?.share ?? 0);
    const roiCount = Number(module?.roi ?? 0);
    const cameraCount = Number(module?.cameras ?? 0);

    const item = document.createElement('div');
    item.className = 'module-summary__item';

    const top = document.createElement('div');
    top.className = 'module-summary__top';

    const badge = document.createElement('span');
    badge.className = 'module-summary__badge';
    badge.textContent = `${cameraCount} กล้อง`;

    const share = document.createElement('span');
    share.className = 'module-summary__share';
    share.textContent = `${Number.isFinite(shareValue) ? shareValue.toFixed(1) : '0.0'}%`;

    top.append(badge, share);

    const title = document.createElement('p');
    title.className = 'module-summary__title';
    title.textContent = module?.name || 'ไม่ระบุโมดูล';

    const meta = document.createElement('p');
    meta.className = 'module-summary__meta';
    meta.textContent = `ROI ${roiCount} โซน`;

    const progress = document.createElement('div');
    progress.className = 'module-summary__progress';
    const fill = document.createElement('div');
    fill.className = 'module-summary__progress-fill';
    const width = Number.isFinite(shareValue) ? Math.max(8, Math.min(shareValue, 100)) : 0;
    fill.style.width = `${width.toFixed(0)}%`;
    progress.appendChild(fill);

    item.append(top, title, meta, progress);
    grid.appendChild(item);
  });

  container.appendChild(grid);
}

function renderIntervalIssues(examples = []) {
  const container = document.getElementById('interval-issues');
  if (!container) return;
  container.innerHTML = '';

  const items = Array.isArray(examples) ? examples : [];
  if (!items.length) {
    const empty = document.createElement('li');
    empty.className = 'interval-issues__empty';
    empty.textContent = 'ยังไม่พบความล่าช้า';
    container.appendChild(empty);
    return;
  }

  items.forEach((entry) => {
    const li = document.createElement('li');
    li.className = 'interval-issues__item';

    const header = document.createElement('div');
    header.className = 'interval-issues__header';

    const name = document.createElement('p');
    name.className = 'interval-issues__cam';
    name.textContent = entry?.cam_id || 'ไม่ทราบกล้อง';
    header.appendChild(name);

    if (entry?.group) {
      const badge = document.createElement('span');
      badge.className = 'interval-issues__badge';
      badge.textContent = entry.group;
      header.appendChild(badge);
    }

    li.appendChild(header);

    const intervalText = formatSeconds(entry?.interval ?? 0, 2);
    const averageText = formatSeconds(entry?.average_cycle ?? 0, 2);
    const latestText = formatSeconds(entry?.latest_cycle ?? 0, 2);

    const meta = document.createElement('p');
    meta.className = 'interval-issues__meta';
    meta.textContent = `Interval ${intervalText} · ใช้จริง ${averageText}`;

    const latest = document.createElement('p');
    latest.className = 'interval-issues__latest';
    latest.textContent = `รอบล่าสุด ${latestText}`;

    li.append(meta, latest);
    container.appendChild(li);
  });
}

function updateProcessingAnalytics(analytics = {}) {
  const processing = analytics?.processing || {};
  const cycleStats = processing.per_cycle || {};
  const roiStats = processing.per_roi || {};
  const intervalHealth = analytics?.interval_health || {};

  const cycleCount = cycleStats.count ?? 0;
  const roiCount = roiStats.count ?? 0;

  setElementText(
    'processing-cycle-average',
    cycleCount ? formatSeconds(cycleStats.average ?? 0, 2) : '-',
  );
  setElementText(
    'processing-cycle-max',
    cycleCount ? formatSeconds(cycleStats.max ?? 0, 2) : '-',
  );
  setElementText(
    'processing-roi-average',
    roiCount ? formatSeconds(roiStats.average ?? 0, 2) : '-',
  );
  setElementText(
    'processing-roi-p95',
    roiCount ? formatSeconds(roiStats.p95 ?? 0, 2) : '-',
  );

  const statusText = intervalHealth.status_text || 'รอข้อมูล';
  const metaText = intervalHealth.meta || '-';
  setElementText('processing-interval-status', statusText);
  setElementText('processing-interval-meta', metaText);

  renderIntervalIssues(intervalHealth.examples || []);
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
  const streamOffline = camera?.stream_offline || camera?.is_online === false;
  let statusBadgeClass;
  if (streamOffline && (camera?.inference_running || camera?.roi_running)) {
    statusBadgeClass = 'bg-danger';
  } else if (camera?.inference_running) {
    statusBadgeClass = 'bg-success';
  } else if (camera?.roi_running) {
    statusBadgeClass = 'bg-info text-dark';
  } else {
    statusBadgeClass = 'bg-secondary';
  }
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
    const analytics = data.analytics || {};

    updateSummary(data.summary, analytics);
    updateCameraInsights(data.cameras, analytics?.roi?.total);
    updateInsights(data.summary, data.cameras, data.alerts, alertAnalytics);
    updateCameraTable(data.cameras);
    updateAlerts(data.alerts);
    updateAlertSummary(data.alerts, alertAnalytics);
    updateGroupOverview(data.groups, data.cameras);
    updatePageOverview(data.page_jobs);
    updateModuleSummary(analytics.modules, analytics.roi);
    updateProcessingAnalytics(analytics);
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
