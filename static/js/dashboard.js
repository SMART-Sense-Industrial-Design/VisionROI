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

function updateSummary(summary = {}) {
  setMetric('metric-total', summary.total_cameras ?? 0);
  setMetric('metric-online', summary.online_cameras ?? 0);
  setMetric('metric-running', summary.inference_running ?? 0);
  setMetric('metric-alerts', summary.alerts_last_hour ?? 0);
  setMetric('metric-interval', summary.average_interval ?? 0, 2);
  setMetric('metric-processing', summary.average_processing_ms ?? 0, 0);

  const total = summary.total_cameras ?? 0;
  const online = summary.online_cameras ?? 0;
  const running = summary.inference_running ?? 0;
  const alerts = summary.alerts_last_hour ?? 0;

  const offline = Math.max(total - online, 0);
  const onlineRate = total ? (online / total) * 100 : 0;
  const runningRate = total ? (running / Math.max(total, 1)) * 100 : 0;
  const alertDensity = total ? alerts / Math.max(total, 1) : 0;

  setElementText('summary-online-rate', `${onlineRate.toFixed(0)}%`);
  setElementText('summary-offline', offline);
  setElementText('summary-alert-density', alertDensity.toFixed(2));
  setElementText('metric-online-rate-meta', `${onlineRate.toFixed(0)}%`);
  setElementText('metric-running-ratio', `${runningRate.toFixed(0)}%`);
  setElementText('metric-alert-density', alertDensity.toFixed(2));
  setElementText('chip-running', running);
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

    const processing = Number(camera?.processing_time_ms);
    if (Number.isFinite(processing)) {
      acc.maxProcessing = Math.max(acc.maxProcessing, processing);
      acc.processingTotal += processing;
      acc.processingCount += 1;
    }

    return acc;
  }, {
    totalRoi: 0,
    minInterval: Number.POSITIVE_INFINITY,
    intervalCount: 0,
    maxProcessing: Number.NEGATIVE_INFINITY,
    processingTotal: 0,
    processingCount: 0,
  });
}

function updateCameraInsights(cameras = []) {
  const stats = calculateCameraInsights(cameras);
  const minInterval = stats.intervalCount ? stats.minInterval : null;
  const maxProcessing = stats.processingCount ? stats.maxProcessing : null;
  const averageProcessing = stats.processingCount
    ? stats.processingTotal / stats.processingCount
    : 0;

  setElementText('metric-total-roi', stats.totalRoi);
  setElementText('metric-total-roi-chip', stats.totalRoi);
  setElementText('metric-min-interval', minInterval !== null ? `${minInterval.toFixed(2)}s` : '-');
  setElementText(
    'metric-max-processing',
    maxProcessing !== null ? `${maxProcessing.toFixed(0)} ms` : '-',
  );
  setElementText(
    'insight-average-processing',
    stats.processingCount ? averageProcessing.toFixed(0) : '0',
  );
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

function updateGroupOverview(cameras = []) {
  const container = document.getElementById('group-overview');
  if (!container) return;
  container.innerHTML = '';

  if (!cameras.length) {
    const empty = document.createElement('div');
    empty.className = 'group-overview__empty';
    empty.textContent = 'ยังไม่มีกลุ่มที่สร้างไว้';
    container.appendChild(empty);
    return;
  }

  const groups = new Map();

  cameras.forEach((camera) => {
    const key = camera?.group || 'ไม่ระบุกลุ่ม';
    const entry = groups.get(key) || {
      name: key,
      cameras: 0,
      running: 0,
      online: 0,
      roi: 0,
      processingTotal: 0,
      processingCount: 0,
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

    const processing = Number(camera?.processing_time_ms);
    if (Number.isFinite(processing)) {
      entry.processingTotal += processing;
      entry.processingCount += 1;
    }

    groups.set(key, entry);
  });

  const sorted = [...groups.values()].sort((a, b) => {
    if (b.running !== a.running) {
      return b.running - a.running;
    }
    return b.cameras - a.cameras;
  });

  sorted.forEach((group) => {
    const runningRate = group.cameras ? (group.running / group.cameras) * 100 : 0;
    const averageProcessing = group.processingCount
      ? group.processingTotal / group.processingCount
      : 0;

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
    meta.textContent = `เวลาเฉลี่ย ${averageProcessing.toFixed(0)} ms · ROI ${group.roi}`;

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
    `;

    card.append(header, meta, progressTrack, progressMeta, footer);
    container.appendChild(card);
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

  const processingCell = document.createElement('td');
  const processingValue = Number(camera?.processing_time_ms);
  processingCell.textContent = Number.isFinite(processingValue)
    ? `${processingValue.toFixed(0)}`
    : '-';

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
    statusCell,
    groupCell,
    intervalCell,
    processingCell,
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
    emptyRow.innerHTML = '<td colspan="9" class="text-center text-muted py-4">ไม่มีข้อมูลกล้อง</td>';
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
    const processingValue = Number(camera?.processing_time_ms);
    const processingText = Number.isFinite(processingValue)
      ? `${processingValue.toFixed(0)} ms`
      : '-';
    meta.textContent = `${camera.group || 'ไม่มีข้อมูลกลุ่ม'} · เวลา ${processingText}`;

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
    updateInsights(data.summary, data.cameras, data.alerts, alertAnalytics);
    updateCameraTable(data.cameras);
    updateAlerts(data.alerts);
    updateAlertSummary(data.alerts, alertAnalytics);
    updateGroupOverview(data.cameras);
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
