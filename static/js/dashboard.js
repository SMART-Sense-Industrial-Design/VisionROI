const DASHBOARD_ENDPOINT = '/api/dashboard';
const REFRESH_INTERVAL = 6000;

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

function updateSummary(summary = {}) {
  setMetric('metric-total', summary.total_cameras ?? 0);
  setMetric('metric-online', summary.online_cameras ?? 0);
  setMetric('metric-running', summary.inference_running ?? 0);
  setMetric('metric-alerts', summary.alerts_last_hour ?? 0);
  setMetric('metric-interval', summary.average_interval ?? 0, 2);
  setMetric('metric-fps', summary.average_fps ?? 0, 2);
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
    updateSummary(data.summary);
    updateCameraTable(data.cameras);
    updateAlerts(data.alerts);
    updateStreams(data.cameras);
    updateGeneratedAt(data.generated_at);
  } catch (error) {
    console.error('โหลดข้อมูลแดชบอร์ดไม่สำเร็จ:', error);
  }
}

document.addEventListener('DOMContentLoaded', () => {
  loadDashboard();
  setInterval(loadDashboard, REFRESH_INTERVAL);
});
