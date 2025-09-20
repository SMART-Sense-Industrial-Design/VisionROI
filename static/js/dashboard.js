const DASHBOARD_ENDPOINT = '/api/dashboard';
const REFRESH_INTERVAL = 6000;
const WEEKDAY_NAMES = ['จันทร์', 'อังคาร', 'พุธ', 'พฤหัสบดี', 'ศุกร์', 'เสาร์', 'อาทิตย์'];

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
  const intervalDisplay = camera.interval ? `${camera.interval.toFixed(2)}s` : '-';
  const fpsDisplay = camera.fps ? camera.fps.toFixed(2) : '-';
  const statusBadgeClass = camera.inference_running
    ? 'bg-success'
    : camera.roi_running
      ? 'bg-info text-dark'
      : 'bg-secondary';
  tr.innerHTML = `
    <td>
      <div class="fw-semibold">${camera.cam_id}</div>
      <div class="text-muted small">${camera.backend || '-'}${camera.roi_count ? ` · ROI ${camera.roi_count}` : ''}</div>
    </td>
    <td>
      <div class="fw-semibold">${camera.name || '-'}</div>
      <div class="text-muted small">${camera.source || '-'}</div>
    </td>
    <td>
      <span class="badge ${statusBadgeClass} bg-opacity-75">${camera.status}</span>
    </td>
    <td>${camera.group || '-'}</td>
    <td>${intervalDisplay}</td>
    <td>${fpsDisplay}</td>
    <td>
      <div class="small text-truncate" style="max-width: 180px;">${camera.last_output || '-'}</div>
    </td>
    <td>
      <span class="badge bg-primary-subtle text-primary">${camera.alerts_count ?? 0}</span>
    </td>
    <td>${formatDateTime(camera.last_activity)}</td>
  `;
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
  const resultsPreview = (alert.results || []).map((item) => item.text || item.name || '').filter(Boolean);
  li.innerHTML = `
    <div class="timeline__icon">
      <i class="bi bi-bell-fill"></i>
    </div>
    <div class="timeline__content">
      <p class="timeline__title mb-1">${alert.cam_id || 'ไม่ทราบกล้อง'}${alert.group ? ` · ${alert.group}` : ''}</p>
      <p class="timeline__meta mb-1">${formatDateTime(alert.timestamp)} · ${alert.source || 'ไม่ระบุแหล่ง'}</p>
      <div class="small text-muted">${resultsPreview.length ? resultsPreview.join(', ') : 'ไม่มีรายละเอียดเพิ่มเติม'}</div>
    </div>
  `;
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

function formatWeekdays(weekdays = []) {
  if (!weekdays.length) {
    return 'ทุกวัน';
  }
  return weekdays
    .map((index) => WEEKDAY_NAMES[index] || index)
    .join(', ');
}

function renderSchedule(schedule) {
  const card = document.createElement('div');
  card.className = 'schedule-card';
  const statusText = schedule.enabled ? 'เปิดใช้งาน' : 'ปิดอยู่';
  const statusClass = schedule.enabled ? 'text-success' : 'text-muted';
  card.innerHTML = `
    <div class="schedule-card__header">
      <p class="schedule-card__title mb-0">${schedule.label || schedule.cam_id}</p>
      <span class="schedule-card__status ${statusClass}">${schedule.is_running ? 'กำลังทำงาน' : statusText}</span>
    </div>
    <div class="schedule-card__meta">${schedule.start_time} - ${schedule.end_time} · ${formatWeekdays(schedule.weekdays)}</div>
    <div class="schedule-card__meta">กล้อง: ${schedule.cam_id}${schedule.group ? ` · กลุ่ม ${schedule.group}` : ''}</div>
  `;
  return card;
}

function updateSchedules(schedules = []) {
  const container = document.getElementById('schedule-list');
  if (!container) return;
  container.innerHTML = '';
  if (!schedules.length) {
    const empty = document.createElement('div');
    empty.className = 'schedule-list__empty text-muted';
    empty.textContent = 'ยังไม่มีการตั้งตารางงาน';
    container.appendChild(empty);
    return;
  }
  schedules.forEach((schedule) => {
    container.appendChild(renderSchedule(schedule));
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
    item.innerHTML = `
      <a href="/inference?cam_id=${encodeURIComponent(camera.cam_id)}" class="text-decoration-none text-white">
        <img src="${camera.snapshot_url}" alt="snapshot-${camera.cam_id}" loading="lazy">
        <div class="stream-grid__caption">
          <p class="stream-grid__title mb-1">${camera.cam_id}</p>
          <p class="stream-grid__meta mb-0">${camera.group || 'ไม่มีข้อมูลกลุ่ม'} · FPS ${camera.fps ? camera.fps.toFixed(2) : '-'}</p>
        </div>
      </a>
    `;
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
    updateSchedules(data.schedules);
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
