"use strict";

function reportRuntimeError(message) {
  const detail = message || "未知错误";
  const notice = document.getElementById("replay-notice-text");
  const status = document.getElementById("replay-source-status");
  if (notice) notice.textContent = `页面脚本错误：${detail}`;
  if (status) status.textContent = "页面脚本错误";
  console.error(detail);
}

window.addEventListener("error", (event) => {
  reportRuntimeError(event.error?.message || event.message);
});
window.addEventListener("unhandledrejection", (event) => {
  reportRuntimeError(event.reason?.message || String(event.reason));
});

const DECISION_DT = 0.05;
const POT_CENTER = { x: 1.8, y: 0.0 };
const POT_RADIUS = 0.8;

const STAGES = {
  low: { label: "Low", source: "multi_low", maxActive: 3 },
  realistic: { label: "Realistic", source: "multi_realistic", maxActive: 4 },
  hard: { label: "Hard", source: "multi_hard", maxActive: 6 },
  extreme: { label: "Extreme", source: "multi_extreme", maxActive: 8 },
};

const REPLAY_SOURCES = {
  ours: {
    label: "V12 H2 + LSTM-PPO 残差",
    directory: "data/ours",
    badge: "V12 PPO seed0 · eval seed100",
  },
  horizon2: {
    label: "Horizon-2",
    directory: "data/horizon2",
    badge: "Horizon-2 · eval seed100",
  },
  no_attention: {
    label: "消融 · No attention",
    directory: "data/no_attention",
    badge: "No attention seed0 · eval seed100",
  },
  no_carry: {
    label: "消融 · No carry",
    directory: "data/no_carry",
    badge: "No carry seed0 · eval seed100",
  },
  no_prediction: {
    label: "消融 · No prediction",
    directory: "data/no_prediction",
    badge: "No prediction seed0 · eval seed100",
  },
  no_service_reward: {
    label: "消融 · No service reward",
    directory: "data/no_service_reward",
    badge: "No service reward seed0 · eval seed100",
  },
  no_residual: {
    label: "消融 · No residual",
    directory: "data/no_residual",
    badge: "No residual seed0 · eval seed100",
  },
};

const VIEW_TITLES = {
  live: "实时作业",
  algorithm: "算法配置",
  experiments: "实验评估",
  diagnostics: "系统诊断",
};

// Values are copied from runs/v12_small_paper_suite/paper_tables/main_results.csv.
const EVALUATION_DATA = {
  coverage: {
    title: "覆盖率主结果",
    suffix: "",
    ours: [0.917, 0.901, 0.870, 0.855],
    base: [0.919, 0.896, 0.870, 0.816],
    oursStd: [0.010, 0.016, 0.024, 0.024],
    baseStd: [0.004, 0.009, 0.019, 0.026],
    min: 0.70,
    max: 0.96,
  },
  latency: {
    title: "平均响应延迟",
    suffix: " s",
    ours: [12.54, 16.04, 19.59, 23.88],
    base: [12.16, 16.52, 19.23, 21.38],
    min: 0,
    max: 28,
  },
  p90: {
    title: "P90 响应延迟",
    suffix: " s",
    ours: [24.15, 29.70, 38.10, 46.01],
    base: [23.72, 29.45, 35.44, 39.92],
    min: 0,
    max: 52,
  },
};

const PHASE_LABELS = {
  burst: "逐点爆发",
  dense: "密集服务",
  lull: "沉寂服务",
  charging: "热量积累",
  mid: "过渡阶段",
};

const state = {
  running: false,
  emergency: false,
  loading: true,
  stage: "hard",
  policy: "ours",
  playbackSpeed: 4,
  replayRows: [],
  replaySummary: null,
  replayIndex: 0,
  loadToken: 0,
  error: null,
  elapsed: 0,
  robot: { ...POT_CENTER },
  targetId: -1,
  targetPosition: null,
  steams: [],
  covered: 0,
  spawned: 0,
  missed: 0,
  coverage: 0,
  effectiveCoverage: 0,
  avgLatency: 0,
  p90Latency: 0,
  slaRate: 0,
  oldestAge: 0,
  phase: "加载中",
  reward: 0,
  trail: [],
  trends: [],
  events: [],
  showTrail: true,
};

const dom = {};
let toastTimer = null;

function byId(id) {
  return document.getElementById(id);
}

function clamp(value, min, max) {
  return Math.max(min, Math.min(max, value));
}

function numeric(value, fallback = 0) {
  const parsed = Number(value);
  return Number.isFinite(parsed) ? parsed : fallback;
}

function distance(a, b) {
  return Math.hypot(a.x - b.x, a.y - b.y);
}

function formatDuration(seconds) {
  const minutes = Math.floor(Math.max(seconds, 0) / 60);
  const remaining = Math.floor(Math.max(seconds, 0) % 60);
  return `${String(minutes).padStart(2, "0")}:${String(remaining).padStart(2, "0")}`;
}

function phaseLabel(phase) {
  return PHASE_LABELS[phase] || (phase ? phase : "待机");
}

function percentile(values, percentileValue) {
  if (!values.length) return 0;
  const sorted = [...values].sort((a, b) => a - b);
  const index = (sorted.length - 1) * percentileValue;
  const lower = Math.floor(index);
  const upper = Math.ceil(index);
  if (lower === upper) return sorted[lower];
  return sorted[lower] + (sorted[upper] - sorted[lower]) * (index - lower);
}

function parseCsvLine(line) {
  const values = [];
  let value = "";
  let quoted = false;
  for (let index = 0; index < line.length; index += 1) {
    const character = line[index];
    const next = line[index + 1];
    if (character === '"') {
      if (quoted && next === '"') {
        value += '"';
        index += 1;
      } else {
        quoted = !quoted;
      }
    } else if (character === "," && !quoted) {
      values.push(value);
      value = "";
    } else {
      value += character;
    }
  }
  values.push(value);
  return values;
}

function parseActiveSteams(raw) {
  if (!raw) return [];
  return raw.split(";").filter(Boolean).map((part) => {
    const [id, x, y, ageSteps] = part.split(":");
    return {
      id: numeric(id, -1),
      x: numeric(x),
      y: numeric(y),
      ageSteps: numeric(ageSteps),
      age: numeric(ageSteps) * DECISION_DT,
    };
  }).filter((steam) => steam.id >= 0);
}

function parseTrajectory(csvText) {
  const lines = csvText.replace(/\r/g, "").split("\n").filter((line) => line.length > 0);
  if (lines.length < 2) throw new Error("trajectory.csv 没有可用记录");
  const headers = parseCsvLine(lines[0]);
  const positions = Object.fromEntries(headers.map((header, index) => [header, index]));
  const read = (cells, name) => cells[positions[name]] ?? "";
  return lines.slice(1).map((line) => {
    const cells = parseCsvLine(line);
    return {
      step: numeric(read(cells, "step")),
      reward: numeric(read(cells, "reward")),
      coverX: numeric(read(cells, "cover_x"), POT_CENTER.x),
      coverY: numeric(read(cells, "cover_y"), POT_CENTER.y),
      targetId: numeric(read(cells, "selected_target_id"), -1),
      targetX: numeric(read(cells, "selected_target_x"), POT_CENTER.x),
      targetY: numeric(read(cells, "selected_target_y"), POT_CENTER.y),
      targetRisk: numeric(read(cells, "selected_target_risk_score")),
      coverage: numeric(read(cells, "coverage_rate")),
      effectiveCoverage: numeric(read(cells, "effective_coverage_rate")),
      latency: numeric(read(cells, "cover_latency_seconds")),
      p90: numeric(read(cells, "cover_latency_p90_seconds")),
      slaRate: numeric(read(cells, "response_sla_success_rate")),
      covered: numeric(read(cells, "success_count")),
      spawned: numeric(read(cells, "spawned_count")),
      missed: numeric(read(cells, "missed_count")),
      activeCount: numeric(read(cells, "steam_count")),
      pendingCount: numeric(read(cells, "pending_steam_count")),
      phase: read(cells, "burst_lull_phase"),
      routeConfidence: numeric(read(cells, "route_confidence")),
      routeStagnation: numeric(read(cells, "route_stagnation_score")),
      activeSteams: parseActiveSteams(read(cells, "active_steams")),
    };
  });
}

function addEvent(type, message, time = state.elapsed) {
  state.events.unshift({ type, message, time });
  state.events = state.events.slice(0, 40);
}

function replayDirectory() {
  return `${REPLAY_SOURCES[state.policy].directory}/${STAGES[state.stage].source}`;
}

function currentSourceLabel() {
  return `${REPLAY_SOURCES[state.policy].label} · ${STAGES[state.stage].label}`;
}

function updateLoadingState(message) {
  state.loading = true;
  dom.replaySourceStatus.textContent = message;
  dom.runStateText.textContent = message;
  dom.replayNoticeText.textContent = message;
  dom.startBtn.disabled = true;
  updateInterface();
}

async function loadReplay({ autoplay = false } = {}) {
  const token = ++state.loadToken;
  state.running = false;
  state.emergency = false;
  state.replayRows = [];
  state.replaySummary = null;
  state.replayIndex = 0;
  state.error = null;
  updateLoadingState(`正在载入 ${currentSourceLabel()} 的真实轨迹`);

  try {
    const directory = replayDirectory();
    const [trajectoryResponse, summaryResponse] = await Promise.all([
      fetch(`${directory}/trajectory.csv`, { cache: "no-store" }),
      fetch(`${directory}/summary.json`, { cache: "no-store" }),
    ]);
    if (!trajectoryResponse.ok) throw new Error(`trajectory.csv HTTP ${trajectoryResponse.status}`);
    const csvText = await trajectoryResponse.text();
    const summary = summaryResponse.ok ? await summaryResponse.json() : null;
    if (token !== state.loadToken) return;
    state.replayRows = parseTrajectory(csvText);
    state.replaySummary = summary;
    state.loading = false;
    resetSession({ silent: true });
    dom.replaySourceStatus.textContent = "真实轨迹已载入";
    dom.replayNoticeText.textContent = `${currentSourceLabel()} · 单次 held-out rollout（seed 100），不代表论文均值`;
    addEvent("system", `${currentSourceLabel()} 已载入 · ${state.replayRows.length - 1} steps`);
    updateInterface();
    if (autoplay) startSession();
  } catch (error) {
    if (token !== state.loadToken) return;
    state.loading = false;
    state.error = error.message;
    dom.replaySourceStatus.textContent = "轨迹载入失败";
    dom.replayNoticeText.textContent = `无法载入真实轨迹：${error.message}。请通过 HTTP 端口访问页面。`;
    showToast("真实轨迹载入失败");
    updateInterface();
  }
}

function resetSession({ silent = false } = {}) {
  state.running = false;
  state.emergency = false;
  state.replayIndex = 0;
  state.elapsed = 0;
  state.robot = { ...POT_CENTER };
  state.targetId = -1;
  state.targetPosition = null;
  state.steams = [];
  state.covered = 0;
  state.spawned = 0;
  state.missed = 0;
  state.coverage = 0;
  state.effectiveCoverage = 0;
  state.avgLatency = 0;
  state.p90Latency = 0;
  state.slaRate = 0;
  state.oldestAge = 0;
  state.phase = "待机";
  state.reward = 0;
  state.trail = [];
  state.trends = [];
  state.events = [];
  if (state.replayRows.length) applyReplayRow(state.replayRows[0], { recordEvent: false });
  if (!silent) showToast("真实轨迹已回到起点");
  updateInterface();
}

function applyReplayRow(row, { recordEvent = true } = {}) {
  const previousSteams = state.steams;
  const previousIds = new Set(previousSteams.map((steam) => steam.id));
  const currentIds = new Set(row.activeSteams.map((steam) => steam.id));
  const previousCovered = state.covered;
  const previousPhase = state.phase;

  state.elapsed = row.step * DECISION_DT;
  state.robot = { x: row.coverX, y: row.coverY };
  state.targetId = row.targetId;
  state.targetPosition = row.targetId >= 0 ? { x: row.targetX, y: row.targetY } : null;
  state.steams = row.activeSteams;
  state.covered = row.covered;
  state.spawned = row.spawned;
  state.missed = row.missed;
  state.coverage = row.coverage;
  state.effectiveCoverage = row.effectiveCoverage;
  state.avgLatency = row.latency;
  state.p90Latency = row.p90;
  state.slaRate = row.slaRate;
  state.oldestAge = row.activeSteams.reduce((maxAge, steam) => Math.max(maxAge, steam.age), 0);
  state.phase = phaseLabel(row.phase);
  state.reward = row.reward;

  if (!state.trail.length || distance(state.robot, state.trail[state.trail.length - 1]) > 0.00001) {
    state.trail.push({ ...state.robot });
  }
  if (row.step % 20 === 0 || !state.trends.length) {
    state.trends.push({ time: state.elapsed, coverage: state.coverage, backlog: state.steams.length });
    state.trends = state.trends.slice(-180);
  }

  if (!recordEvent) return;
  for (const steam of row.activeSteams) {
    if (!previousIds.has(steam.id)) {
      addEvent("spawn", `S${String(steam.id).padStart(2, "0")} 出现在真实记录中`);
    }
  }
  const removedIds = [...previousIds].filter((id) => !currentIds.has(id));
  if (row.covered > previousCovered) {
    if (removedIds.length) {
      for (const id of removedIds.slice(0, row.covered - previousCovered)) {
        const oldSteam = previousSteams.find((steam) => steam.id === id);
        addEvent("cover", `S${String(id).padStart(2, "0")} 已覆盖 · ${(oldSteam?.age ?? 0).toFixed(2)} s`);
      }
    } else {
      addEvent("cover", `真实记录新增 ${row.covered - previousCovered} 个覆盖点`);
    }
  }
  if (state.phase !== previousPhase && state.phase !== "待机") {
    addEvent("system", `阶段切换为 ${state.phase}`);
  }
}

function startSession() {
  if (state.loading || !state.replayRows.length) return;
  if (state.replayIndex >= state.replayRows.length - 1) resetSession({ silent: true });
  state.emergency = false;
  state.running = true;
  addEvent("system", "开始播放真实 MuJoCo rollout");
  updateInterface();
}

function pauseSession() {
  if (state.running) addEvent("system", "真实轨迹回放已暂停");
  state.running = false;
  updateInterface();
}

function stopSession() {
  state.running = false;
  state.emergency = true;
  addEvent("alert", "回放已停止；未向机械臂发送动作");
  updateInterface();
  showToast("回放已停止，系统没有输出实机动作");
}

function advanceReplay() {
  if (!state.running || state.loading || !state.replayRows.length) return;
  const rowsPerTick = Math.max(1, Math.round(2 * state.playbackSpeed));
  for (let count = 0; count < rowsPerTick; count += 1) {
    if (state.replayIndex >= state.replayRows.length - 1) break;
    state.replayIndex += 1;
    applyReplayRow(state.replayRows[state.replayIndex]);
  }
  if (state.replayIndex >= state.replayRows.length - 1) {
    state.running = false;
    addEvent("system", "真实 rollout 回放结束");
  }
  updateInterface();
}

function updateInterface() {
  const stage = STAGES[state.stage];
  dom.metricCoverage.textContent = `${(state.coverage * 100).toFixed(1)}%`;
  dom.metricCovered.textContent = String(state.covered);
  dom.metricSpawned.textContent = String(state.spawned);
  dom.metricLatency.textContent = `${state.avgLatency.toFixed(2)} s`;
  dom.metricP90.textContent = `${state.p90Latency.toFixed(2)} s`;
  dom.metricBacklog.textContent = String(state.steams.length);
  dom.metricMaxActive.textContent = String(stage.maxActive);
  dom.metricPhase.textContent = state.loading ? "加载中" : state.emergency ? "已停止" : state.running ? state.phase : state.replayIndex >= state.replayRows.length - 1 ? "已结束" : "已暂停";
  dom.metricSessionTime.textContent = formatDuration(state.elapsed);
  dom.metricSlaNote.textContent = `SLA 4.0 s · 真实记录`;
  dom.queueCount.textContent = String(state.steams.length);
  dom.oldestAge.textContent = `${state.oldestAge.toFixed(2)} s`;
  dom.slaRate.textContent = `${(state.slaRate * 100).toFixed(0)}%`;
  dom.liveIndicator.classList.toggle("is-running", state.running);
  dom.runStateText.textContent = state.loading ? "轨迹加载中" : state.emergency ? "回放已停止" : state.running ? `真实 rollout 播放中 · ${state.playbackSpeed}×` : state.replayIndex >= state.replayRows.length - 1 ? "真实 rollout 已结束" : "真实 rollout 已暂停";
  dom.residualBadge.textContent = `${REPLAY_SOURCES[state.policy].badge} · step ${state.replayIndex}`;
  dom.replaySourceStatus.textContent = state.loading ? "真实轨迹载入中" : state.error ? "轨迹载入失败" : "真实轨迹已载入";
  dom.startBtn.disabled = state.loading || !state.replayRows.length;
  dom.pauseBtn.disabled = !state.running;
  dom.stopBtn.disabled = !state.running;
  renderQueue();
  renderEvents();
  drawWorkspace();
  drawTrendChart();
}

function renderQueue() {
  if (!state.steams.length) {
    dom.queueList.innerHTML = '<div class="empty-state"><span>当前记录没有活动目标</span></div>';
    return;
  }
  const ordered = [...state.steams].sort((a, b) => b.age - a.age);
  dom.queueList.replaceChildren();
  for (const steam of ordered) {
    const row = document.createElement("div");
    const urgent = steam.age >= 3.0;
    row.className = `queue-row${urgent ? " is-urgent" : ""}`;

    const name = document.createElement("div");
    name.className = "target-name";
    const dot = document.createElement("span");
    dot.className = "target-dot";
    const text = document.createElement("div");
    const strong = document.createElement("b");
    strong.textContent = `S${String(steam.id).padStart(2, "0")}`;
    const small = document.createElement("small");
    small.textContent = `x ${steam.x.toFixed(2)} · y ${steam.y.toFixed(2)}`;
    text.append(strong, small);
    name.append(dot, text);

    const age = document.createElement("span");
    age.textContent = `${steam.age.toFixed(2)} s`;
    const status = document.createElement("span");
    status.className = "queue-state";
    status.textContent = steam.id === state.targetId ? "当前" : "活动";
    row.append(name, age, status);
    dom.queueList.append(row);
  }
}

function renderEvents() {
  dom.eventList.replaceChildren();
  for (const event of state.events.slice(0, 12)) {
    const item = document.createElement("li");
    const time = document.createElement("time");
    time.textContent = formatDuration(event.time);
    const type = document.createElement("span");
    const typeClass = event.type === "spawn" ? "spawn" : event.type === "alert" ? "alert" : "system";
    type.className = `event-type event-type--${typeClass}`;
    type.textContent = event.type === "spawn" ? "+" : event.type === "cover" ? "✓" : event.type === "alert" ? "!" : "i";
    const message = document.createElement("span");
    message.textContent = event.message;
    item.append(time, type, message);
    dom.eventList.append(item);
  }
}

function prepareCanvas(canvas) {
  const rect = canvas.getBoundingClientRect();
  if (rect.width <= 1 || rect.height <= 1) return null;
  const ratio = Math.min(window.devicePixelRatio || 1, 2);
  const width = Math.round(rect.width * ratio);
  const height = Math.round(rect.height * ratio);
  if (canvas.width !== width || canvas.height !== height) {
    canvas.width = width;
    canvas.height = height;
  }
  const context = canvas.getContext("2d");
  context.setTransform(ratio, 0, 0, ratio, 0, 0);
  context.clearRect(0, 0, rect.width, rect.height);
  return { context, width: rect.width, height: rect.height };
}

function drawWorkspace() {
  const prepared = prepareCanvas(dom.workspaceCanvas);
  if (!prepared) return;
  const { context: ctx, width, height } = prepared;
  const center = { x: width * 0.50, y: height * 0.50 };
  const radius = Math.min(width * 0.36, height * 0.42);
  const map = (point) => ({
    x: center.x + ((point.x - POT_CENTER.x) / POT_RADIUS) * radius,
    y: center.y - ((point.y - POT_CENTER.y) / POT_RADIUS) * radius,
  });

  ctx.fillStyle = "#f8faf9";
  ctx.fillRect(0, 0, width, height);
  ctx.save();
  ctx.beginPath();
  ctx.arc(center.x, center.y, radius, 0, Math.PI * 2);
  ctx.clip();
  ctx.fillStyle = "#edf2ef";
  ctx.fillRect(center.x - radius, center.y - radius, radius * 2, radius * 2);
  ctx.strokeStyle = "rgba(82, 96, 106, 0.12)";
  ctx.lineWidth = 1;
  for (let index = -4; index <= 4; index += 1) {
    const offset = index * radius / 4;
    ctx.beginPath();
    ctx.moveTo(center.x - radius, center.y + offset);
    ctx.lineTo(center.x + radius, center.y + offset);
    ctx.stroke();
    ctx.beginPath();
    ctx.moveTo(center.x + offset, center.y - radius);
    ctx.lineTo(center.x + offset, center.y + radius);
    ctx.stroke();
  }
  ctx.restore();

  ctx.beginPath();
  ctx.arc(center.x, center.y, radius, 0, Math.PI * 2);
  ctx.strokeStyle = "#6f7e87";
  ctx.lineWidth = 2;
  ctx.stroke();
  ctx.beginPath();
  ctx.arc(center.x, center.y, radius * 0.86, 0, Math.PI * 2);
  ctx.strokeStyle = "rgba(111, 126, 135, 0.34)";
  ctx.setLineDash([6, 6]);
  ctx.stroke();
  ctx.setLineDash([]);

  if (state.showTrail && state.trail.length > 1) {
    ctx.beginPath();
    const first = map(state.trail[0]);
    ctx.moveTo(first.x, first.y);
    for (const trailPoint of state.trail.slice(1)) {
      const point = map(trailPoint);
      ctx.lineTo(point.x, point.y);
    }
    ctx.strokeStyle = "rgba(37, 99, 235, 0.64)";
    ctx.lineWidth = 2;
    ctx.stroke();
  }

  const target = state.steams.find((steam) => steam.id === state.targetId);
  if (target) {
    const robotPoint = map(state.robot);
    const targetPoint = map(target);
    ctx.beginPath();
    ctx.moveTo(robotPoint.x, robotPoint.y);
    ctx.lineTo(targetPoint.x, targetPoint.y);
    ctx.setLineDash([7, 5]);
    ctx.strokeStyle = "rgba(24, 32, 39, 0.48)";
    ctx.lineWidth = 1.4;
    ctx.stroke();
    ctx.setLineDash([]);
  }

  for (const steam of state.steams) {
    const point = map(steam);
    const urgent = steam.age >= 3.0;
    const selected = steam.id === state.targetId;
    ctx.beginPath();
    ctx.arc(point.x, point.y, selected ? 12 : 10, 0, Math.PI * 2);
    ctx.fillStyle = urgent ? "rgba(194, 65, 58, 0.14)" : "rgba(245, 158, 11, 0.16)";
    ctx.fill();
    ctx.beginPath();
    ctx.arc(point.x, point.y, selected ? 6 : 5, 0, Math.PI * 2);
    ctx.fillStyle = urgent ? "#c2413a" : "#f59e0b";
    ctx.fill();
    if (selected) {
      ctx.beginPath();
      ctx.arc(point.x, point.y, 14, 0, Math.PI * 2);
      ctx.strokeStyle = "#182027";
      ctx.lineWidth = 1.5;
      ctx.stroke();
    }
    ctx.fillStyle = "#29333a";
    ctx.font = "700 9px Inter, Arial, sans-serif";
    ctx.fillText(`S${steam.id}`, point.x + 9, point.y - 7);
    ctx.font = "9px Inter, Arial, sans-serif";
    ctx.fillStyle = "#66737c";
    ctx.fillText(`${steam.age.toFixed(1)}s`, point.x + 9, point.y + 5);
  }

  const endpoint = map(state.robot);
  ctx.beginPath();
  ctx.arc(endpoint.x, endpoint.y, 9, 0, Math.PI * 2);
  ctx.fillStyle = "#2563eb";
  ctx.fill();
  ctx.strokeStyle = "#ffffff";
  ctx.lineWidth = 2;
  ctx.stroke();
  ctx.strokeStyle = "#1d4ed8";
  ctx.lineWidth = 1;
  ctx.beginPath();
  ctx.moveTo(endpoint.x - 14, endpoint.y);
  ctx.lineTo(endpoint.x + 14, endpoint.y);
  ctx.moveTo(endpoint.x, endpoint.y - 14);
  ctx.lineTo(endpoint.x, endpoint.y + 14);
  ctx.stroke();

  ctx.fillStyle = "#52606a";
  ctx.font = "11px Inter, Arial, sans-serif";
  ctx.fillText(`stage ${STAGES[state.stage].label} · step ${state.replayIndex}/${Math.max(state.replayRows.length - 1, 0)}`, 16, 22);
  ctx.fillText(`x ${state.robot.x.toFixed(3)} · y ${state.robot.y.toFixed(3)}`, 16, height - 16);
}

function drawTrendChart() {
  const prepared = prepareCanvas(dom.trendCanvas);
  if (!prepared) return;
  const { context: ctx, width, height } = prepared;
  const margin = { left: 38, right: 24, top: 16, bottom: 28 };
  const chartWidth = width - margin.left - margin.right;
  const chartHeight = height - margin.top - margin.bottom;
  ctx.fillStyle = "#ffffff";
  ctx.fillRect(0, 0, width, height);
  ctx.strokeStyle = "#e2e7ea";
  ctx.lineWidth = 1;
  ctx.font = "9px Inter, Arial, sans-serif";
  ctx.fillStyle = "#7b8790";
  for (let index = 0; index <= 4; index += 1) {
    const y = margin.top + chartHeight * index / 4;
    ctx.beginPath();
    ctx.moveTo(margin.left, y);
    ctx.lineTo(width - margin.right, y);
    ctx.stroke();
    ctx.fillText(`${Math.round((1 - index / 4) * 100)}%`, 5, y + 3);
  }
  if (state.trends.length < 2) return;
  const minTime = state.trends[0].time;
  const maxTime = Math.max(state.trends[state.trends.length - 1].time, minTime + 1);
  const xAt = (time) => margin.left + (time - minTime) / (maxTime - minTime) * chartWidth;
  const drawLine = (field, color, scale) => {
    ctx.beginPath();
    state.trends.forEach((entry, index) => {
      const x = xAt(entry.time);
      const y = margin.top + (1 - clamp(entry[field] / scale, 0, 1)) * chartHeight;
      if (index === 0) ctx.moveTo(x, y);
      else ctx.lineTo(x, y);
    });
    ctx.strokeStyle = color;
    ctx.lineWidth = 2;
    ctx.stroke();
  };
  drawLine("coverage", "#2563eb", 1);
  drawLine("backlog", "#b45309", Math.max(STAGES[state.stage].maxActive, 1));
  ctx.fillStyle = "#7b8790";
  ctx.fillText(formatDuration(minTime), margin.left, height - 8);
  const rightLabel = formatDuration(maxTime);
  ctx.fillText(rightLabel, width - margin.right - ctx.measureText(rightLabel).width, height - 8);
}

function drawEvaluationChart() {
  const prepared = prepareCanvas(dom.evaluationCanvas);
  if (!prepared) return;
  const { context: ctx, width, height } = prepared;
  const selected = EVALUATION_DATA[dom.evalMetricSelect.value];
  const stages = ["Low", "Realistic", "Hard", "Extreme"];
  const margin = { left: 52, right: 24, top: 46, bottom: 48 };
  const chartWidth = width - margin.left - margin.right;
  const chartHeight = height - margin.top - margin.bottom;
  const range = selected.max - selected.min;
  dom.evalChartTitle.textContent = selected.title;
  ctx.fillStyle = "#ffffff";
  ctx.fillRect(0, 0, width, height);
  ctx.font = "10px Inter, Arial, sans-serif";
  for (let index = 0; index <= 5; index += 1) {
    const value = selected.min + range * (1 - index / 5);
    const y = margin.top + chartHeight * index / 5;
    ctx.strokeStyle = "#e2e7ea";
    ctx.beginPath();
    ctx.moveTo(margin.left, y);
    ctx.lineTo(width - margin.right, y);
    ctx.stroke();
    ctx.fillStyle = "#7b8790";
    ctx.fillText(selected.suffix ? value.toFixed(0) : value.toFixed(2), 8, y + 3);
  }
  const groupWidth = chartWidth / stages.length;
  const barWidth = Math.min(38, groupWidth * 0.24);
  const yAt = (value) => margin.top + (selected.max - value) / range * chartHeight;
  const baseY = yAt(selected.min);
  stages.forEach((stage, index) => {
    const groupCenter = margin.left + groupWidth * (index + 0.5);
    [selected.base[index], selected.ours[index]].forEach((value, seriesIndex) => {
      const x = groupCenter + (seriesIndex === 0 ? -barWidth - 2 : 2);
      const y = yAt(value);
      ctx.fillStyle = seriesIndex === 0 ? "#737d89" : "#2563eb";
      ctx.fillRect(x, y, barWidth, baseY - y);
      ctx.fillStyle = "#33404a";
      ctx.font = "700 9px Inter, Arial, sans-serif";
      const label = selected.suffix ? value.toFixed(1) : value.toFixed(3);
      ctx.fillText(label, x + (barWidth - ctx.measureText(label).width) / 2, y - 7);
      const stdValues = selected.oursStd && selected.baseStd;
      if (stdValues) {
        const std = seriesIndex === 0 ? selected.baseStd[index] : selected.oursStd[index];
        const errorTop = yAt(Math.min(selected.max, value + std));
        const errorBottom = yAt(Math.max(selected.min, value - std));
        const errorX = x + barWidth / 2;
        ctx.strokeStyle = "#182027";
        ctx.lineWidth = 1.3;
        ctx.beginPath();
        ctx.moveTo(errorX, errorTop);
        ctx.lineTo(errorX, errorBottom);
        ctx.moveTo(errorX - 5, errorTop);
        ctx.lineTo(errorX + 5, errorTop);
        ctx.moveTo(errorX - 5, errorBottom);
        ctx.lineTo(errorX + 5, errorBottom);
        ctx.stroke();
      }
    });
    ctx.fillStyle = "#52606a";
    ctx.font = "10px Inter, Arial, sans-serif";
    ctx.fillText(stage, groupCenter - ctx.measureText(stage).width / 2, height - 17);
  });
  const legendY = 18;
  ctx.fillStyle = "#737d89";
  ctx.fillRect(width - 192, legendY, 13, 8);
  ctx.fillStyle = "#52606a";
  ctx.fillText("Horizon-2", width - 173, legendY + 8);
  ctx.fillStyle = "#2563eb";
  ctx.fillRect(width - 92, legendY, 13, 8);
  ctx.fillStyle = "#52606a";
  ctx.fillText("Ours", width - 73, legendY + 8);
}

function showToast(message) {
  dom.toast.textContent = message;
  dom.toast.classList.add("is-visible");
  window.clearTimeout(toastTimer);
  toastTimer = window.setTimeout(() => dom.toast.classList.remove("is-visible"), 2300);
}

function exportSession() {
  const row = state.replayRows[state.replayIndex] || null;
  const payload = {
    exportedAt: new Date().toISOString(),
    source: currentSourceLabel(),
    stage: state.stage,
    policy: state.policy,
    replayStep: state.replayIndex,
    replaySummary: state.replaySummary,
    currentRecord: row,
    events: state.events,
  };
  const blob = new Blob([JSON.stringify(payload, null, 2)], { type: "application/json" });
  const url = URL.createObjectURL(blob);
  const anchor = document.createElement("a");
  anchor.href = url;
  anchor.download = `v12-rollout-${state.stage}-${state.policy}-seed100.json`;
  anchor.click();
  URL.revokeObjectURL(url);
  showToast("当前真实回放状态已导出");
}

function switchView(view) {
  document.querySelectorAll(".nav-item").forEach((button) => {
    button.classList.toggle("is-active", button.dataset.view === view);
  });
  document.querySelectorAll(".view").forEach((section) => {
    section.classList.toggle("is-active", section.id === `view-${view}`);
  });
  dom.viewTitle.textContent = VIEW_TITLES[view];
  if (view === "experiments") window.requestAnimationFrame(drawEvaluationChart);
  if (view === "live") window.requestAnimationFrame(updateInterface);
}

async function setStage(stage) {
  if (!STAGES[stage]) return;
  state.stage = stage;
  document.querySelectorAll("[data-stage]").forEach((button) => {
    button.classList.toggle("is-selected", button.dataset.stage === stage);
  });
  await loadReplay();
  showToast(`已载入 ${STAGES[stage].label} 的真实 rollout`);
}

async function setPolicy(policy) {
  if (!REPLAY_SOURCES[policy]) return;
  state.policy = policy;
  await loadReplay();
  showToast(`已载入 ${REPLAY_SOURCES[policy].label} 的真实 rollout`);
}

async function openReplayValidation(policy) {
  if (!REPLAY_SOURCES[policy]) return;
  state.policy = policy;
  state.stage = "extreme";
  dom.policySelect.value = policy;
  document.querySelectorAll("[data-stage]").forEach((button) => {
    button.classList.toggle("is-selected", button.dataset.stage === state.stage);
  });
  switchView("live");
  await loadReplay({ autoplay: true });
  showToast(`正在验证 ${REPLAY_SOURCES[policy].label} 的 Extreme 轨迹`);
}

function refreshDiagnostics() {
  const now = new Date().toLocaleTimeString("zh-CN", { hour12: false });
  dom.diagnosticTime.textContent = `最后检测 ${now}`;
  showToast("模块状态检测完成");
}

function bindDom() {
  const ids = {
    viewTitle: "view-title",
    systemClock: "system-clock",
    replaySourceStatus: "replay-source-status",
    replayNoticeText: "replay-notice-text",
    policySelect: "policy-select",
    replaySpeedSelect: "replay-speed-select",
    startBtn: "start-btn",
    pauseBtn: "pause-btn",
    resetBtn: "reset-btn",
    stopBtn: "stop-btn",
    metricCoverage: "metric-coverage",
    metricCovered: "metric-covered",
    metricSpawned: "metric-spawned",
    metricLatency: "metric-latency",
    metricP90: "metric-p90",
    metricBacklog: "metric-backlog",
    metricMaxActive: "metric-max-active",
    metricPhase: "metric-phase",
    metricSessionTime: "metric-session-time",
    metricSlaNote: "metric-sla-note",
    liveIndicator: "live-indicator",
    runStateText: "run-state-text",
    residualBadge: "residual-badge",
    workspaceCanvas: "workspace-canvas",
    trendCanvas: "trend-canvas",
    queueCount: "queue-count",
    queueList: "queue-list",
    oldestAge: "oldest-age",
    slaRate: "sla-rate",
    eventList: "event-list",
    exportBtn: "export-btn",
    centerViewBtn: "center-view-btn",
    trailToggleBtn: "trail-toggle-btn",
    evalMetricSelect: "eval-metric-select",
    evalChartTitle: "eval-chart-title",
    evaluationCanvas: "evaluation-canvas",
    diagnosticRefreshBtn: "diagnostic-refresh-btn",
    diagnosticTime: "diagnostic-time",
    toast: "toast",
  };
  for (const [name, id] of Object.entries(ids)) dom[name] = byId(id);
}

function bindEvents() {
  document.querySelectorAll(".nav-item").forEach((button) => {
    button.addEventListener("click", () => switchView(button.dataset.view));
  });
  document.querySelectorAll("[data-stage]").forEach((button) => {
    button.addEventListener("click", () => setStage(button.dataset.stage));
  });
  document.querySelectorAll("[data-replay-policy]").forEach((button) => {
    button.addEventListener("click", () => openReplayValidation(button.dataset.replayPolicy));
  });
  dom.policySelect.addEventListener("change", () => setPolicy(dom.policySelect.value));
  dom.replaySpeedSelect.addEventListener("change", () => {
    state.playbackSpeed = numeric(dom.replaySpeedSelect.value, 4);
    updateInterface();
  });
  dom.startBtn.addEventListener("click", startSession);
  dom.pauseBtn.addEventListener("click", pauseSession);
  dom.resetBtn.addEventListener("click", () => resetSession());
  dom.stopBtn.addEventListener("click", stopSession);
  dom.exportBtn.addEventListener("click", exportSession);
  dom.centerViewBtn.addEventListener("click", () => {
    drawWorkspace();
    showToast("真实工作区坐标已重新居中");
  });
  dom.trailToggleBtn.addEventListener("click", () => {
    state.showTrail = !state.showTrail;
    dom.trailToggleBtn.classList.toggle("is-selected", state.showTrail);
    drawWorkspace();
  });
  dom.evalMetricSelect.addEventListener("change", drawEvaluationChart);
  dom.diagnosticRefreshBtn.addEventListener("click", refreshDiagnostics);
  window.addEventListener("resize", () => {
    window.requestAnimationFrame(() => {
      drawWorkspace();
      drawTrendChart();
      drawEvaluationChart();
    });
  });
}

function updateClock() {
  dom.systemClock.textContent = new Date().toLocaleTimeString("zh-CN", { hour12: false });
}

function initializeIcons() {
  if (window.lucide) window.lucide.createIcons({ attrs: { "aria-hidden": "true" } });
}

async function initialize() {
  bindDom();
  bindEvents();
  updateClock();
  refreshDiagnostics();
  window.setInterval(updateClock, 1000);
  window.setInterval(advanceReplay, 100);
  window.requestAnimationFrame(() => {
    drawWorkspace();
    drawTrendChart();
    drawEvaluationChart();
    initializeIcons();
  });
  window.setTimeout(initializeIcons, 650);
  await loadReplay();
}

window.addEventListener("lucide-ready", initializeIcons);
document.addEventListener("DOMContentLoaded", initialize);
