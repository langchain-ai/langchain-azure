const elements = {
  form: document.querySelector("#run-form"),
  reviewHeading: document.querySelector("#review-heading"),
  prompt: document.querySelector("#prompt"),
  runButton: document.querySelector("#run-button"),
  runStatus: document.querySelector("#run-status"),
  statusDot: document.querySelector("#status-dot"),
  handoffCount: document.querySelector("#handoff-count"),
  elapsed: document.querySelector("#elapsed"),
  sourceFiles: document.querySelector("#source-files"),
  sourceAccount: document.querySelector("#source-account"),
  sourceRoute: document.querySelector("#source-route"),
  outputFiles: document.querySelector("#output-files"),
  outputAccount: document.querySelector("#output-account"),
  outputRoute: document.querySelector("#output-route"),
  modelName: document.querySelector("#model-name"),
  packetMap: document.querySelector("#packet-map"),
  transferLayer: document.querySelector("#transfer-layer"),
  eventCount: document.querySelector("#event-count"),
  timeline: document.querySelector("#timeline"),
  directionBadge: document.querySelector("#direction-badge"),
  operationDetail: document.querySelector("#operation-detail"),
  decisionSection: document.querySelector("#decision-section"),
  decisionStatus: document.querySelector("#decision-status"),
  decisionContent: document.querySelector("#decision-content"),
  previewDialog: document.querySelector("#preview-dialog"),
  previewKind: document.querySelector("#preview-kind"),
  previewTitle: document.querySelector("#preview-title"),
  previewContent: document.querySelector("#preview-content"),
  closePreview: document.querySelector("#close-preview"),
};

const expectedOutputs = [
  "01-packet-index.json",
  "02-classification.json",
  "03-extracted-facts.json",
  "04-underwriting-decision.md",
];

let timer;
let startedAt;
let eventCount = 0;
let handoffs = 0;
let runCompleted = false;
let activeTransfer = null;
let transferSettled = false;
let decisionVerified = false;

function setRunState(label, status) {
  elements.runStatus.textContent = label;
  elements.statusDot.className = `status-dot ${status}`;
  document.body.dataset.runState = status;
}

function setDecisionState(state, label, placeholder) {
  elements.decisionSection.dataset.state = state;
  elements.decisionStatus.className = `decision-status ${state}`;
  elements.decisionStatus.textContent = label;
  if (placeholder === undefined) return;
  elements.decisionContent.replaceChildren();
  const message = document.createElement("p");
  message.className = "decision-placeholder";
  message.textContent = placeholder;
  elements.decisionContent.append(message);
}

function formatElapsed(milliseconds) {
  const seconds = Math.max(0, Math.floor(milliseconds / 1000));
  return `${Math.floor(seconds / 60).toString().padStart(2, "0")}:${(seconds % 60).toString().padStart(2, "0")}`;
}

function startTimer() {
  window.clearInterval(timer);
  startedAt = Date.now();
  elements.elapsed.textContent = "00:00";
  timer = window.setInterval(() => {
    elements.elapsed.textContent = formatElapsed(Date.now() - startedAt);
  }, 500);
}

function stopTimer() {
  window.clearInterval(timer);
  timer = undefined;
  if (startedAt) elements.elapsed.textContent = formatElapsed(Date.now() - startedAt);
}

function fileExtension(path) {
  const name = path.split("/").pop() || "file";
  return (name.includes(".") ? name.split(".").pop() : "file").slice(0, 4);
}

function formatBytes(value) {
  const bytes = Number(value || 0);
  if (bytes < 1024) return `${bytes} B`;
  return `${(bytes / 1024).toFixed(1)} KB`;
}

function encodedPath(path, root) {
  return path
    .replace(new RegExp(`^/${root}/`), "")
    .split("/")
    .map(encodeURIComponent)
    .join("/");
}

async function openPreview(kind, path, url) {
  elements.previewKind.textContent = `${kind} Blob preview`;
  elements.previewTitle.textContent = path;
  elements.previewContent.textContent = "Loading from Blob Storage...";
  elements.previewDialog.showModal();
  try {
    const response = await fetch(url);
    const payload = await response.json();
    if (!response.ok) throw new Error(payload.detail || "Preview failed");
    elements.previewContent.textContent = payload.content;
  } catch (error) {
    elements.previewContent.textContent = `Preview unavailable: ${error.message}`;
  }
}

function buildFileRow(name, path, meta) {
  const type = document.createElement("span");
  type.className = "file-type";
  type.textContent = fileExtension(name);
  const detail = document.createElement("span");
  detail.className = "file-detail";
  const fileName = document.createElement("span");
  fileName.className = "file-name";
  fileName.textContent = name;
  const filePath = document.createElement("span");
  filePath.className = "file-path";
  filePath.textContent = path;
  const fileMeta = document.createElement("span");
  fileMeta.className = "file-meta";
  fileMeta.textContent = meta;
  detail.append(fileName, filePath, fileMeta);
  return [type, detail];
}

function renderSourceFiles(files) {
  elements.sourceFiles.replaceChildren();
  files.forEach((file) => {
    const row = document.createElement("button");
    row.type = "button";
    row.className = "file-row";
    row.dataset.path = file.virtualPath;
    row.append(...buildFileRow(
      file.virtualPath.split("/").pop(),
      file.blobName,
      `${formatBytes(file.size)} | click to preview`,
    ));
    row.addEventListener("click", () => openPreview(
      "Source",
      file.virtualPath,
      `/api/source/${encodedPath(file.virtualPath, "source")}`,
    ));
    elements.sourceFiles.append(row);
  });
}

function renderOutputTargets() {
  elements.outputFiles.replaceChildren();
  expectedOutputs.forEach((name) => {
    const row = document.createElement("button");
    row.type = "button";
    row.className = "artifact-row";
    row.dataset.name = name;
    row.dataset.path = `/output/${name}`;
    row.disabled = true;
    row.append(...buildFileRow(name, "Awaiting coordinator", "pending"));
    elements.outputFiles.append(row);
  });
}

function renderArtifact(artifact) {
  const row = elements.outputFiles.querySelector(`[data-name="${CSS.escape(artifact.name)}"]`);
  if (!row) return;
  row.disabled = false;
  row.classList.add("verified");
  row.querySelector(".file-path").textContent = artifact.blob_name;
  row.querySelector(".file-meta").textContent = `${formatBytes(new Blob([artifact.content]).size)} | verified, click to preview`;
  row.addEventListener("click", () => openPreview(
    "Output",
    artifact.virtual_path,
    `/api/output/${encodeURIComponent(artifact.run_id)}/${encodedPath(artifact.virtual_path, "output")}`,
  ));
}

function markPipeline(status) {
  document.querySelectorAll(".pipeline-agent").forEach((node) => {
    node.classList.remove("active", "complete");
    if (status) node.classList.add(status);
  });
}

function markAgent(agent, status) {
  document.querySelectorAll(".pipeline-agent").forEach((node) => {
    if (node.dataset.agent !== agent) return;
    node.classList.remove("active", "complete");
    if (status) node.classList.add(status);
  });
}

function transferEndpoints(event) {
  const agent = elements.packetMap.querySelector(`[data-agent="${CSS.escape(event.agent)}"]`);
  const file = elements.packetMap.querySelector(`[data-path="${CSS.escape(event.path)}"]`);
  if (!agent || !file) return null;
  return event.direction === "write" ? [agent, file] : [file, agent];
}

function clearTransferPath() {
  elements.transferLayer
    .querySelectorAll(".transfer-path, .transfer-particle")
    .forEach((node) => node.remove());
  document
    .querySelectorAll(".active-transfer")
    .forEach((node) => node.classList.remove("active-transfer"));
}

function edgeMidpoint(rect, target, mapBox) {
  const center = {
    x: rect.left + rect.width / 2 - mapBox.left,
    y: rect.top + rect.height / 2 - mapBox.top,
  };
  const deltaX = target.x - center.x;
  const deltaY = target.y - center.y;
  if (Math.abs(deltaX) >= Math.abs(deltaY)) {
    return {
      x: (deltaX >= 0 ? rect.right : rect.left) - mapBox.left,
      y: center.y,
    };
  }
  return {
    x: center.x,
    y: (deltaY >= 0 ? rect.bottom : rect.top) - mapBox.top,
  };
}

function settleTransferPath() {
  elements.transferLayer.querySelector(".transfer-path")?.classList.add("settled");
  elements.transferLayer
    .querySelectorAll(".transfer-particle")
    .forEach((node) => node.remove());
}

function drawTransferPath(event) {
  clearTransferPath();
  const endpoints = transferEndpoints(event);
  if (!endpoints || window.matchMedia("(max-width: 760px)").matches) return;

  const [origin, destination] = endpoints;
  origin.classList.add("active-transfer");
  destination.classList.add("active-transfer");
  const mapBox = elements.packetMap.getBoundingClientRect();
  const originBox = origin.getBoundingClientRect();
  const destinationBox = destination.getBoundingClientRect();
  const originCenter = {
    x: originBox.left + originBox.width / 2 - mapBox.left,
    y: originBox.top + originBox.height / 2 - mapBox.top,
  };
  const destinationCenter = {
    x: destinationBox.left + destinationBox.width / 2 - mapBox.left,
    y: destinationBox.top + destinationBox.height / 2 - mapBox.top,
  };
  const start = edgeMidpoint(originBox, destinationCenter, mapBox);
  const end = edgeMidpoint(destinationBox, originCenter, mapBox);
  const deltaX = end.x - start.x;
  const deltaY = end.y - start.y;
  const distance = Math.hypot(deltaX, deltaY);
  if (distance < 1) return;
  const normalX = -deltaY / distance;
  const normalY = deltaX / distance;
  const bow = Math.min(32, distance * 0.08) * (deltaY >= 0 ? -1 : 1);
  const firstControl = {
    x: start.x + deltaX * 0.34 + normalX * bow,
    y: start.y + deltaY * 0.34 + normalY * bow,
  };
  const secondControl = {
    x: start.x + deltaX * 0.68 + normalX * bow,
    y: start.y + deltaY * 0.68 + normalY * bow,
  };
  const pathData = `M ${start.x} ${start.y} C ${firstControl.x} ${firstControl.y}, ${secondControl.x} ${secondControl.y}, ${end.x} ${end.y}`;
  const namespace = "http://www.w3.org/2000/svg";
  const path = document.createElementNS(namespace, "path");
  path.setAttribute("d", pathData);
  path.setAttribute("class", `transfer-path ${event.direction}`);
  const particle = document.createElementNS(namespace, "polygon");
  particle.setAttribute("points", "-7,-4 7,0 -7,4");
  particle.setAttribute("class", "transfer-particle");
  particle.setAttribute(
    "fill",
    event.direction === "write" ? "var(--green)" : "var(--cyan)",
  );
  const movement = document.createElementNS(namespace, "animateMotion");
  movement.setAttribute("dur", "1.2s");
  movement.setAttribute("repeatCount", "indefinite");
  movement.setAttribute("rotate", "auto");
  movement.setAttribute("path", pathData);
  particle.append(movement);
  elements.transferLayer.append(path, particle);
  if (transferSettled) settleTransferPath();
}

function appendTimeline(category, agent, text, timestamp) {
  elements.timeline.querySelector(".timeline-empty")?.remove();
  eventCount += 1;
  elements.eventCount.textContent = `${eventCount} steps`;
  const item = document.createElement("li");
  item.className = `timeline-event ${category}`;
  const meta = document.createElement("div");
  meta.className = "event-meta";
  const agentName = document.createElement("span");
  agentName.className = "event-agent";
  agentName.textContent = agent;
  const time = document.createElement("time");
  time.dateTime = timestamp;
  time.textContent = new Date(timestamp).toLocaleTimeString([], { hour12: false });
  meta.append(agentName, time);
  const message = document.createElement("p");
  message.className = "event-message";
  message.textContent = text;
  item.append(meta, message);
  elements.timeline.append(item);
  elements.timeline.scrollTop = elements.timeline.scrollHeight;
}

function renderOperation(event) {
  activeTransfer = event;
  elements.directionBadge.className = `direction-badge ${event.direction}`;
  elements.directionBadge.textContent = event.direction.toUpperCase();
  elements.operationDetail.replaceChildren();
  const route = document.createElement("p");
  route.className = "transfer-route";
  route.textContent = `${event.agent} ${event.direction === "write" ? "writes to" : "reads from"} ${event.path}`;
  const tool = document.createElement("p");
  tool.className = "transfer-path-label";
  tool.textContent = event.tool;
  elements.operationDetail.append(route, tool);
  if (event.path.startsWith("/source/")) {
    elements.sourceFiles
      .querySelector(`[data-path="${CSS.escape(event.path)}"]`)
      ?.classList.add("accessed");
  }
  drawTransferPath(event);
}

function resetLiveState() {
  eventCount = 0;
  handoffs = 0;
  runCompleted = false;
  activeTransfer = null;
  transferSettled = false;
  decisionVerified = false;
  elements.eventCount.textContent = "0 steps";
  elements.handoffCount.textContent = "0 / 4 handoffs";
  elements.timeline.replaceChildren();
  const empty = document.createElement("li");
  empty.className = "timeline-empty";
  empty.textContent = "Waiting for the coordinator.";
  elements.timeline.append(empty);
  elements.directionBadge.className = "direction-badge idle";
  elements.directionBadge.textContent = "Waiting";
  elements.operationDetail.replaceChildren();
  const operation = document.createElement("p");
  operation.className = "empty-state";
  operation.textContent = "Waiting for a specialist filesystem operation.";
  elements.operationDetail.append(operation);
  markPipeline();
  clearTransferPath();
  setDecisionState(
    "processing",
    "Processing",
    "Underwriting will begin after intake and extraction artifacts are ready.",
  );
}

function renderMarkdown(markdown) {
  elements.decisionContent.replaceChildren();
  markdown.split(/\r?\n/).forEach((line) => {
    if (!line.trim()) return;
    const heading = line.match(/^(#{1,3})\s+(.+)$/);
    const node = document.createElement(heading ? `h${heading[1].length}` : "p");
    node.textContent = heading ? heading[2] : line.replace(/^\s*-\s+/, "");
    elements.decisionContent.append(node);
  });
}

async function loadConfig() {
  try {
    const response = await fetch("/api/config");
    const config = await response.json();
    if (!response.ok) throw new Error(config.detail || "Configuration unavailable");
    const accountName = config.account
      ? new URL(config.account).hostname.split(".")[0]
      : "Azurite";
    elements.modelName.textContent = config.model;
    elements.reviewHeading.textContent = `Packet ${config.packetId}`;
    elements.prompt.value = `Process packet ${config.packetId} through all four specialist stages. Persist each stage output and return a cited underwriting decision.`;
    elements.sourceAccount.textContent = accountName;
    elements.sourceRoute.textContent = config.source;
    elements.outputAccount.textContent = accountName;
    elements.outputRoute.textContent = config.outputs;
    renderSourceFiles(config.files);
    setRunState("Ready", "idle");
  } catch (error) {
    setRunState("Configuration error", "failed");
    elements.modelName.textContent = error.message;
    elements.runButton.disabled = true;
  }
}

function handleEvent(event) {
  if (event.type === "run.started") {
    markAgent("orchestrator", "active");
    appendTimeline("delegation", "orchestrator", "Started mortgage packet processing.", event.timestamp);
  } else if (event.type === "delegation.started") {
    markAgent(event.targetAgent, "active");
    appendTimeline("delegation", event.agent, `Delegated to ${event.targetAgent}: ${event.summary}`, event.timestamp);
  } else if (event.type === "filesystem.started") {
    markAgent(event.agent, "active");
    renderOperation(event);
    appendTimeline(event.direction, event.agent, `${event.tool} ${event.path}`, event.timestamp);
  } else if (event.type === "handoff.completed") {
    handoffs = event.handoff;
    elements.handoffCount.textContent = `${handoffs} / 4 handoffs`;
    markAgent(event.agent, "complete");
    settleTransferPath();
    appendTimeline("complete", event.agent, `Completed handoff ${handoffs} of 4.`, event.timestamp);
  } else if (event.type === "delegation.failed") {
    markAgent(event.agent);
    appendTimeline("error", event.agent, event.error, event.timestamp);
  } else if (event.type === "artifact.verified") {
    renderArtifact(event.artifact);
    if (event.artifact.name === "04-underwriting-decision.md") {
      decisionVerified = true;
      renderMarkdown(event.artifact.content);
      setDecisionState("verified", "Verified");
    }
    appendTimeline("write", "server", `Verified ${event.artifact.name} in Blob Storage.`, event.timestamp);
  } else if (event.type === "run.completed") {
    runCompleted = true;
    transferSettled = true;
    markPipeline("complete");
    if (!decisionVerified) {
      renderMarkdown(event.result.decision.content);
      setDecisionState("verified", "Verified");
    }
    setRunState("Completed", "complete");
    stopTimer();
    settleTransferPath();
    appendTimeline("complete", "orchestrator", "All required artifacts are verified.", event.timestamp);
  } else if (event.type === "run.failed") {
    transferSettled = true;
    setRunState(event.error, "failed");
    stopTimer();
    settleTransferPath();
    if (!decisionVerified) {
      setDecisionState(
        "failed",
        "Not verified",
        "No verified underwriting decision was produced for this run.",
      );
    }
    appendTimeline("error", "server", event.error, event.timestamp);
  }
}

elements.form.addEventListener("submit", (event) => {
  event.preventDefault();
  const prompt = elements.prompt.value.trim();
  if (!prompt) return;

  resetLiveState();
  renderOutputTargets();
  elements.runButton.disabled = true;
  elements.runButton.textContent = "Processing...";
  setRunState("Processing", "running");
  markPipeline("active");
  startTimer();

  const protocol = window.location.protocol === "https:" ? "wss:" : "ws:";
  const socket = new WebSocket(`${protocol}//${window.location.host}/ws/runs`);
  socket.addEventListener("open", () => socket.send(JSON.stringify({ prompt })));
  socket.addEventListener("message", (message) => handleEvent(JSON.parse(message.data)));
  socket.addEventListener("error", () => {
    if (!runCompleted) {
      setRunState("Connection error", "failed");
      stopTimer();
    }
  });
  socket.addEventListener("close", () => {
    stopTimer();
    elements.runButton.disabled = false;
    elements.runButton.textContent = "Process packet";
    if (!runCompleted && !elements.statusDot.classList.contains("failed")) {
      setRunState("Disconnected", "failed");
      stopTimer();
    }
  });
});

elements.closePreview.addEventListener("click", () => elements.previewDialog.close());
elements.previewDialog.addEventListener("click", (event) => {
  if (event.target === elements.previewDialog) elements.previewDialog.close();
});

const resizeObserver = new ResizeObserver(() => {
  if (activeTransfer) drawTransferPath(activeTransfer);
});
resizeObserver.observe(elements.packetMap);

renderOutputTargets();
loadConfig();