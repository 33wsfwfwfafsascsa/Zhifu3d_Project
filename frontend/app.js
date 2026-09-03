const $ = (id) => document.getElementById(id);

const IMPORT_API = "http://127.0.0.1:8000"; // 知识库导入 API（8000，已开 CORS）

let sessionId = localStorage.getItem("zhifu3d_session") || crypto.randomUUID();
localStorage.setItem("zhifu3d_session", sessionId);
$("sessionId").textContent = sessionId.slice(0, 8);

const chatMessages = $("chatMessages");
const agentMessages = $("agentMessages");
let activeAgentSession = null;
let streamingBubble = null;

// 轻量富文本：转义后渲染 **加粗**、## 标题、图片 URL 行；不做完整 Markdown
function renderRich(text) {
  const esc = (s) => s.replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;");
  return String(text).split(/\r?\n/).map((raw) => {
    const line = raw.trim();
    if (!line) return "<br>";
    if (/^https?:\/\/\S+\.(?:jpe?g|png|gif|webp|svg)(?:\?\S*)?$/i.test(line)) {
      return `<img class="md-img" src="${esc(line)}" alt="图片">`;
    }
    let html = esc(line);
    html = html.replace(/\*\*(.+?)\*\*/g, "<b>$1</b>");
    if (/^#{1,3}\s/.test(line)) {
      return `<span class="md-h">${html.replace(/^#{1,3}\s/, "")}</span>`;
    }
    return html;
  }).join("<br>");
}

function appendMessage(container, role, text, extra = null) {
  const div = document.createElement("div");
  div.className = `msg ${role}`;
  if (extra) {
    const meta = document.createElement("span");
    meta.className = "meta";
    meta.textContent = extra;
    div.appendChild(meta);
  }
  const body = document.createElement("span");
  if (role === "assistant") body.innerHTML = renderRich(text);
  else body.textContent = text;
  div.appendChild(body);
  container.appendChild(div);
  container.scrollTop = container.scrollHeight;
  return div;
}

function appendUser(text) {
  appendMessage(chatMessages, "user", text);
}

function startStreamingBubble() {
  streamingBubble = document.createElement("div");
  streamingBubble.className = "msg assistant";
  const status = document.createElement("span");
  status.className = "stream-status";
  status.textContent = "正在处理…";
  const body = document.createElement("span");
  body.id = "streamBody";
  streamingBubble.append(status, body);
  chatMessages.appendChild(streamingBubble);
  chatMessages.scrollTop = chatMessages.scrollHeight;
}

function appendDelta(text) {
  if (!streamingBubble) startStreamingBubble();
  const status = streamingBubble.querySelector(".stream-status");
  if (status) status.remove();
  const body = streamingBubble.querySelector("#streamBody");
  body.dataset.raw = (body.dataset.raw || "") + text;
  body.innerHTML = renderRich(body.dataset.raw);
  chatMessages.scrollTop = chatMessages.scrollHeight;
}

function finalize(payload) {
  if (!streamingBubble) startStreamingBubble();
  const status = streamingBubble.querySelector(".stream-status");
  if (status) status.remove();
  const body = streamingBubble.querySelector("#streamBody");
  if (!body.dataset.raw && payload.answer) {
    body.dataset.raw = payload.answer;
    body.innerHTML = renderRich(payload.answer);
  }
  if (payload.image_urls && payload.image_urls.length) {
    const images = document.createElement("div");
    images.className = "images";
    payload.image_urls.forEach((url) => {
      const img = document.createElement("img");
      img.src = url;
      images.appendChild(img);
    });
    streamingBubble.appendChild(images);
  }
  if (payload.citations && payload.citations.length) {
    const cites = document.createElement("div");
    cites.className = "citations";
    cites.textContent = "引用：" + payload.citations.map((c) => c.title || c.source).filter(Boolean).join("、");
    streamingBubble.appendChild(cites);
  }
  streamingBubble = null;
}

function appendSystem(text) {
  appendMessage(chatMessages, "system", text);
}

function appendOperator(payload) {
  appendMessage(chatMessages, "operator", payload.text, `人工坐席 · ${payload.operator_name}`);
}

function showEscalate(payload) {
  appendSystem(`已转人工：${payload.summary || ""}`);
}

async function fetchHistory() {
  try {
    const res = await fetch(`/api/history/${sessionId}`);
    if (!res.ok) return;
    const items = await res.json();
    chatMessages.innerHTML = "";
    items.forEach((msg) => {
      if (msg.role === "user") appendUser(msg.text);
      else if (msg.role === "operator") appendMessage(chatMessages, "operator", msg.text, "人工坐席");
      else appendMessage(chatMessages, "assistant", msg.text);
    });
  } catch (err) {
    console.error("拉取历史失败", err);
  }
}

let stream = null;
function connectStream() {
  if (stream) stream.close();
  const es = new EventSource(`/api/stream/${sessionId}`);
  stream = es;
  es.addEventListener("ready", fetchHistory);
  es.addEventListener("delta", (e) => appendDelta(JSON.parse(e.data).delta));
  es.addEventListener("progress", (e) => {
    const data = JSON.parse(e.data);
    const status = streamingBubble ? streamingBubble.querySelector(".stream-status") : null;
    if (status && data.label) status.textContent = data.label;
  });
  es.addEventListener("final", (e) => finalize(JSON.parse(e.data)));
  es.addEventListener("escalate", (e) => showEscalate(JSON.parse(e.data)));
  es.addEventListener("operator", (e) => appendOperator(JSON.parse(e.data)));
  es.addEventListener("ticket", (e) => {
    const data = JSON.parse(e.data);
    appendSystem(`已为您创建工单 ${data.ticket_id}，当前状态：${data.status}`);
  });
  es.addEventListener("error", () => {});
}

async function sendChat() {
  const input = $("chatInput");
  const text = input.value.trim();
  if (!text) return;
  appendUser(text);
  input.value = "";
  startStreamingBubble();
  try {
    await fetch("/api/chat", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ message: text, session_id: sessionId, is_stream: true }),
    });
  } catch (err) {
    appendSystem("发送失败：" + err);
  }
}

$("chatSend").addEventListener("click", sendChat);
$("chatInput").addEventListener("keydown", (e) => {
  if (e.key === "Enter") sendChat();
});

// ===== 视图切换 =====
document.querySelectorAll(".tab").forEach((tab) => {
  tab.addEventListener("click", () => {
    document.querySelectorAll(".tab").forEach((t) => t.classList.toggle("active", t === tab));
    document.querySelectorAll(".view").forEach((v) => v.classList.toggle("active", v.id === tab.dataset.view));
  });
});

// ===== 新建会话：换新 sessionId，旧会话保留在 Mongo 可追溯 =====
$("newSession").addEventListener("click", () => {
  sessionId = crypto.randomUUID();
  localStorage.setItem("zhifu3d_session", sessionId);
  $("sessionId").textContent = sessionId.slice(0, 8);
  chatMessages.innerHTML = "";
  streamingBubble = null;
  connectStream();
});

// ===== 坐席工作台 =====
async function pollQueue() {
  try {
    const res = await fetch("/api/agent/queue");
    const items = await res.json();
    $("queueCount").textContent = items.length ? `(${items.length})` : "";
    const list = $("queueList");
    list.innerHTML = "";
    items.forEach((item) => {
      const card = document.createElement("div");
      card.className = "queue-card";
      const reason = document.createElement("div");
      reason.className = "reason";
      reason.textContent = `[${item.escalate_reason || "转人工"}] ${item.session_id.slice(0, 8)}`;
      const summary = document.createElement("div");
      summary.className = "summary";
      summary.textContent = item.summary || item.customer_desc || "";
      const btn = document.createElement("button");
      btn.textContent = "领取";
      btn.addEventListener("click", () => takeSession(item.session_id));
      card.append(reason, summary, btn);
      list.appendChild(card);
    });
  } catch (err) {
    console.error("队列拉取失败", err);
  }
}

async function takeSession(sid) {
  const operatorName = $("operatorName").value.trim() || "坐席A";
  try {
    const res = await fetch(`/api/agent/queue/${sid}/take`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ operator_name: operatorName }),
    });
    if (!res.ok) {
      const err = await res.json();
      alert(err.detail || "领取失败");
      return;
    }
    activeAgentSession = sid;
    await loadAgentDetail(sid);
    pollQueue();
  } catch (err) {
    alert("领取失败：" + err);
  }
}

async function loadAgentDetail(sid) {
  try {
    const res = await fetch(`/api/agent/session/${sid}`);
    const data = await res.json();
    const s = data.session;
    $("agentEmpty").classList.add("hidden");
    $("agentDetail").classList.remove("hidden");
    const entities = s.entities || {};
    $("ctxContent").innerHTML = `
      <div class="row"><b>状态</b>${s.status || ""}（${s.operator_name || "未认领"}）</div>
      <div class="row"><b>原因</b>${s.escalate_reason || ""}</div>
      <div class="row"><b>机型</b>${(s.models || []).join("、") || "—"}</div>
      <div class="row"><b>订单</b>${(entities.order_ids || []).join("、") || "—"}</div>
      <div class="row"><b>摘要</b>${s.summary || "—"}</div>`;
    const draft = s.ticket_draft || {};
    $("ticketCategory").value = draft.category || "";
    $("ticketModel").value = draft.model || "";
    $("ticketSummary").value = draft.summary || "";
    agentMessages.innerHTML = "";
    data.messages.forEach((msg) => {
      const label = { user: "用户", assistant: "机器人", operator: "坐席" }[msg.role];
      appendMessage(agentMessages, msg.role || "assistant", msg.text, label);
    });
  } catch (err) {
    console.error("加载会话详情失败", err);
  }
}

async function sendReply() {
  if (!activeAgentSession) return;
  const input = $("agentInput");
  const text = input.value.trim();
  if (!text) return;
  const operatorName = $("operatorName").value.trim() || "坐席A";
  try {
    const res = await fetch("/api/agent/reply", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ session_id: activeAgentSession, operator_name: operatorName, text }),
    });
    if (!res.ok) {
      const err = await res.json();
      alert(err.detail || "回复失败");
      return;
    }
    appendMessage(agentMessages, "operator", text, "坐席");
    input.value = "";
  } catch (err) {
    alert("回复失败：" + err);
  }
}

async function submitTicket() {
  if (!activeAgentSession) return;
  try {
    const res = await fetch("/api/agent/tickets", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        session_id: activeAgentSession,
        category: $("ticketCategory").value.trim(),
        model: $("ticketModel").value.trim(),
        summary: $("ticketSummary").value.trim(),
      }),
    });
    if (!res.ok) {
      const err = await res.json();
      alert(err.detail || "提交失败");
      return;
    }
    const ticket = await res.json();
    appendMessage(agentMessages, "system", `工单已创建：${ticket.ticket_id}（${ticket.status}）`);
  } catch (err) {
    alert("提交失败：" + err);
  }
}

async function closeSession() {
  if (!activeAgentSession) return;
  await fetch(`/api/agent/session/${activeAgentSession}/close`, { method: "POST" });
  activeAgentSession = null;
  $("agentDetail").classList.add("hidden");
  $("agentEmpty").classList.remove("hidden");
  pollQueue();
}

$("agentSend").addEventListener("click", sendReply);
$("agentInput").addEventListener("keydown", (e) => {
  if (e.key === "Enter") sendReply();
});
$("ticketSubmit").addEventListener("click", submitTicket);
$("sessionClose").addEventListener("click", closeSession);

// ===== 知识库导入（调用 8000 import_service） =====
const dropZone = $("dropZone");
const fileInput = $("fileInput");
let importTimer = null;

function setFile(file) {
  $("fileName").textContent = file ? file.name : "";
  $("uploadBtn").disabled = !file;
}

dropZone.addEventListener("click", () => fileInput.click());
dropZone.addEventListener("dragover", (e) => { e.preventDefault(); dropZone.classList.add("drag"); });
dropZone.addEventListener("dragleave", () => dropZone.classList.remove("drag"));
dropZone.addEventListener("drop", (e) => {
  e.preventDefault();
  dropZone.classList.remove("drag");
  const f = e.dataTransfer.files && e.dataTransfer.files[0];
  if (f) { fileInput.files = e.dataTransfer.files; setFile(f); }
});
fileInput.addEventListener("change", () => setFile(fileInput.files[0]));

async function loadCatalog() {
  try {
    const res = await fetch(`${IMPORT_API}/api/catalog`);
    if (!res.ok) return;
    const data = await res.json();
    const sel = $("productModel");
    (data.models || []).forEach((m) => {
      const opt = document.createElement("option");
      opt.value = m;
      opt.textContent = m;
      sel.appendChild(opt);
    });
  } catch (err) {
    console.error("机型目录拉取失败（默认仅 general）", err);
  }
}

function renderTask(s) {
  $("taskStatus").textContent = s.status;
  $("taskStatus").className = "task-status " + (s.status || "");
  const nodes = [...(s.done_list || [])].map((n) => `<span class="node done">${n}</span>`)
    .concat([...(s.running_list || [])].map((n) => `<span class="node running">${n}</span>`));
  $("taskNodes").innerHTML = nodes.join("");
  $("taskError").textContent = s.error || "";
}

async function pollTask(taskId) {
  try {
    const res = await fetch(`${IMPORT_API}/api/status/${taskId}`);
    if (!res.ok) return;
    const s = await res.json();
    renderTask(s);
    if (s.status === "completed" || s.status === "failed") {
      clearInterval(importTimer);
      importTimer = null;
    }
  } catch (err) {
    console.error("进度查询失败", err);
  }
}

$("uploadBtn").addEventListener("click", async () => {
  const file = fileInput.files[0];
  if (!file) return;
  const form = new FormData();
  form.append("file", file);
  form.append("knowledge_type", $("knowledgeType").value);
  form.append("product_model", $("productModel").value);
  $("uploadBtn").disabled = true;
  try {
    const res = await fetch(`${IMPORT_API}/api/upload`, { method: "POST", body: form });
    if (!res.ok) {
      const err = await res.json();
      throw new Error(err.detail || `上传失败（HTTP ${res.status}）`);
    }
    const data = await res.json();
    $("taskCard").classList.remove("hidden");
    $("taskError").textContent = "";
    renderTask({ status: "pending", done_list: [], running_list: [], error: "" });
    pollTask(data.task_id);
    importTimer = setInterval(() => pollTask(data.task_id), 1500);
  } catch (err) {
    $("taskCard").classList.remove("hidden");
    $("taskStatus").textContent = "failed";
    $("taskStatus").className = "task-status failed";
    $("taskError").textContent = String(err.message || err);
  } finally {
    $("uploadBtn").disabled = !fileInput.files[0];
  }
});

loadCatalog();
connectStream();
pollQueue();
setInterval(pollQueue, 3000);
