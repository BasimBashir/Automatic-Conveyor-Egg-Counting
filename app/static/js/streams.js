const SLOT_COUNT = 10;
const gridView = document.getElementById("gridView");
const detailView = document.getElementById("detailView");
const gridEl = document.getElementById("grid");
const totalCountEl = document.getElementById("totalCount");
const activeCountEl = document.getElementById("activeCount");

const detailTitle = document.getElementById("detailTitle");
const detailFeed = document.getElementById("detailFeed");
const detailPlaceholder = document.getElementById("detailPlaceholder");
const detailEggCount = document.getElementById("detailEggCount");
const detailFps = document.getElementById("detailFps");
const detailUrl = document.getElementById("detailUrl");
const detailFile = document.getElementById("detailFile");
const detailFilename = document.getElementById("detailFilename");
const detailEnabled = document.getElementById("detailEnabled");
const detailCountOnStart = document.getElementById("detailCountOnStart");

const btnBack = document.getElementById("btnBack");
const btnConnect = document.getElementById("btnConnect");
const btnDisconnect = document.getElementById("btnDisconnect");
const btnCountStart = document.getElementById("btnCountStart");
const btnCountStop = document.getElementById("btnCountStop");
const btnReset = document.getElementById("btnReset");
const btnSave = document.getElementById("btnSave");

let currentSlot = null;
let currentDirection = "tb";
let currentSourceType = "rtsp";
let pollHandle = null;

function buildGrid() {
    gridEl.innerHTML = "";
    for (let i = 1; i <= SLOT_COUNT; i++) {
        const tile = document.createElement("div");
        tile.className = "card";
        tile.style.cursor = "pointer";
        tile.dataset.slot = i;
        tile.innerHTML = `
            <div style="display:flex; justify-content:space-between; align-items:center;">
                <strong>Slot ${i}</strong>
                <span class="status-dot inactive" data-status></span>
            </div>
            <img data-feed style="display:none; width:100%; margin-top:8px; border-radius:4px;">
            <p data-empty style="margin-top:8px; font-size:12px; color:var(--text-dim);">Empty</p>
            <div style="display:flex; justify-content:space-between; margin-top:8px; font-size:13px;">
                <span data-count>0</span>
                <span data-fps>0 fps</span>
                <span data-dir></span>
            </div>
        `;
        tile.addEventListener("click", () => openDetail(i));
        gridEl.appendChild(tile);
    }
}

async function refreshGrid() {
    try {
        const resp = await fetch("/api/streams/status");
        const status = await resp.json();
        let total = 0;
        let active = 0;
        for (let i = 1; i <= SLOT_COUNT; i++) {
            const s = status[String(i)] || {};
            const tile = gridEl.querySelector(`[data-slot="${i}"]`);
            const dot = tile.querySelector("[data-status]");
            const feed = tile.querySelector("[data-feed]");
            const empty = tile.querySelector("[data-empty]");
            const countEl = tile.querySelector("[data-count]");
            const fpsEl = tile.querySelector("[data-fps]");
            const dirEl = tile.querySelector("[data-dir]");

            dot.className = "status-dot " + (s.is_connected ? "active" : "inactive");
            if (s.is_connected) {
                feed.style.display = "block";
                if (!feed.src) feed.src = `/api/streams/${i}/feed?t=${Date.now()}`;
                empty.style.display = "none";
                active += 1;
            } else {
                feed.style.display = "none";
                feed.src = "";
                empty.style.display = "block";
                empty.textContent = s.error ? s.error : "Not connected";
            }
            countEl.textContent = s.egg_count || 0;
            fpsEl.textContent = `${s.fps || 0} fps`;
            dirEl.textContent = s.direction === "lr" ? "→" : (s.direction === "tb" ? "↓" : "");
            total += (s.egg_count || 0);
        }
        totalCountEl.textContent = total;
        activeCountEl.textContent = active;
    } catch (_) {}
}

async function openDetail(slot) {
    currentSlot = slot;
    detailTitle.textContent = `Slot ${slot}`;
    const resp = await fetch(`/api/streams/${slot}`);
    const data = await resp.json();
    populateDetail(data);
    gridView.style.display = "none";
    detailView.style.display = "block";

    if (pollHandle) clearInterval(pollHandle);
    pollHandle = setInterval(refreshDetail, 1000);
    refreshDetail();
}

function populateDetail(data) {
    const cfg = data.config;
    if (cfg) {
        currentDirection = cfg.direction || "tb";
        currentSourceType = cfg.source ? cfg.source.type : "rtsp";
        detailEnabled.checked = !!cfg.enabled;
        detailCountOnStart.checked = !!cfg.count_on_start;
        detailUrl.value = (cfg.source && cfg.source.type === "rtsp") ? (cfg.source.url || "") : "";
        detailFilename.textContent = (cfg.source && cfg.source.type === "file") ? (cfg.source.filename || "") : "";
    } else {
        currentDirection = "tb";
        currentSourceType = "rtsp";
        detailEnabled.checked = false;
        detailCountOnStart.checked = false;
        detailUrl.value = "";
        detailFilename.textContent = "";
    }
    document.querySelectorAll("[data-dir]").forEach(b => {
        b.classList.toggle("active", b.dataset.dir === currentDirection);
    });
    document.querySelectorAll("[data-src-tab]").forEach(b => {
        const active = b.dataset.srcTab === currentSourceType;
        b.classList.toggle("active", active);
    });
    document.getElementById("srcRtsp").style.display =
        currentSourceType === "rtsp" ? "block" : "none";
    document.getElementById("srcFile").style.display =
        currentSourceType === "file" ? "block" : "none";
}

async function refreshDetail() {
    if (currentSlot === null) return;
    try {
        const resp = await fetch(`/api/streams/${currentSlot}/status`);
        const s = await resp.json();
        detailEggCount.textContent = s.egg_count || 0;
        detailFps.textContent = s.fps || 0;
        if (s.is_connected) {
            if (!detailFeed.src || detailFeed.style.display === "none") {
                detailFeed.src = `/api/streams/${currentSlot}/feed?t=${Date.now()}`;
            }
            detailFeed.style.display = "block";
            detailPlaceholder.style.display = "none";
        } else {
            detailFeed.style.display = "none";
            detailFeed.src = "";
            detailPlaceholder.style.display = "block";
            detailPlaceholder.textContent = s.error || "Not connected";
        }
    } catch (_) {}
}

document.querySelectorAll("[data-dir]").forEach(btn => {
    btn.addEventListener("click", () => {
        document.querySelectorAll("[data-dir]").forEach(b => b.classList.remove("active"));
        btn.classList.add("active");
        currentDirection = btn.dataset.dir;
    });
});

document.querySelectorAll("[data-src-tab]").forEach(btn => {
    btn.addEventListener("click", () => {
        document.querySelectorAll("[data-src-tab]").forEach(b => b.classList.remove("active"));
        btn.classList.add("active");
        currentSourceType = btn.dataset.srcTab;
        document.getElementById("srcRtsp").style.display =
            currentSourceType === "rtsp" ? "block" : "none";
        document.getElementById("srcFile").style.display =
            currentSourceType === "file" ? "block" : "none";
    });
});

btnSave.addEventListener("click", async () => {
    if (currentSlot === null) return;
    if (currentSourceType === "file" && detailFile.files[0]) {
        const fd = new FormData();
        fd.append("file", detailFile.files[0]);
        await fetch(`/api/streams/${currentSlot}/upload`, { method: "POST", body: fd });
    }
    const patch = {
        direction: currentDirection,
        enabled: detailEnabled.checked,
        count_on_start: detailCountOnStart.checked,
    };
    if (currentSourceType === "rtsp" && detailUrl.value.trim()) {
        patch.source = { type: "rtsp", url: detailUrl.value.trim() };
    }
    await fetch(`/api/streams/${currentSlot}/config`, {
        method: "PATCH",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(patch),
    });
});

btnConnect.addEventListener("click", async () => {
    await fetch(`/api/streams/${currentSlot}/start`, { method: "POST" });
});
btnDisconnect.addEventListener("click", async () => {
    await fetch(`/api/streams/${currentSlot}/stop`, { method: "POST" });
});
btnCountStart.addEventListener("click", async () => {
    await fetch(`/api/streams/${currentSlot}/counting/start`, { method: "POST" });
});
btnCountStop.addEventListener("click", async () => {
    await fetch(`/api/streams/${currentSlot}/counting/stop`, { method: "POST" });
});
btnReset.addEventListener("click", async () => {
    await fetch(`/api/streams/${currentSlot}/reset`, { method: "POST" });
});

btnBack.addEventListener("click", () => {
    if (pollHandle) { clearInterval(pollHandle); pollHandle = null; }
    currentSlot = null;
    detailView.style.display = "none";
    gridView.style.display = "block";
});

buildGrid();
refreshGrid();
setInterval(refreshGrid, 1000);
