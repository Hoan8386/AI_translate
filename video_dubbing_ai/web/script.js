/**
 * AI Dubbing System — Dashboard JavaScript
 * ==========================================
 * Tab switching, Upload, Monitor, Step Viewer, WebSocket realtime.
 */

// ============================================
// Configuration
// ============================================
const API_BASE = window.location.origin;
const WS_BASE = `ws://${window.location.host}`;

// Stage metadata
const STAGE_META = {
    1:  { name: 'Video Processor',  badge: null },
    2:  { name: 'Audio Extractor',  badge: null },
    3:  { name: 'Speaker Detector', badge: 'gpu' },
    4:  { name: 'Segment Creator',  badge: null },
    5:  { name: 'Chinese ASR',      badge: 'gpu' },
    6:  { name: 'Translation',      badge: 'api' },
    7:  { name: 'Voice Cloning',    badge: 'gpu' },
    8:  { name: 'Audio Alignment',  badge: null },
    9:  { name: 'Lip Sync',         badge: 'gpu' },
    10: { name: 'Video Renderer',   badge: null },
};

// ============================================
// State
// ============================================
const state = {
    selectedFile: null,
    currentJobId: null,
    isProcessing: false,
    ws: null,
    wsRetryCount: 0,
    maxWsRetries: 5,
    currentTab: 'home',
    // Step Viewer
    stageDataMap: {},      // {stageNum: {...data}} from WS
    stageTimings: {},      // {stageNum: {duration_s: ...}}
    completedStages: new Set(),
    activeStageInViewer: null,
};

// ============================================
// Tab Switching
// ============================================
const SECTIONS = {
    home:    ['hero', 'features-section'],
    upload:  ['upload'],
    monitor: ['monitor'],
    steps:   ['steps'],
};

function switchTab(tab) {
    state.currentTab = tab;

    // Update nav links
    document.querySelectorAll('.nav-link').forEach(el => el.classList.remove('active'));
    const navEl = document.getElementById(`nav-${tab}`);
    if (navEl) navEl.classList.add('active');

    // Show/hide sections
    const allSections = document.querySelectorAll('.hero, .features-section, .upload-section, .monitor-section, .steps-section');
    allSections.forEach(s => s.style.display = 'none');

    if (tab === 'home') {
        document.querySelector('.hero').style.display = '';
        document.querySelector('.features-section').style.display = '';
    } else if (tab === 'upload') {
        document.querySelector('.upload-section').style.display = '';
    } else if (tab === 'monitor') {
        document.querySelector('.monitor-section').style.display = '';
    } else if (tab === 'steps') {
        document.querySelector('.steps-section').style.display = '';
        stepViewer.refresh();
    }

    window.scrollTo({ top: 0, behavior: 'smooth' });
}

// Legacy helper for button onclick
function scrollToSection(id) {
    switchTab(id);
}

// ============================================
// Upload Handling
// ============================================
const uploadArea = document.getElementById('uploadArea');
const fileInput = document.getElementById('fileInput');
const optionsPanel = document.getElementById('optionsPanel');
const uploadProgress = document.getElementById('uploadProgress');

uploadArea.addEventListener('click', () => { fileInput.click(); });

fileInput.addEventListener('change', (e) => {
    if (e.target.files.length > 0) handleFile(e.target.files[0]);
});

uploadArea.addEventListener('dragover', (e) => {
    e.preventDefault();
    uploadArea.classList.add('drag-over');
});
uploadArea.addEventListener('dragleave', () => uploadArea.classList.remove('drag-over'));
uploadArea.addEventListener('drop', (e) => {
    e.preventDefault();
    uploadArea.classList.remove('drag-over');
    if (e.dataTransfer.files.length > 0) handleFile(e.dataTransfer.files[0]);
});

function handleFile(file) {
    const ext = file.name.split('.').pop().toLowerCase();
    const validExts = ['mp4', 'mkv', 'avi', 'mov', 'webm', 'flv'];
    if (!validExts.includes(ext)) {
        addLog('error', `File không hợp lệ: ${file.name}. Hỗ trợ: MP4, MKV, AVI, MOV, WebM`);
        return;
    }
    if (file.size > 500 * 1024 * 1024) {
        addLog('error', `File quá lớn: ${(file.size / (1024*1024)).toFixed(0)}MB. Giới hạn: 500MB`);
        return;
    }
    state.selectedFile = file;
    uploadFile(file);
}

async function uploadFile(file) {
    uploadArea.style.display = 'none';
    uploadProgress.style.display = 'block';

    const formData = new FormData();
    formData.append('file', file);

    try {
        const xhr = new XMLHttpRequest();
        xhr.upload.addEventListener('progress', (e) => {
            if (e.lengthComputable) {
                const pct = Math.round((e.loaded / e.total) * 100);
                document.getElementById('uploadProgressFill').style.width = `${pct}%`;
                document.getElementById('uploadProgressText').textContent =
                    `Đang upload... ${pct}% (${(e.loaded/(1024*1024)).toFixed(1)}/${(e.total/(1024*1024)).toFixed(1)} MB)`;
            }
        });

        xhr.onload = () => {
            if (xhr.status === 200) {
                const data = JSON.parse(xhr.responseText);
                state.currentJobId = data.job_id;
                uploadProgress.style.display = 'none';
                optionsPanel.style.display = 'block';
                document.getElementById('selectedFileName').textContent =
                    `${file.name} (${(file.size/(1024*1024)).toFixed(1)} MB) — Job: ${data.job_id}`;
                addLog('success', `Upload thành công: ${file.name} → Job ID: ${data.job_id}`);
            } else {
                const err = JSON.parse(xhr.responseText);
                addLog('error', `Upload thất bại: ${err.detail}`);
                resetUploadUI();
            }
        };

        xhr.onerror = () => {
            addLog('error', 'Kết nối thất bại khi upload');
            resetUploadUI();
        };

        xhr.open('POST', `${API_BASE}/api/upload`);
        xhr.send(formData);
    } catch (err) {
        addLog('error', `Upload error: ${err.message}`);
        resetUploadUI();
    }
}

function resetUploadUI() {
    uploadProgress.style.display = 'none';
    uploadArea.style.display = 'flex';
}

function clearUpload() {
    state.selectedFile = null;
    state.currentJobId = null;
    fileInput.value = '';
    uploadArea.style.display = 'flex';
    uploadProgress.style.display = 'none';
    optionsPanel.style.display = 'none';
}

// ============================================
// Start Processing
// ============================================
async function startProcessing() {
    if (!state.currentJobId || state.isProcessing) return;

    const skipLipsync = !document.getElementById('lipSyncToggle').checked;
    const translationProvider = document.getElementById('translationApi').value;

    try {
        const res = await fetch(
            `${API_BASE}/api/jobs/${state.currentJobId}/start?skip_lipsync=${skipLipsync}&translation_provider=${translationProvider}`,
            { method: 'POST' }
        );
        const data = await res.json();

        if (!res.ok) {
            addLog('error', `Không thể bắt đầu: ${data.detail}`);
            return;
        }

        state.isProcessing = true;
        document.getElementById('startBtn').disabled = true;
        document.getElementById('startBtn').textContent = 'Đang xử lý...';

        // Reset step viewer state
        state.stageDataMap = {};
        state.stageTimings = {};
        state.completedStages = new Set();
        state.activeStageInViewer = null;

        // Update nav badge
        document.getElementById('stepsNavBadge').style.display = 'inline-flex';
        document.getElementById('stepsNavBadge').textContent = '0';

        // Navigate to monitor
        switchTab('monitor');

        // Connect WebSocket
        connectWebSocket(state.currentJobId);

        // Init Step Viewer
        stepViewer.init(state.currentJobId);

        addLog('info', `Pipeline bắt đầu cho job: ${state.currentJobId}`);
        setMonitorBadge('PROCESSING', 'badge-processing');
    } catch (err) {
        addLog('error', `Lỗi kết nối: ${err.message}`);
    }
}

// ============================================
// WebSocket
// ============================================
function connectWebSocket(jobId) {
    if (state.ws) {
        state.ws.close();
    }

    const wsUrl = `${WS_BASE}/ws/jobs/${jobId}`;
    state.ws = new WebSocket(wsUrl);

    state.ws.onopen = () => {
        state.wsRetryCount = 0;
        setConnectionStatus(true);
        addLog('info', 'WebSocket kết nối thành công');

        // Ping interval
        state._pingInterval = setInterval(() => {
            if (state.ws && state.ws.readyState === WebSocket.OPEN) {
                state.ws.send('ping');
            }
        }, 25000);
    };

    state.ws.onmessage = (event) => {
        try {
            const data = JSON.parse(event.data);
            handleWebSocketMessage(data);
        } catch (e) {
            console.error('WS parse error:', e);
        }
    };

    state.ws.onclose = () => {
        setConnectionStatus(false);
        clearInterval(state._pingInterval);
        if (!state.isProcessing) return;
        if (state.wsRetryCount < state.maxWsRetries) {
            state.wsRetryCount++;
            setTimeout(() => connectWebSocket(jobId), 2000 * state.wsRetryCount);
        }
    };

    state.ws.onerror = () => {
        setConnectionStatus(false);
    };
}

function handleWebSocketMessage(data) {
    if (data.type === 'pong') return;

    if (data.type === 'progress') {
        updateProgress(data.stage, data.stage_name, data.progress);
        if (data.log) addLog(data.log.type, data.log.message);
        if (data.segments) updateSegmentsTable(data.segments);
        if (data.vram) updateVRAM(data.vram.used, data.vram.model);
        // Step Viewer update
        if (data.stage_data) {
            const { stage, name, data: sData } = data.stage_data;
            stepViewer.onStageComplete(stage, name, sData);
        }
        if (data.timing && data.stage) {
            state.stageTimings[data.stage] = data.timing;
            stepViewer.updateSidebarTiming(data.stage, data.timing);
        }
    }

    if (data.type === 'status') {
        if (data.log) addLog('info', data.message || data.log.message);

        if (data.status === 'completed') {
            state.isProcessing = false;
            setMonitorBadge('COMPLETED', 'badge-success');
            showResultPanel(data.output_path || '');
            addLog('success', '🎉 DUBBING HOÀN THÀNH!');
            document.getElementById('stepsNavBadge').textContent = '✓';

            // Auto-switch to step viewer after completion
            setTimeout(() => switchTab('steps'), 1500);

            // Fetch output files
            if (state.currentJobId) stepViewer.fetchOutputFiles();
        }

        if (data.status === 'failed') {
            state.isProcessing = false;
            setMonitorBadge('FAILED', 'badge-error');
            addLog('error', `Pipeline thất bại: ${data.error}`);
        }

        if (data.status === 'processing') {
            setMonitorBadge('PROCESSING', 'badge-processing');
        }
    }
}

// ============================================
// Monitor UI Helpers
// ============================================
function updateProgress(stage, stageName, progress) {
    const ring = document.getElementById('progressRing');
    const val = document.getElementById('progressValue');
    const stageEl = document.getElementById('progressStage');

    const circumference = 2 * Math.PI * 60;
    const offset = circumference - (progress / 100) * circumference;

    if (ring) { ring.style.strokeDashoffset = offset; ring.setAttribute('stroke', 'url(#progressGrad)'); }
    if (val) val.textContent = Math.round(progress);
    if (stageEl) stageEl.textContent = stageName ? `Stage ${stage}: ${stageName}` : 'Processing...';

    // Update sidebar stages
    document.querySelectorAll('.stage-item').forEach(el => {
        const s = parseInt(el.dataset.stage);
        el.classList.remove('stage-active', 'stage-done');
        const dot = el.querySelector('.stage-dot');
        if (s < stage) {
            el.classList.add('stage-done');
            if (dot) dot.innerHTML = '✓';
        } else if (s === stage) {
            el.classList.add('stage-active');
            if (dot) dot.innerHTML = '◉';
        } else {
            if (dot) dot.innerHTML = '';
        }
    });

    // Mark stage as currently running in step viewer sidebar
    stepViewer.markRunning(stage, stageName);
}

function updateVRAM(usedMb, modelName) {
    const pct = Math.min((usedMb / 8192) * 100, 100);
    document.getElementById('vramFill').style.width = `${pct}%`;
    document.getElementById('vramUsed').textContent = usedMb > 0 ? `${usedMb} MB` : '0 MB';
    document.getElementById('vramModel').textContent = modelName || 'Không có model trên GPU';

    const fill = document.getElementById('vramFill');
    if (pct > 80) fill.style.background = 'linear-gradient(90deg, #f59e0b, #ef4444)';
    else fill.style.background = '';
}

function updateSegmentsTable(segments) {
    const panel = document.getElementById('segmentsPanel');
    const tbody = document.getElementById('segmentsBody');
    if (!segments || segments.length === 0) return;
    panel.style.display = 'block';
    tbody.innerHTML = segments.map(s => `
        <tr>
            <td>${s.id || '-'}</td>
            <td><span class="speaker-tag">${s.speaker || '-'}</span></td>
            <td class="time-cell">${formatTime(s.start)} → ${formatTime(s.end)}</td>
            <td class="zh-text">${s.zh_text || '-'}</td>
            <td class="vi-text">${s.vi_text || '-'}</td>
        </tr>
    `).join('');
}

function showResultPanel(outputPath) {
    const panel = document.getElementById('resultPanel');
    panel.style.display = 'block';
    if (outputPath) {
        const fname = outputPath.split(/[\\/]/).pop();
        document.getElementById('resultFilename').textContent = fname;
        const downloadBtn = document.getElementById('downloadBtn');
        downloadBtn.href = `${API_BASE}/api/jobs/${state.currentJobId}/download`;
    }
}

function addLog(type, message) {
    const container = document.getElementById('logContainer');
    if (!container) return;
    const entry = document.createElement('div');
    entry.className = `log-entry log-${type}`;
    const now = new Date();
    const time = `${String(now.getHours()).padStart(2,'0')}:${String(now.getMinutes()).padStart(2,'0')}:${String(now.getSeconds()).padStart(2,'0')}`;
    entry.innerHTML = `<span class="log-time">${time}</span><span class="log-msg">${message}</span>`;
    container.appendChild(entry);
    container.scrollTop = container.scrollHeight;
    // Max 200 log entries
    while (container.children.length > 200) container.removeChild(container.firstChild);
}

function setConnectionStatus(connected) {
    const el = document.getElementById('connectionStatus');
    const dot = el.querySelector('.conn-dot');
    const text = el.querySelector('.conn-text');
    dot.className = `conn-dot ${connected ? 'conn-connected' : 'conn-disconnected'}`;
    text.textContent = connected ? 'Live' : 'Offline';
}

function setMonitorBadge(text, cls) {
    const el = document.getElementById('monitorBadge');
    el.textContent = text;
    el.className = `section-badge ${cls}`;
}

function formatTime(secs) {
    if (secs == null) return '--';
    const m = Math.floor(secs / 60);
    const s = (secs % 60).toFixed(1);
    return `${m}:${String(s).padStart(4, '0')}`;
}

// ============================================
// Step Viewer Class
// ============================================
const stepViewer = {
    jobId: null,

    init(jobId) {
        this.jobId = jobId;
        state.stageDataMap = {};
        state.stageTimings = {};
        state.completedStages = new Set();
        state.activeStageInViewer = null;

        document.getElementById('stepsEmpty').style.display = 'none';
        document.getElementById('stepsViewer').style.display = 'grid';
        document.getElementById('outputSummary').style.display = 'none';

        document.getElementById('stepsJobId').textContent = `Job: ${jobId}`;
        document.getElementById('stepsBadge').textContent = 'RUNNING';
        document.getElementById('stepsBadge').className = 'section-badge badge-processing';

        this.renderSidebar();
        this.showPlaceholder();
    },

    refresh() {
        if (!this.jobId) {
            document.getElementById('stepsEmpty').style.display = 'block';
            document.getElementById('stepsViewer').style.display = 'none';
            return;
        }
        // Already initialized
        this.renderSidebar();
    },

    renderSidebar() {
        const list = document.getElementById('stepsSidebarList');
        list.innerHTML = Object.entries(STAGE_META).map(([num, meta]) => {
            const n = parseInt(num);
            const completed = state.completedStages.has(n);
            const timing = state.stageTimings[n];
            const timingStr = timing ? `${timing.duration_s}s` : '';
            const badgeHtml = meta.badge
                ? `<span class="step-sb-badge ${meta.badge}">${meta.badge.toUpperCase()}</span>` : '';
            return `
            <div class="step-sidebar-item ${completed ? 'completed' : ''}"
                 id="step-sb-${n}"
                 onclick="stepViewer.selectStage(${n})">
                <span class="step-sb-num">${String(n).padStart(2,'0')}</span>
                <div class="step-sb-info">
                    <div class="step-sb-name">${meta.name}</div>
                    ${timingStr ? `<div class="step-sb-timing">⏱ ${timingStr}</div>` : ''}
                </div>
                ${badgeHtml}
                <div class="step-sb-status" id="step-sb-status-${n}">
                    ${completed ? '✓' : ''}
                </div>
            </div>`;
        }).join('');
    },

    markRunning(stageNum, stageName) {
        // Reset all
        document.querySelectorAll('.step-sidebar-item').forEach(el => {
            el.classList.remove('active-running');
        });
        const el = document.getElementById(`step-sb-${stageNum}`);
        if (el && !state.completedStages.has(stageNum)) {
            el.classList.add('active-running');
            const status = document.getElementById(`step-sb-status-${stageNum}`);
            if (status) status.innerHTML = '<svg class="spin-anim" width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5"><path d="M12 2v4M12 18v4M4.93 4.93l2.83 2.83M16.24 16.24l2.83 2.83M2 12h4M18 12h4M4.93 19.07l2.83-2.83M16.24 7.76l2.83-2.83"/></svg>';
        }
    },

    onStageComplete(stageNum, stageName, data) {
        state.completedStages.add(stageNum);
        state.stageDataMap[stageNum] = { name: stageName, data };

        // Update sidebar item
        const el = document.getElementById(`step-sb-${stageNum}`);
        if (el) {
            el.classList.remove('active-running');
            el.classList.add('completed');
        }
        const status = document.getElementById(`step-sb-status-${stageNum}`);
        if (status) status.innerHTML = '✓';

        // Update nav badge
        const badge = document.getElementById('stepsNavBadge');
        if (badge) {
            badge.style.display = 'inline-flex';
            badge.textContent = state.completedStages.size;
        }

        // Auto-select if in steps tab and nothing selected yet
        if (state.currentTab === 'steps' && !state.activeStageInViewer) {
            this.selectStage(stageNum);
        }
        // If currently viewing this stage, refresh detail
        if (state.activeStageInViewer === stageNum) {
            this.renderDetail(stageNum);
        }
    },

    updateSidebarTiming(stageNum, timing) {
        state.stageTimings[stageNum] = timing;
        const el = document.getElementById(`step-sb-${stageNum}`);
        if (el) {
            const infoEl = el.querySelector('.step-sb-info');
            if (infoEl) {
                let timingEl = infoEl.querySelector('.step-sb-timing');
                if (!timingEl) {
                    timingEl = document.createElement('div');
                    timingEl.className = 'step-sb-timing';
                    infoEl.appendChild(timingEl);
                }
                timingEl.textContent = `⏱ ${timing.duration_s}s`;
            }
        }
        // Update overall timing
        const total = Object.values(state.stageTimings).reduce((s, t) => s + (t.duration_s || 0), 0);
        document.getElementById('stepsOverallTiming').textContent = total > 0 ? `Total: ${total.toFixed(1)}s` : '';
    },

    selectStage(stageNum) {
        state.activeStageInViewer = stageNum;

        // Update sidebar active state
        document.querySelectorAll('.step-sidebar-item').forEach(el => el.classList.remove('active'));
        const el = document.getElementById(`step-sb-${stageNum}`);
        if (el) el.classList.add('active');

        this.renderDetail(stageNum);
    },

    renderDetail(stageNum) {
        const placeholder = document.getElementById('stepsPlaceholder');
        const content = document.getElementById('stepsDetailContent');

        const meta = STAGE_META[stageNum] || { name: `Stage ${stageNum}` };
        const stageInfo = state.stageDataMap[stageNum];
        const timing = state.stageTimings[stageNum];

        // If stage not completed yet
        if (!stageInfo) {
            placeholder.style.display = 'flex';
            content.style.display = 'none';
            return;
        }

        placeholder.style.display = 'none';
        content.style.display = 'flex';

        // Header
        document.getElementById('detailStageNum').textContent = String(stageNum).padStart(2, '0');
        document.getElementById('detailStageName').textContent = meta.name;

        const statusEl = document.getElementById('detailStatus');
        statusEl.textContent = 'completed';
        statusEl.className = 'stage-detail-status completed';

        document.getElementById('detailTimingVal').textContent =
            timing ? `${timing.duration_s}s` : '-';

        // Data cards
        this.renderDataCards(stageNum, stageInfo.data || {});

        // Segments (for ASR/Translation stages)
        const segs = stageInfo.data?.segments;
        if (segs && segs.length > 0) {
            this.renderSegments(segs, stageNum);
            document.getElementById('accordionSegments').style.display = 'block';
        } else {
            document.getElementById('accordionSegments').style.display = 'none';
        }

        // Files (from API or from data)
        this.renderFiles(stageNum, stageInfo.data || {});

        // Raw JSON
        const jsonPre = document.getElementById('jsonPre');
        jsonPre.innerHTML = syntaxHighlightJson(JSON.stringify(stageInfo.data || {}, null, 2));
    },

    renderDataCards(stageNum, data) {
        const container = document.getElementById('detailDataContent');
        const cards = [];

        // Generic key-value render (skip segments arrays which go in their own panel)
        const skipKeys = new Set(['segments', 'segments_preview', 'diarization']);

        for (const [key, val] of Object.entries(data)) {
            if (skipKeys.has(key)) continue;
            if (typeof val === 'object' && val !== null) {
                // Nested object: render as sub-cards
                if (key === 'video_info') {
                    cards.push(renderVideoInfo(val));
                    continue;
                }
                continue;
            }

            let valClass = '';
            let valStr = String(val ?? '-');

            if (key === 'num_segments' || key === 'num_speakers' || key === 'num_cloned') valClass = 'accent';
            else if (key === 'duration_s' || key.includes('path') || key.includes('dir')) valClass = 'mono';
            else if (typeof val === 'boolean') { valClass = val ? 'success' : 'error'; }

            cards.push(`
                <div class="data-card">
                    <div class="data-card-key">${key.replace(/_/g, ' ')}</div>
                    <div class="data-card-val ${valClass}" title="${valStr}">${truncate(valStr, 40)}</div>
                </div>
            `);
        }

        // Speaker list
        if (data.speakers && Array.isArray(data.speakers)) {
            cards.push(`
                <div class="data-card">
                    <div class="data-card-key">Speakers</div>
                    <div class="data-card-val accent">${data.speakers.join(', ') || '-'}</div>
                </div>
            `);
        }

        const count = cards.length;
        document.getElementById('dataCount').textContent = count;
        container.innerHTML = `<div class="data-cards">${cards.join('')}</div>`;
    },

    renderSegments(segs, stageNum) {
        const container = document.getElementById('detailSegmentsContent');
        document.getElementById('segmentsCount').textContent = segs.length;

        const showZh = segs.some(s => s.zh_text);
        const showVi = segs.some(s => s.vi_text);

        container.innerHTML = `
            <div class="segments-mini-table">
                <table>
                    <thead>
                        <tr>
                            <th>#</th>
                            <th>Speaker</th>
                            <th>Time</th>
                            ${showZh ? '<th>中文</th>' : ''}
                            ${showVi ? '<th>Tiếng Việt</th>' : ''}
                        </tr>
                    </thead>
                    <tbody>
                        ${segs.slice(0, 100).map(s => `
                            <tr>
                                <td>${s.id ?? '-'}</td>
                                <td><span class="seg-speaker-tag">${s.speaker ?? '-'}</span></td>
                                <td class="seg-time">${formatTime(s.start)} → ${formatTime(s.end)}</td>
                                ${showZh ? `<td class="seg-text-zh" title="${s.zh_text||''}">${truncate(s.zh_text||'-', 30)}</td>` : ''}
                                ${showVi ? `<td class="seg-text-vi" title="${s.vi_text||''}">${truncate(s.vi_text||'-', 30)}</td>` : ''}
                            </tr>
                        `).join('')}
                    </tbody>
                </table>
                ${segs.length > 100 ? `<p class="detail-empty">... và ${segs.length - 100} segments nữa</p>` : ''}
            </div>
        `;
    },

    renderFiles(stageNum, data) {
        const container = document.getElementById('detailFilesContent');
        const files = [];

        // Try to infer files from stage data
        const pathKeys = Object.keys(data).filter(k =>
            (k.includes('path') || k.includes('dir') || k.includes('json')) &&
            typeof data[k] === 'string' && data[k].length > 0
        );

        pathKeys.forEach(key => {
            const p = data[key];
            const name = p.split(/[\\/]/).pop();
            const ext = name.split('.').pop().toLowerCase();
            files.push({ name, ext, key, path: p });
        });

        document.getElementById('filesCount').textContent = files.length;

        if (files.length === 0) {
            container.innerHTML = '<p class="detail-empty">Chưa có files output</p>';
            return;
        }

        container.innerHTML = `<div class="file-list">
            ${files.map(f => `
                <div class="file-item">
                    <div class="file-icon ${getFileIconClass(f.ext)}">${f.ext}</div>
                    <div class="file-info">
                        <div class="file-name" title="${f.path}">${f.name}</div>
                        <div class="file-size">${f.key.replace(/_/g, ' ')}</div>
                    </div>
                </div>
            `).join('')}
        </div>`;
    },

    async fetchOutputFiles() {
        if (!this.jobId) return;
        try {
            const res = await fetch(`${API_BASE}/api/jobs/${this.jobId}/output-files`);
            if (!res.ok) return;
            const data = await res.json();
            if (!data.files || data.files.length === 0) return;

            const list = document.getElementById('outputFilesList');
            const summary = document.getElementById('outputSummary');

            list.innerHTML = data.files.map(f => `
                <div class="output-file-chip">
                    <div class="file-icon ${getFileIconClass(f.type)}">${f.type}</div>
                    <span class="chip-name" title="${f.name}">${f.name}</span>
                    <span class="chip-size">${f.size_mb} MB</span>
                </div>
            `).join('');

            summary.style.display = 'block';
        } catch (e) {
            console.warn('fetchOutputFiles error:', e);
        }
    },

    showPlaceholder() {
        document.getElementById('stepsPlaceholder').style.display = 'flex';
        document.getElementById('stepsDetailContent').style.display = 'none';
    },
};

// ============================================
// Accordion
// ============================================
function toggleAccordion(name) {
    const bodyId = {
        data: 'accordionBodyData',
        segments: 'accordionBodySegments',
        files: 'accordionBodyFiles',
        json: 'accordionBodyJson',
    }[name];

    if (!bodyId) return;
    const body = document.getElementById(bodyId);
    if (!body) return;
    body.classList.toggle('open');

    // Rotate chevron
    const header = body.previousElementSibling;
    if (header) {
        const chevron = header.querySelector('.accordion-chevron');
        if (chevron) {
            chevron.style.transform = body.classList.contains('open') ? 'rotate(180deg)' : '';
        }
    }
}

// ============================================
// Utility Helpers
// ============================================
function truncate(str, maxLen) {
    if (!str) return '-';
    return str.length > maxLen ? str.slice(0, maxLen) + '…' : str;
}

function getFileIconClass(ext) {
    const map = { mp4: 'mp4', mkv: 'mp4', wav: 'wav', mp3: 'wav', json: 'json', txt: 'txt' };
    return map[ext] || 'other';
}

function renderVideoInfo(info) {
    const items = [
        ['Resolution', `${info.width}×${info.height}`],
        ['FPS', info.fps],
        ['Codec', info.video_codec],
        ['Duration', `${info.duration?.toFixed(1)}s`],
        ['Size', `${info.size_mb?.toFixed(1)} MB`],
        ['Audio', info.audio_codec || '-'],
    ];
    return items.map(([k, v]) => `
        <div class="data-card">
            <div class="data-card-key">${k}</div>
            <div class="data-card-val">${v || '-'}</div>
        </div>
    `).join('');
}

function syntaxHighlightJson(json) {
    return json
        .replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;')
        .replace(/("(\\u[a-zA-Z0-9]{4}|\\[^u]|[^\\"])*"(\s*:)?|\b(true|false|null)\b|-?\d+(?:\.\d*)?(?:[eE][+\-]?\d+)?)/g,
            (match) => {
                let cls = 'json-number';
                if (/^"/.test(match)) {
                    cls = /:$/.test(match) ? 'json-key' : 'json-string';
                } else if (/true|false/.test(match)) {
                    cls = 'json-bool';
                } else if (/null/.test(match)) {
                    cls = 'json-null';
                }
                return `<span class="${cls}">${match}</span>`;
            }
        );
}

// ============================================
// Init on Load
// ============================================
document.addEventListener('DOMContentLoaded', () => {
    // Show hero by default
    switchTab('home');

    // Check connection
    fetch(`${API_BASE}/api/jobs`)
        .then(r => r.ok ? setConnectionStatus(true) : null)
        .catch(() => {});

    // Restore active job from local storage
    const savedJobId = localStorage.getItem('activeJobId');
    if (savedJobId) {
        fetch(`${API_BASE}/api/jobs/${savedJobId}`)
            .then(r => r.json())
            .then(data => {
                if (data.id && (data.status === 'processing' || data.status === 'completed')) {
                    state.currentJobId = savedJobId;
                    stepViewer.init(savedJobId);
                    if (data.status === 'processing') {
                        state.isProcessing = true;
                        connectWebSocket(savedJobId);
                    }
                    // Load existing stage data
                    if (data.stage_data) {
                        Object.entries(data.stage_data).forEach(([num, sd]) => {
                            const n = parseInt(num);
                            state.stageDataMap[n] = sd;
                            state.completedStages.add(n);
                        });
                        if (data.stage_timings) {
                            Object.entries(data.stage_timings).forEach(([num, t]) => {
                                state.stageTimings[parseInt(num)] = t;
                            });
                        }
                        stepViewer.renderSidebar();
                    }
                }
            }).catch(() => {});
    }
});

// Save job id to local storage when set
const _origStart = startProcessing;
window.startProcessing = async function() {
    await _origStart();
    if (state.currentJobId) {
        localStorage.setItem('activeJobId', state.currentJobId);
    }
};
